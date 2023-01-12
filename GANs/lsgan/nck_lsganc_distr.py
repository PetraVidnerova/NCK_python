import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.datasets.mnist import load_data

#conda install scikit-image
from skimage.transform import resize
from scipy.linalg import sqrtm
from PIL import Image

FIDmodel = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

RES = 64
assert (RES == 64 or RES == 128)
NOISE_DIM = 100; COND_DIM = 6
BUFFER_SIZE = 60000
BATCH_SIZE = 32
FID_BATCH = 1000

IMGS_DIR = f'lsganc_images{RES}'
if not os.path.isdir(IMGS_DIR):
  os.mkdir(IMGS_DIR)
print(os.path.abspath(IMGS_DIR))

CKPT_DIR = f'./{IMGS_DIR}/training_checkpoints'
RESTORE_CKPT = False

def _extract_fn(tfrecord, res=RES):
  ximg = res #64, 128
  yimg = int(0.75*ximg) #48, 96
  # Extract features
  features = {
    'fpath': tf.io.FixedLenFeature([1], tf.string),
    'image': tf.io.FixedLenFeature([ximg * yimg], tf.int64),
    'label': tf.io.FixedLenFeature([6], tf.float32)
  }
  # Extract the data record
  sample = tf.io.parse_single_example(tfrecord, features)
  fpath = sample['fpath']
  image = sample['image']
  label = sample['label']

  fpath = tf.cast(fpath, tf.string)

  image = tf.reshape(image, [yimg, ximg, 1])
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1

  coords = tf.cast(label, 'float32')
  attrs = coords
  attrs = tf.broadcast_to(attrs, shape=(yimg, ximg, 6))
  return image, attrs

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0).astype('float32')
		images_list.append(new_image)
	return np.asarray(images_list)

def calculate_fid(model, act1, images2):
	act2 = model.predict(images2)
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	ssdiff = np.sum((mu1 - mu2)**2.0)
	covmean = sqrtm(sigma1.dot(sigma2))
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def get_icdf():
  pddata = pd.read_csv('datacoords.txt', header=None)
  data = pddata.to_numpy()
  a = data.min(axis=0)
  b = data.max(axis=0)
  k = 2/(b-a); q = (-a-b)/(b-a) #[a,b]->[-1,1]
  data = k*data+q
  #---
  from statsmodels.distributions.empirical_distribution import ECDF
  lps = []
  for i in range(6):
    x = data[:, i]
    ecdf = ECDF(x)
    f = ecdf(x)
    l = np.polynomial.legendre.legfit(f, x, deg=10)
    y = np.polynomial.legendre.legval(f, l)
    lps.append(l)
  return k, q, lps

# Dataset
xres = RES; yres=int(0.75*RES); dstype = 'all'
tfrecord_file = f"E:\\NCK\\gan_{xres}{yres}{dstype}.tfrecord"
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(_extract_fn)
dataset = dataset.repeat()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(4)
train_dataset = dataset

fiddataset = tf.data.TFRecordDataset(tfrecord_file)
fiddataset = fiddataset.map(_extract_fn)
fiddataset = fiddataset.repeat()
fiddataset = fiddataset.batch(FID_BATCH)

print("FID preparing real activations ...")
for item in fiddataset.take(1):
  images_real, _ = item
images_real = (images_real + 1.) * 127.5
images_real = tf.concat((images_real,)*3, axis=3)
images_real = scale_images(images_real.numpy(), (299, 299, 3))
images_real = preprocess_input(images_real)
act_real = FIDmodel.predict(images_real)
del images_real
print("Done reals.")

# 64x48 conditioned Generator
def make_cgenerator_model64(z_dim, cond_dim):
  gf_dim = 64
  model = tf.keras.Sequential()

  model.add(layers.Dense(gf_dim * 8 * 3 * 4, input_shape=(z_dim + cond_dim,)))
  model.add(layers.Reshape((3, 4, gf_dim * 8)))  # (4, 3, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((3, 4, gf_dim * 8)))  # (4, 4, 512)
  assert model.output_shape == (None, 3, 4, 512)  # Note: None is the batch size

  model.add(layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 6, 8, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 12, 16, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(gf_dim * 1, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 24, 32, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
  assert model.output_shape == (None, 48, 64, 1)

  return model

# 128x96 conditioned Generator
def make_cgenerator_model128(z_dim, cond_dim):
  gf_dim = 64
  model = tf.keras.Sequential()

  model.add(layers.Dense(gf_dim * 8 * 3 * 4, input_shape=(z_dim + cond_dim,)))
  model.add(layers.Reshape((3, 4, gf_dim * 8)))  # (3, 4, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((3, 4, gf_dim * 8)))  # (3, 4, 512)
  assert model.output_shape == (None, 3, 4, 512)

  model.add(layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 6, 8, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 12, 16, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(gf_dim * 1, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 24, 32, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
  assert model.output_shape == (None, 48, 64, 32)

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
  assert model.output_shape == (None, 96, 128, 1)

  return model

if RES == 64:
  generator = make_cgenerator_model64(NOISE_DIM, COND_DIM)
elif RES == 128:
  generator = make_cgenerator_model128(NOISE_DIM, COND_DIM)

noise = tf.random.normal([1, NOISE_DIM])
attrs = tf.random.uniform([1, 6], -1, 1)
input = tf.concat([noise, attrs], axis=-1)
generated_image = generator(input, training=False)
#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#print(generator.summary())

# 64x48 conditioned Discriminator
def make_cdiscriminator_model64(cond_dim):
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(48, 64, 1 + cond_dim)))
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  image = layers.Input(shape=(48, 64, 1))
  attrs = layers.Input(shape=(48, 64, cond_dim))

  c_input = tf.concat([image, attrs], axis=-1)
  output = model(c_input)

  Model = tf.keras.Model([image, attrs], output)

  return Model

def make_cdiscriminator_model128(cond_dim):
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(96, 128, 1 + cond_dim)))
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  image = layers.Input(shape=(96, 128, 1))
  attrs = layers.Input(shape=(96, 128, cond_dim))

  c_input = tf.concat([image, attrs], axis=-1)
  output = model(c_input)

  Model = tf.keras.Model([image, attrs], output)

  return Model

if RES == 64:
  discriminator = make_cdiscriminator_model64(COND_DIM)
elif RES == 128:
  discriminator = make_cdiscriminator_model128(COND_DIM)

attrs = tf.random.uniform((COND_DIM,), -1, 1)
#print(attrs.numpy())
if RES == 64:
  attrs = tf.broadcast_to(attrs, shape=(48, 64, COND_DIM))
else:
  attrs = tf.broadcast_to(attrs, shape=(96, 128, COND_DIM))
attrs = tf.expand_dims(attrs, axis=0)
decision = discriminator([generated_image, attrs])
#print(decision)
#print(discriminator.summary())

# Define loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Checkpoint setting
if not os.path.isdir(CKPT_DIR):
  os.mkdir(CKPT_DIR)
checkpoint_prefix = os.path.join(CKPT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Define the training loop
EPOCHS = 50
NUM_EXAMPLES_TO_GENERATE = 16

@tf.function
def train_step(images, attrs):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    #noise = tf.random.uniform([BATCH_SIZE, NOISE_DIM], -1, 1)

    s_attrs = attrs[:,0,0,:]
    c_noise = tf.concat([noise, s_attrs], axis=-1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(c_noise, training=True)

      d_real_logits = discriminator([images, attrs], training=True)
      d_fake_logits = discriminator([generated_images, attrs], training=True)

      gen_loss = tf.reduce_mean(tf.nn.l2_loss(d_fake_logits - tf.ones_like(d_fake_logits)))
      d_loss_real = tf.reduce_mean(tf.nn.l2_loss(d_real_logits - tf.ones_like(d_real_logits)))
      d_loss_fake = tf.reduce_mean(tf.nn.l2_loss(d_fake_logits - tf.zeros_like(d_real_logits)))
      disc_loss = d_loss_real + d_loss_fake

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
  k, q, lps = get_icdf()
  fid_best = 1000
  epoch = 0
  while epoch < EPOCHS:
    epoch += 1
    epoch_start = time.time()

    for image_batch in dataset.take(1000):
      images, attrs = image_batch
      g_loss, d_loss = train_step(images, attrs)

    # Checkpoint restoration
    if RESTORE_CKPT and epoch == 1 and os.path.exists(os.path.join(CKPT_DIR, 'checkpoint')):
      status = checkpoint.restore(tf.train.latest_checkpoint(CKPT_DIR))
      status.assert_existing_objects_matched()
      status.assert_consumed()
      print("---")
      print(f"Checkpoint restored.")

    print(g_loss.numpy(), d_loss.numpy())

    seed = np.random.normal(size=(NUM_EXAMPLES_TO_GENERATE, NOISE_DIM)).astype(np.float32)
    #---
    cond = np.zeros((16, 6))
    noise = np.random.uniform(0, 1, size=[4, 6]).astype(np.float32)
    for i in range(6):
      x = noise[:, i]
      y = np.polynomial.legendre.legval(x, lps[i]).clip(-1, 1)  #[-1, 1]
      noise[:, i] = (y - q[i]) / k[i]
    for i in range(4):
      for j in range(4):
        cond[i*4+j, :] = noise[i, :]
    #---
    cond_input = np.concatenate((seed, cond), axis=-1)
    generate_and_save_images(generator, epoch, cond_input, RES)

    # Compute FID for FID_BATCH examples
    if epoch % 10 == 0 or epoch == 1:
      print("Calculating FID ... ", end="", flush=True)
      fid_noise = np.random.normal(size=[FID_BATCH, NOISE_DIM]).astype(np.float32)
      fid_fake=np.empty([FID_BATCH, int(RES*0.75), RES, 1])
      dn = 100
      for i in range(FID_BATCH//dn):
        fid_batch = fid_noise[i*dn:(i+1)*dn]
        fid_attrs = np.random.uniform(0, 1, size=(dn, 6)).astype(np.float32)
        for j in range(6):
          x = fid_attrs[:, j]
          y = np.polynomial.legendre.legval(x, lps[j]).clip(-1, 1)  #[-1, 1]
          fid_attrs[:, j] = (y - q[j]) / k[j]
        c_fid_batch = tf.concat([fid_batch, fid_attrs], axis=-1)
        g = generator(c_fid_batch, training=False)
        g = (g + 1.) * 127.5
        fid_fake[i*dn:(i+1)*dn] = g.numpy()
      fid_fake = np.concatenate((fid_fake,) * 3, axis=3)
      fid_fake = np.rint(fid_fake).clip(0, 255).astype(np.uint8)
      fid_fake = fid_fake.astype('float32')
      fid_fake = scale_images(fid_fake, (299, 299, 3))
      images_fake = preprocess_input(fid_fake)

      fid_start = time.time()
      fid = calculate_fid(FIDmodel, act_real, images_fake)
      fid_end = time.time()
      msg = f"{fid:6.2f}, time {fid_end - fid_start:.2f} sec"
      print(msg)
      with open(f"{IMGS_DIR}/fid.txt", "a+") as fidfile:
        fidfile.writelines([f"{epoch:03}_FID {msg}\n"])
      print(f"{fid:.2f}", f"{fid_best:.2f}")

      if fid<fid_best and epoch>5:
        fid_best = fid
        generate_fake_images(generator, epoch, k, q, lps)

    # Save the model every 10 epoch
    if epoch % 10 == 0:
      checkpoint_prefix = os.path.join(CKPT_DIR, f"ckpt_{epoch}")
      checkpoint.save(file_prefix = checkpoint_prefix)
      print(f'Checkpoint saved at epoch {epoch}.')

    print (f"Time for epoch {epoch} is {time.time()-epoch_start:.2f} sec")

  # Generate after the final epoch
  seed = np.random.normal(size=(NUM_EXAMPLES_TO_GENERATE, NOISE_DIM)).astype(np.float32)
  cond = np.zeros((16, 6))
  noise = np.random.uniform(0, 1, size=[4, 6]).astype(np.float32)
  for i in range(6):
    x = noise[:, i]
    y = np.polynomial.legendre.legval(x, lps[i]).clip(-1, 1)  #[-1, 1]
    noise[:, i] = (y - q[i]) / k[i]
  for i in range(4):
    for j in range(4):
      cond[i * 4 + j, :] = noise[i, :]
  # ---
  cond_input = np.concatenate((seed, cond), axis=-1)
  generate_and_save_images(generator, epoch, cond_input, RES)


def generate_and_save_images(model, epoch, test_input, res):
  ximg = res
  yimg = int(0.75*res)

  predictions = model(test_input, training=False)

  g = predictions
  g = (g + 1.) * 127.5
  canvas = np.empty((yimg * 4, ximg * 4, 1))
  c = 0
  for i in range(4):
    for j in range(4):
      c = c + 1
      canvas[j * yimg:(j + 1) * yimg, i * ximg:(i + 1) * ximg] = g[c-1]

  image = np.rint(canvas).clip(0, 255).astype(np.uint8)
  image = np.squeeze(image)
  image = Image.fromarray(image)
  image.save(f"{IMGS_DIR}/fake_{epoch*1000}.jpg")

def save_real_images(dataset, res):
  ximg = res
  yimg = int(0.75*res)

  for images in dataset.take(1):
    g = (images[0][:16] + 1.) * 127.5

  canvas = np.empty((yimg * 4, ximg * 4, 1))
  c = 0
  for i in range(4):
    for j in range(4):
      c = c + 1
      canvas[j * yimg:(j + 1) * yimg, i * ximg:(i + 1) * ximg] = g[c-1]

  image = np.rint(canvas).clip(0, 255).astype(np.uint8)
  image = np.squeeze(image)
  image = Image.fromarray(image)
  image.save(f"{IMGS_DIR}/reals.jpg")


def generate_fake_images(model, epoch, k, q, lps):
  FAKE_BATCH = 500
  fake_batch = np.empty([FAKE_BATCH, int(RES*0.75), RES, 1])
  fake_noise = np.random.normal(size=[FAKE_BATCH, NOISE_DIM]).astype(np.float32)
  fake_attrs = np.empty([FAKE_BATCH, 6])
  dn = 100
  for i in range(FAKE_BATCH // dn):
    frac_batch = fake_noise[i * dn:(i + 1) * dn]
    frac_attrs = np.random.uniform(0, 1, size=(dn, 6)).astype(np.float32)
    for j in range(6):
      x = frac_attrs[:, j]
      y = np.polynomial.legendre.legval(x, lps[j]).clip(-1, 1)  # [-1, 1]
      frac_attrs[:, j] = (y - q[j]) / k[j]
    c_frac_batch = tf.concat([frac_batch, frac_attrs], axis=-1)
    g = generator(c_frac_batch, training=False)
    g = (g + 1.) * 127.5
    fake_batch[i * dn:(i + 1) * dn] = g.numpy()
    fake_attrs[i * dn:(i + 1) * dn] = frac_attrs
  fake_batch = np.rint(fake_batch).clip(0, 255).astype(np.uint8)
  #---
  dir_path = f"./{IMGS_DIR}/fake_train/fake_epoch_{epoch * 1000}"
  print(f"Creating FAKE images in {dir_path} ...")
  if not (os.path.isdir(dir_path)):
    os.makedirs(dir_path)
  attrsname = dir_path + "/0_fakeattrs.txt"
  with open(attrsname, "w") as file:
    for i in range(FAKE_BATCH):
      image = fake_batch[i]
      image = np.squeeze(image)
      image = Image.fromarray(image)
      imgname = f"fake_{i:>04d}.jpg"
      filename = dir_path + "/" + imgname
      image.save(filename)
      #---
      line = imgname
      for j in range(6):
        line += f", {fake_attrs[i, j]:.6f}"
      line += "\n"
      file.write(line)

#===
print("Start training ... ")

f = open(f"{IMGS_DIR}/fid.txt", "w"); f.close()
save_real_images(train_dataset, RES)
train(train_dataset, EPOCHS)
#===

anim_file = f"{IMGS_DIR}/progress.gif"

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob(f"{IMGS_DIR}/fake*.jpg")
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)