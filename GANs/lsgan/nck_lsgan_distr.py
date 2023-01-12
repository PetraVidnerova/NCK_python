import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import functools

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
BUFFER_SIZE = 60000
NOISE_DIM = 100
BATCH_SIZE = 32
FID_BATCH = 1000

IMGS_DIR = f'dcgan_distr_images{RES}'
if not os.path.isdir(IMGS_DIR):
  os.mkdir(IMGS_DIR)
print(os.path.abspath(IMGS_DIR))

CKPT_DIR = f'./{IMGS_DIR}/training_checkpoints'
CKPT_RESTORE = True

#---
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
  return image, attrs

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0).astype('float32')
		images_list.append(new_image)
	return np.asarray(images_list)

def calculate_fid(model, act1, images2):
	# calculate activations
	act2 = model.predict(images2)
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	ssdiff = np.sum((mu1 - mu2)**2.0)
	covmean = sqrtm(sigma1.dot(sigma2))
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# Distribute strategy
strategy = tf.distribute.MirroredStrategy(
  cross_device_ops=tf.distribute.ReductionToOneDevice()
  #cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
  #cross_device_ops=tf.distribute.NcclAllReduce()
)
#strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
num_gpus = strategy.num_replicas_in_sync
print ('Number of devices: {}'.format(num_gpus))

BATCH_SIZE_PER_REPLICA = BATCH_SIZE
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_gpus

# Dataset
xres = RES; yres=int(0.75*RES); dstype = 'train'
tfrecord_file = f"E:\\NCK\\gan_{xres}{yres}{dstype}.tfrecord"
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(_extract_fn)
dataset = dataset.repeat()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(GLOBAL_BATCH_SIZE)
dataset = dataset.prefetch(4)
train_dataset = dataset
# distributed dataset
train_dataset = strategy.experimental_distribute_dataset(dataset)

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
print("FID done reals.")

# 64x48 Generator
def make_generator_model64():
  gf_dim = 64
  model = tf.keras.Sequential()
  model.add(layers.Dense(gf_dim * 8 * 3 * 4, input_shape=(100,)))
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

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
  assert model.output_shape == (None, 48, 64, 1)

  return model

# 128x96 Generator
def make_generator_model128():
  gf_dim = 64
  model = tf.keras.Sequential()
  model.add(layers.Dense(gf_dim * 8 * 3 * 4, input_shape=(100,)))
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

with strategy.scope():
  if RES == 64:
    generator = make_generator_model64()
  elif RES == 128:
    generator = make_generator_model128()

noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False) #NHWC
#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#print(generator.summary())

# 64x48 Discriminator
def make_discriminator_model64():
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(48, 64, 1)))
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

  return model

# 128x96 Discriminator
def make_discriminator_model128():
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(96, 128, 1)))
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

  return model

with strategy.scope():
  if RES == 64:
    discriminator = make_discriminator_model64()
  elif RES == 128:
    discriminator = make_discriminator_model128()

decision = discriminator(generated_image)
#print(decision)
#print(discriminator.summary())

# Distributed loss
cross_entropy_distr = tf.keras.losses.BinaryCrossentropy(
  from_logits=True,
  reduction=tf.keras.losses.Reduction.SUM)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy_distr(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy_distr(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy_distr(tf.ones_like(fake_output), fake_output)

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

# Training loop
EPOCHS = 50
NUM_EXAMPLES_TO_GENERATE = 16

seed = np.random.normal(size=[NUM_EXAMPLES_TO_GENERATE, NOISE_DIM]).astype(np.float32)

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    #noise = tf.random.uniform([BATCH_SIZE, NOISE_DIM], -1, 1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

@tf.function
def distributed_train_step(images):
  per_replica_gen_loss, per_replica_disc_loss = strategy.run(train_step, args=(images,))
  mean_g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_gen_loss, axis=None)
  mean_d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_disc_loss, axis=None)
  return mean_g_loss, mean_d_loss


def train(dataset, epochs):
  epoch = 0
  while epoch < EPOCHS:
    epoch += 1
    epoch_start = time.time()

    i = 0
    for image_batch in dataset:
      i += 1
      g_loss, d_loss = distributed_train_step(image_batch[0])
      if i==1000:
        break

    # Checkpoint restoration
    if CKPT_RESTORE and epoch == 1 and os.path.exists(os.path.join(CKPT_DIR, 'checkpoint')):
      status = checkpoint.restore(tf.train.latest_checkpoint(CKPT_DIR))
      status.assert_existing_objects_matched()
      status.assert_consumed()
      #epoch = epoch_var.numpy();
      #cur_nimg = cur_nimg_var.numpy()
      print("---")
      print(f"Checkpoint restored.")
      print("---")

    print(g_loss.numpy(), d_loss.numpy())

    generate_and_save_images(generator, epoch, seed, RES)

    # Compute FID for FID_BATCH examples
    if epoch % 5 == 0 or epoch == 1:
    #if False:
      print("Calculating FID ... ", end="", flush=True)
      fid_noise = np.random.normal(size=[FID_BATCH, NOISE_DIM]).astype(np.float32)
      fid_fake=np.empty([FID_BATCH, int(RES*0.75), RES, 1])
      dn = 100
      for i in range(FID_BATCH//dn):
        fid_batch = fid_noise[i*dn:(i+1)*dn]
        g = generator(fid_batch, training=False)
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

    # Save the model every 10 epoch
    if epoch % 10 == 0:
      checkpoint_prefix = os.path.join(CKPT_DIR, f"ckpt_{epoch}")
      checkpoint.save(file_prefix = checkpoint_prefix)
      print(f'Checkpoint saved at epoch {epoch}.')

    print (f"Time for epoch {epoch} is {time.time()-epoch_start:.2f} sec")

  generate_and_save_images(generator, epochs, seed, RES)

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

  for images in dataset:
    if num_gpus > 1:
      images = images[0].values #1st replica
    g = (images[0][:16] + 1.) * 127.5
    break

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

#===
print("Start training ... ")

f = open(f"{IMGS_DIR}/fid.txt", "w"); f.close()
save_real_images(train_dataset, RES)
train(train_dataset, EPOCHS)

anim_file = f"{IMGS_DIR}/progress.gif"

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob(f"{IMGS_DIR}/fake*.jpg")
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)