# GANs
Learning Genearative Adversial Networks to expand photographs database. Two architectures are implemented Deep Convolutional GAN (DCGAN) and Least Squares GAN (LSGAN). Both unconditional and conditional versions are implemented. Learning is possible in distributed GPU environment.

## Keywords:
generative adversial networks, distributed learning, DCGAN, LSGAN

## Requirements:

tensorflow, numpy, pandas, scipy, pillow, matplotlib, scikit-image, time

## Main features:
- learns and generate artifical photographs of disc
- implements both unconditional and conditional versions
- distibuted version runs in parallel on more GPUs
- computes Frechet Inception Score (FID)

## Usage:
Scripts are controlled via global variables typed in CAPITAL LETTERS and accessible in codes of the scripts. These are the important ones:

- RES - controls resolution of generated images (64 or 128)
- BATCH_SIZE - batch size per single GPU (32)
- FID_BATCH - number of images for FID computation (10000)

- IMGS_DIR - output directory for generated images
- CKPT_DIR - checkpoints directory
- RESTORE_CKPT - start training from a saved checkpoint (False)

- EPOCHS - epochs of learning (50)
- NOISE_DIM - noise dimension for generator (100)

Scripts are run by
```
python name_of_script.py
```

## List of unconditional scripts:
- nck_dcgan.py - unconditional DCGAN
- nck_dcgan_distr.py - distrubuted version running on more GPUs
- nck_lsgan.py - unconditional LSGAN
- nck_lsgan_distr.py - distrubuted version running on more GPUs

## List of conditional scripts:
In conditinal versions, generation is coditioned by coordinates associated with training photograps. In the output directory, a txt file is recorded with coordinates used for generated pjotographs.

- nck_dcganc.py - conditional DCGAN
- nck_dcganc_distr.py - distrubuted version running on more GPUs
- nck_lsganc.py - unconditional LSGAN
- nck_lsganc_distr.py - distrubuted version running on more GPUs