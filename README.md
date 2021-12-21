# image-generation
exploring various models and training regimes for image generation, text-to-image generation, style transfer, etc

## lightweight_gan

### debugging

If there is an issue with kornia filter2d, look at: `https://github.com/lucidrains/lightweight-gan/issues/90`

## dcgan

### running

All of this is summarizing what's at: `https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html`

The `'folder'` dataset type in train_dcgan.py currently uses
the Pytorch default ImageFolder dataset class which
seeks images in the format of:
```
<root_img_dir>
  <class_1_dir>
    1.jpg
    2.jpg
    ....
  <class_2_dir>
    ....
```

However, at the time we are not concerned with classes so
I have lumped all the data into a single subdirectory
for some random class name (eventually, we may want to separate
generation based on classes).

After this is setup, modify the file at: `drivers/run_dcgan.sh` with the relevant paths locally,
change the `--cuda` and `--ngpu` settings to match your system, and it should work as is on any
folder of images.

You can also add the `--dry-run` flag to test on one epoch running only.

### experiment notes

Following the PyTorch example's default settings, we get good results on the smaller Humans dataset from Kaggle (`https://www.kaggle.com/ashwingupta3012/human-faces`) using 64x64 sized images and batchSize=64. I tried with batchSize=256 to see if big batches helps here and it does not, though I was holding the number of epochs constant (25) so it is left to be seen whether running more steps with bigger batches will help further.

Tried to extend just to 128x128 but the models don't seem to do well on this data. Have not tried running on the richer CelebFaces dataset or larger datasets, but initially I'm trying to explore small data settings. Attempts to make 128x128 work included:
  - just using 128x128 image size, keeping ngf=ndf=64, batchSize=64, everything else the same as default parameters
  - increasing batchSize to 128
  - increasing ngf=ndf=128
  - trying ngf=64, ndf=32
  - trying ngf=ndf=64 but switching ReLU in Generator to LeakyReLU

but these all didn't provide good results or completely diverged, but the closest to something reasonable was switching to LeakyReLU in the Generator. It's possible for the above larger network, larger ngf/ndf settings, we would need to train much more than 25 epochs but I didn't visually see reasonable movement in sample quality after 25 epochs so I opted to abandon this direction.

Training with LeakyReLU longer could work, might be time to switch to a more modern GAN codebase with flexibility to do a lot more things like data augs, new losses, all the various tricks that people have found over the years with GANs. Last attempt to make it work on 128x128 was:
  - ngf=ndf=64, LeakyReLU in Generator, nz=200

but this didn't look good either.

This is a solid baseline model, and the code is very simple so it can be adapted and modified quite easily and works fairly well on 64x64 patches on small and large data. Time to move on to something fancier.

If we return to this direction, we shoudl try:
- data augmentations
- loner training time
- more training data
