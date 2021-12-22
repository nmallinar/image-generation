#!/bin/sh

DATAROOT=/media/WD2TB/neil/data/Arian
OUTPATH=/media/WD2TB/neil/dcgan_results/0005

PYTHONPATH=.. python ../train_dcgan.py \
  --dataset "folder" \
  --dataroot $DATAROOT \
  --workers 4 \
  --batchSize 64 \
  --imageSize 64 \
  --nz 100 \
  --ngf 64 \
  --ndf 64 \
  --niter 25 \
  --lr_disc 0.0002 \
  --lr_gen 0.0002 \
  --beta1 0.5 \
  --cuda \
  --ngpu 1 \
  --outf $OUTPATH \
  --model-save-freq 10 \
  --image-save-freq 10 \
  --grad_accumulate 1
