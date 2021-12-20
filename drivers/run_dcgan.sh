#!/bin/sh

DATAROOT=/media/WD2TB/neil/data/Humans
OUTPATH=/media/WD2TB/neil/dcgan_results/0002

PYTHONPATH=.. python main.py \
  --dataset "folder" \
  --dataroot $DATAROOT \
  --workers 4 \
  --batchSize 64 \
  --imageSize 128 \
  --nz 100 \
  --ngf 64 \
  --ndf 64 \
  --niter 25 \
  --lr 0.0002 \
  --beta1 0.5 \
  --cuda \
  --ngpu 2 \
  --outf $OUTPATH \
  --model-save-freq 10 \
  --image-save-freq 100
