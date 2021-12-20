#!/bin/sh

DATAROOT=/media/WD2TB/neil/data/Humans
OUTPATH=/media/WD2TB/neil/dcgan_results

PYTHONPATH=.. python main.py \
  --dry-run \
  --dataset "folder" \
  --dataroot $DATAROOT \
  --workers 4 \
  --batchSize 64 \
  --imageSize 64 \
  --nz 100 \
  --ngf 64 \
  --ndf 64 \
  --niter 25 \
  --lr 0.0002 \
  --beta1 0.5 \
  --cuda \
  --ngpu 2 \
  --outf $OUTPATH
