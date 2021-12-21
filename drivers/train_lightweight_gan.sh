#!/bin/sh
# Various usage details can be found at: https://github.com/lucidrains/lightweight-gan

DATAROOT=/media/WD2TB/neil/data/Humans/face
OUTPATH=/media/WD2TB/neil/lightweight_gan_results/
EXP_CODE=0000

lightweight_gan --data $DATAROOT \
  --image-size 128 \
  --batch-size 64 \
  --name \"${EXP_CODE}\" \
  --gradient-accumulate-every 1 \
  --num-train-steps 200000 \
  --aug-prob 0.25 \
  --aug-types ["translation","cutout","color"] \
  --num-image-tiles 4 \
  --multi-gpus \
  --disc-output-size 1
