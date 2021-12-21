#!/bin/sh
# Various usage details can be found at: https://github.com/lucidrains/lightweight-gan

DATAROOT=/media/WD2TB/neil/data/Humans/face
EXP_CODE=0000
RES_DIR=/media/WD2TB/neil/lightweight_gan_results/${EXP_CODE}/samples
MODELS_DIR=/media/WD2TB/neil/lightweight_gan_results/${EXP_CODE}/models


# --aug-types default settings is ["translation","cutout"], can add "color"
# --amp to do mixed precision training
# --load-from path to pretrained model

lightweight_gan --data $DATAROOT \
  --image-size 128 \
  --batch-size 64 \
  --name \"${EXP_CODE}\" \
  --results-dir \"${RES_DIR}\" \
  --models-dir \"${MODELS_DIR}\" \
  --gradient-accumulate-every 1 \
  --num-train-steps 200000 \
  --aug-prob 0.25 \
  --aug-types ["translation","cutout","color"] \
  --num-image-tiles 4 \
  --multi-gpus \
  --disc-output-size 1 \
  --save-every 1000 \
  --evaluate-every 1000
