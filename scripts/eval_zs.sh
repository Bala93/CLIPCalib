#!/bin/bash

# custom config
DATA=/home/ar88770/CLIPCalib/data  
TRAINER=ADAPTER

DEVICE=$1
DATASET=$2      # target dataset - i.e. {imagenet, caltech101, oxford_pets, stanford_cars, oxford_flowers, food101,
                #                        fgvc_aircraft, sun397, dtd, eurosat, ucf101}
CFG=$3          # config file - SGD_lr1e-1_B256_ep300
SHOTS=$4        # number of shots (1, 2, 4, 8, 16)
INIT=$5         # Method / Linear Probe init - i.e. {RANDOM, ZS, ClipA, TipA, TipA(f), TR, TRenh}
CONSTRAINT=$6   # apply class-adaptive constraint in Linear Probing (CLAP) - i.e. {none, l2}
BACKBONE=$7     # CLIP backbone to sue - i.e. {RN50, RN101, ViT-B/32, ViT-B/16}
FOLDERNAME=$8

OUTDIR=output/FINAL/${DATASET}/${FOLDERNAME}/zs
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${CFG}.yaml --output-dir ${OUTDIR} --zs-eval-only --backbone ${BACKBONE}
