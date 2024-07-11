#!/bin/bash

#cd ../..

# custom config
#DATA=/export/livia/home/vision/Bmurugesan/data/data
DATA='/export/livia/datasets/datasets/public/image_classification/'
TRAINER=CoCoOp_Zs
# TRAINER=CoOp

DATASET=imagenet
SEED=1

#CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=rn50_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=8


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir output/imagenet/CoCoOp/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 10 \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi