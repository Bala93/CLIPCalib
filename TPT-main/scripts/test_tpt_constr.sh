#!/bin/bash

data_root='/home/ar88770/TPT/datasets'
testsets=$1
gpu=$2
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./tpt_classification_constrained.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} \
--tpt --ctx_init ${ctx_init}
