#!/usr/bin/env bash

#source /home/jenny/anaconda3/envs/lissc/bin/activate


CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train.py \
--model='DSFNet' \
--dataset=nyucad \
--gpu=0,1,2,3 \
--epochs=250 \
--batch_size=8 \
--workers=4 \
--lr=0.005 \
--model_name='DSFNet'

# 2>&1 |tee checkpoint/DSFNet_NYUCAD.log


#  deactivate

