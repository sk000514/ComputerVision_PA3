#!/usr/bin/env bash
#--dataset_root /jisu/dataset/iHarmony4resized256/ \

python train.py \
--dataset_root ./dataset \
--name experiment_train_2080 \
--checkpoints_dir ./checkpoints/scratch/ \
--model rainnet \
--netG rainnet \
--dataset_mode iharmony4 \
--is_train 1 \
--gan_mode wgangp \
--normD instance \
--normG RAIN \
--preprocess None \
--niter 40 \
--niter_decay 40 \
--input_nc 3 \
--batch_size 200 \
--lambda_L1 100 \
--num_threads 60 \
--print_freq 400 \
--gpu_ids 0,1,2,3 \
#--continue_train \
#--load_iter 87 \
#--epoch 88 \
