#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1 python  main.py \
# --model 'swin' \
# --dataset 'DAiSEE' \
# --workers 8 \
# --epochs 50 \
# --lr-image-encoder 1e-5 \
# --lr-prompt-learner 1e-3 \
# --lr_temporal_net 1e-3 \
# --batch-size 128 \
# --weight-decay 1e-3 \
# --momentum 0.9 \
# --print-freq 10 \
# --milestones 30 40 \
# --seed 1 \
# --exper-name 'DAiSEE-SwinFace(NoFlow)-main-Swin-(2,2,6,2)' \
# --fuse_type 'lstm' \
# --run_type train \
# --fine_tune 'scratch' \
# --pretrain_path '/data/zky_1/codes/DFER-CLIP/pretrain/swin_tiny_patch4_window7_224.pth' \
# --best_checkpoint_path '/data/zky_1/codes/DFER-CLIP/checkpoint/DAiSEE-2412161947DAiSEE-SwinFace(NoFlow)-main-Swin-T-model_best.pth' \
# --depths 2,2,2 \
# --num_heads 3,6,24 \
# --img_size 112 \
# --weighted_sampler
# --flow 
#开启加权采样
#不用流将其注释

#fine_tune: 'fine_tune' | 'fully_fine_tune' | 'scratch'
CUDA_VISIBLE_DEVICES=0,1 python  main.py \
    --dataset "EmotiW" \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-2 \
    --lr-temporal-net 1e-4 \
    --lr-image-encoder 1e-5 \
    --lr-prompt-learner 1e-5 \
    --lr-gap-net 1e-3 \
    --lr-cross-net 1e-4 \
    --pretrain 'pretrain/ViT-B-32.pt' \
    --load_and_tune_prompt_learner True \
    --contexts-number 8 \
    --class-token-position "end" \
    --class-specific-contexts 'True' \
    --text-type 'class_descriptor' \
    --temporal-layers 1 \