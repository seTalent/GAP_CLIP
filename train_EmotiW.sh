#!/bin/bash
CUDA_VISIBLE_DEVICES='0,1' python main.py \
--dataset 'EmotiW' \
--workers 8 \
--epochs 50 \
--batch-size 32 \
--lr-temporal-net 1e-3 \
--lr-image-encoder 1e-5 \
--lr-prompt-learner 1e-3 \
--lr-gap-net 1e-3 \
--lr-cross-net 1e-4 \
--weight-decay 1e-4 \
--momentum 0.9 \
--print-freq 10 \
--milestones 30 40 \
--contexts-number 8 \
--class-token-position "end" \
--class-specific-contexts 'True' \
--text-type 'class_descriptor' \
--seed 1 \
--temporal-layers 1 \
--exper-name 'EmotiW补充实验' \
--pretrain 'pretrain/ViT-B-32.pt' \
--load_and_tune_prompt_learner True
# --model  1 \
# --alpha 0.5


