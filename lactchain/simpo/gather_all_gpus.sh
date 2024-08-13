#!/bin/bash

BASH_DIR=$(pwd)
echo "Launching Bash Script From {$BASH_DIR}"

echo "Swapped Directory to {$TRAIN_DIR}, Launching Train Script..."

PRETRAINED_ACTOR_DIR='./hf_ckpts/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
PRETRAINED_CRITIC_DIR='./hf_ckpts/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'
LOGGING_SAVE_PATH='./logs/python_logs/bash_debug_simpo_7b.log'
OUTPUT_DIR='./finetuned-critic/'

fabric run lightning_simpo_data_gather.py \
    --backend 'huggingface' \
    --actor_path $PRETRAINED_ACTOR_DIR \
	--actor_model_type 'mistral-7b' \
    --critic_path $PRETRAINED_CRITIC_DIR \
    --logging_level 'info' \
    --logging_save_path $LOGGING_SAVE_PATH \
    --log_wandb_offline False \
    --show_parsing_errors False \
    --gamma 0.99 \
    --learning_rate 1e-4 \
    --global_buffer_size 8 \
    --collection_batch_size 32 \
    --train_batch_size 64 \
    --checkpoints_total_limit 10 \
    --checkpointing_steps 2 \
    --output_dir $OUTPUT_DIR \
    --devices 4 \
    --accelerator 'cuda' \
	--strategy 'ddp'
