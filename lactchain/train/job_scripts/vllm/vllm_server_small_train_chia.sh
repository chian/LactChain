#!/bin/bash
. /etc/profile
#PBS -l select=1
#PBS -A argonne_tpc
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q single-gpu

### Name of your session
#PBS -N SMALL_Critic_Finetuning

### Controlling output of application 
#PBS -k doe
#PBS -o ./output_logs_critic_finetune_small
#PBS -e ./error_logs_critic_finetune_small

### email notification
#PBS -m be
#PBS -M chia@anl.gov

ROOT=/lus/eagle/projects/argonne_tpc/chia-llama2/LactChain
SCRIPT_DIR=$ROOT/lactchain/train/
export HF_HOME="/lus/eagle/projects/argonne_tpc/chia-llama2/.cache"
export HUGGING_FACE_HUB_TOKEN='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'

# export outbound proxies
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

### directions
module use /soft/modulefiles/
module load conda
conda activate ${ROOT}/../conda_envs/lactchain 
echo "Activating environment"
cd $SCRIPT_DIR
export PYTHONPATH=$SCRIPT_DIR/../:$PYTHONPATH
# cd ..
PARENT_DIR=$(pwd)
echo "Changing to current directory for training: {$PARENT_DIR}"

# setting up 4 servers 
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model "$HF_HOME/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de" \
    --dtype auto \
    --port 8000 \
    --api-key lactchain \
    --trust-remote-code True \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.6 

## Running actual script
fabric run critic_train_lightning.py \
    --actor_path "$HF_HOME/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de" \
	--actor_model_type 'mistral-7b' \
    --critic_path "$HF_HOME/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11" \
    --logging_level 'info' \
    --logging_save_path 'sophia_mistral_7b_logging_small.log' \
    --log_wandb_offline False \
    --gamma 0.99 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --num_episodes 10 \
    --global_buffer_size 10000 \
    --collection_batch_size 8 \
    --train_batch_size 64 \
    --checkpoints_total_limit 10 \
    --checkpointing_steps 10 \
    --output_dir 'finetuned-critic'\
    --devices 4 \
    --accelerator 'cuda' \
	--strategy 'ddp'

cd $SCRIPT_DIR/job_scripts/vllm
