#!/bin/bash

#PBS -l select=2
#PBS -A argonne_tpc
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable

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

# Get the node list
all_nodes=($(cat $PBS_NODEFILE | sort | uniq))
num_nodes=${#all_nodes[@]}

# Separate nodes for vLLM and training
vllm_nodes=(${all_nodes[0]})  # Use the first node for vLLM
training_nodes=("${all_nodes[@]:1}")  # Use the rest for training

JOB_ID=$PBS_JOBID
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export MPICH_GPU_SUPPORT_ENABLED=1

# Run vLLM on the first node with proper environment initialization
ssh ${vllm_node} << EOF
    # Load modules and activate conda environment
    module use /soft/modulefiles/
    module load conda
    conda activate ${ROOT}/../conda_envs/lactchain
    echo "Activating environment"
    
    # Set up Python path
    cd $SCRIPT_DIR
    export PYTHONPATH=$SCRIPT_DIR/../:$PYTHONPATH
    
    # Run vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model "$HF_HOME/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de" \
        --dtype auto \
        --port 8000 \
        --api-key lactchain \
        --trust-remote-code True \
        --pipeline-parallel-size 1 \
        --gpu-memory-utilization 0.6 &
EOF

# Wait a bit for vLLM to start up
echo "vLLM server started. Sleeping for 30 seconds..."
sleep 30

# Prepare the host list for training
host_list=$(IFS=,; echo "${training_nodes[*]}")

export MASTER_ADDR=${training_nodes[0]}
#export MASTER_ADDR=`head -n 1 $PBS_NODEFILE`
export MASTER_PORT=29400
NNODES=$((${#all_nodes[@]} - 1))
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

## Running actual script
fabric run critic_train_lightning.py \
    --actor_path "$HF_HOME/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de" \
    --actor_model_type 'mistral-7b' \
    --critic_path "$HF_HOME/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11" \
    --logging_level 'info' \
    --logging_save_path "chia_polaris_mistral_7b_logging_small_${JOB_ID}.log" \
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
    --output_dir "finetuned-critic-${JOB_ID}"\
    --devices $NTOTRANKS \
    --accelerator 'cuda' \
    --strategy 'ddp' \
    --vllm-host $vllm_node

cd $SCRIPT_DIR/job_scripts/vllm
