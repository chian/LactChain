
# NOTE: url = f"http://{host}:{port}/predict" IS THE STRUCTURE OF HOST AND PORT 
# YOU NEED TO EXPORT VLLM_HOST_IP AND 
# proxy servers
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
# export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

# export HOST="http://proxy-01.pub.alcf.anl.gov"
# export PORT=3128
# export VLLM_HOST_IP='http://proxy-01.pub.alcf.anl.gov:3128'
# export LOCAL_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export HOST="http://0.0.0.0"
export PORT=8000
# export VLLM_HOST_IP='http://localhost:8000'
export LOCAL_PROXY="http://0.0.0.0:8000"

# setting up 4 servers 
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de' \
    --dtype auto \
	--host $HOST \
    --port $PORT \
    --api-key lactchain \
    --trust-remote-code \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.6 &

sleep 20

# ## Running actual script
# fabric run critic_train_lightning.py \
#     --actor_path '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'\
# 	--actor_model_type 'mistral-7b' \
#     --critic_path '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'\
#     --logging_level 'info' \
#     --logging_save_path 'sophia_mistral_7b_logging_small.log' \
#     --log_wandb_offline False \
#     --gamma 0.99 \
#     --learning_rate 1e-4 \
#     --hosts $LOCAL_PROXY \
#     --api_key lactchain \
#     --num_epochs 10 \
#     --num_episodes 10 \
#     --global_buffer_size 100 \
#     --collection_batch_size 8 \
#     --train_batch_size 64 \
#     --checkpoints_total_limit 10 \
#     --checkpointing_steps 10 \
#     --output_dir 'finetuned-critic'\
#     --devices 1 \
#     --accelerator 'cuda' \
# 	--strategy 'ddp'
