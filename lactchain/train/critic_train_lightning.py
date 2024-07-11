from __future__ import annotations
from typing import List, Callable, Dict, Any, Tuple, Union, Optional
from pathlib import Path
from torch import Tensor
import torch, numpy as np, gymnasium as gym
from torch.utils.data import DistributedSampler, BatchSampler, RandomSampler
from lightning import Fabric
from lightning.fabric.utilities import AttributeDict
from wandb.integration.lightning.fabric import WandbLogger
import pprint as pp
import os, shutil, math, logging, re
from argparse import ArgumentParser
from pydantic import Field
from tqdm import tqdm
from tqdm.auto import tqdm

from lactchain.models.lightning_agent import LightningA2C
from lactchain.models.critic import ValueFunctionConfig
from lactchain.models.actor import ActorConfig, LoraConfigSettings
from lactchain.environments.grid_world import make_env, process_environment_outputs
from lactchain.configs.base_config import BaseConfig
from lactchain.utils import configure_logger

PathLike=Union[str, Path]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_LOGS']="+dynamo"
os.environ['TORCHDYNAMO_VERBOSE']='1'
# default actor path
ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
CRITIC_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'

class FabricDeviceConfig(BaseConfig): 
    '''Base Config for running fabric accelerator'''
    accelerator:str=Field(
        default='auto', 
        description='What kind of accelerator to use for Fabric'
    )
    devices:int=Field(
        default=..., 
        description='The Number of devices to run on for fabric'
    )
    num_nodes:int=Field(
        default=1, 
        description='The number of nodes to run on for fabric'
    )
    strategy:str=Field(
        default='auto', 
        description='The type of strategy to use for Fabric'
    )
    def __init__(self, devices:int): 
        super().__init__(devices=devices)

def argparse(): 
    parser=ArgumentParser()
    parser.add_argument(
        '--actor_path', 
        type=str, 
        default=ACTOR_PATH, 
        help='''Path to frozen causal model'''
        )
    parser.add_argument(
        '--critic_path', 
        type=str, 
        default=CRITIC_PATH, 
        help='''Path to trainable embedding model'''
        )
    parser.add_argument(
        '--logging_level', 
        type=str, 
        default='info',
        help='The level to choose for logging'
    )
    parser.add_argument(
        '--logging_save_path', 
        type=str, 
        default='training.log', 
        help='The file to save the logging file to'
    )
    parser.add_argument(
        "--log_wandb",
        type=bool,
        default=False,
        help="Whether or not to use fabric.setup or not for the dataloader",
    )
    parser.add_argument(
        '--gamma',
        type=float, 
        default=0.99, 
        help='''Discount factor ratio for calculating returns for Monte Carlo A2C'''
        )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=1e-4, 
        help='''Learning Rate for Optimizer'''
        )
    parser.add_argument(
        '--num_episodes', 
        type=int, 
        default=100, 
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=10, 
        help='''Number of epochs to train on during the learning phase'''
        )
    parser.add_argument(
        '--global_buffer_size', 
        type=int, 
        default=100, 
        help='''The desired max length of the buffer after gathering from all local ranks'''
        )
    parser.add_argument(
        '--collection_batch_size', 
        type=int, 
        default=32, 
        help='''The batch size when collecting experience for buffer; also the same as the environment'''
        )
    parser.add_argument(
        '--train_batch_size', 
        type=int, 
        default=32, 
        help='''Sampling batch size to train on '''
        )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="actor-finetuned-critic",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    return parser.parse_args()

def unfold_list_of_lists(list_of_list:list[list[Any]]) -> list: 
    '''Unfolds a list of lists into a single large list that preserves order'''
    unfolded_list=[item for sublist in list_of_list for item in sublist]
    return unfolded_list

def add_inputs_to_dict(batched_inputs:Dict[str, Tensor], inputs:Dict[str, Tensor]) -> Dict[str, Tensor]: 
    '''Helper function that adds the tensors inputs dictionaries from tokenizer along 0-th dimension'''
    for key in inputs:
        if key in batched_inputs:
            # Concatenate tensors along dim=0 --> [B*n, T]
            batched_inputs[key] = torch.cat((batched_inputs[key], inputs[key]), dim=0)
        else:
            batched_inputs[key] = inputs[key]
    return batched_inputs

def train(
    fabric: Fabric,
    agent: LightningA2C,
    optimizer: torch.optim.Optimizer,
    lr_scheduler:torch.optim.lr_scheduler._LRScheduler, 
    data: Dict[str, list[np.ndarray | Dict[str, Any]]],
    args: ArgumentParser,
    logger:Optional[logging.Logger]=None
    ):    
    rewards=data["rewards"]
    inputs=data['batched_inputs']
    
    indices=list(range(rewards.shape[0]))
    logger.info(f'REWARDS:{rewards}\nREWARDS SHAPE: {rewards.shape}\nTOTAL INDICES: {indices}\nLENGTH OF INDICES: {len(indices)}\n')
    
    sampler = DistributedSampler(
        indices, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True
    )
    sampler = BatchSampler(sampler, batch_size=args.train_batch_size, drop_last=False)

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.num_epochs):
            epoch_loss=0
            sampler.sampler.set_epoch(epoch)
            for batch_indices in sampler:
                logger.info(f'''Batch_Indices for rank {fabric.global_rank}: {batch_indices}''')
                fabric.barrier()
                optimizer.zero_grad()
                selected_rewards=rewards[batch_indices] # shape [B, T]
                selected_inputs={'input_ids':inputs['input_ids'][batch_indices], 
                                 'attention_mask':inputs['attention_mask'][batch_indices]}
                
                logger.debug(f'Batch Indices:\n{batch_indices}')
                logger.debug(f'Selected Rewards:\n{selected_rewards}')
                logger.debug(f'Selected Input_Ids:\n{selected_inputs["input_ids"].shape}')
                logger.debug(f'Selected Attention_Masks:\n{selected_inputs["attention_mask"].shape}')

                loss = agent.training_step(selected_rewards, selected_inputs) 
                epoch_loss+=loss
        
                fabric.backward(loss)
                optimizer.step()
                
            fabric.log_dict({"Epoch loss": epoch_loss})
            logger.info(f'Epoch Loss: {epoch_loss}')
            
            lr_scheduler.step()

def main():
    args=argparse()
    logger = configure_logger(args.logging_level, args.logging_save_path)
    wandb_logger=WandbLogger(project="Lactchain-Critic-Tuning", offline=args.log_wandb)
    # Initialize Fabric
    fabric = Fabric(loggers=wandb_logger)
    # device settings
    global_rank = fabric.global_rank # rank on global devices 
    world_size = fabric.world_size # total num devices 
    device = fabric.device
    local_rank=fabric.local_rank # rank on local node 
    
    global_buffer_size=args.global_buffer_size
    local_buffer_size=int(global_buffer_size // world_size)
    num_collection_steps=math.floor(local_buffer_size / args.collection_batch_size)
    
    logging.info(f'''LOGGING WITH THE FOLLOWING:\n\
                 GLOBAL BUFFER SIZE: {global_buffer_size}\
                 \nNUMBER OF RANKS: {world_size}\
                 \nLOCAL BUFFER SIZE: {local_buffer_size}\  
                 \nCOLLECTION BATCH SIZE PER RANK: {args.collection_batch_size}\
                 \nTRAIN BATCH SIZE PER RANK: {args.train_batch_size}\
                 \nNUM EPOCHS: {args.num_epochs}\
                 \nNUM EPISODES: {args.num_episodes}
                 ''')
    
    actor_config=ActorConfig()
    lora_config=LoraConfigSettings()
    critic_config=ValueFunctionConfig()
    
    vector_env = gym.vector.AsyncVectorEnv([make_env for _ in range(args.collection_batch_size)])
    
    agent=LightningA2C(args.actor_path, actor_config, lora_config, 
                       args.critic_path, critic_config, args.gamma)
    logger.info(f'Model Trainable Params:\n{agent.model_trainable_params}, Device:\n{fabric.accelerator}')
    optimizer = agent.configure_optimizers(args.learning_rate)
    
    # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("critic_checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("critic_checkpoint") and d.endswith(".ckpt")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split(".")[0]))
            path = dirs[-1] if len(dirs) > 0 else None
                    
        if path is None:
            fabric.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            fabric.print(f"Resuming from checkpoint {path}")
            full_checkpoint = fabric.load(os.path.join(args.output_dir, path))
            parsed_filename = re.search(r'critic_checkpoint-(\d+)\.ckpt', path)
            episode = int(parsed_filename.group(1))
            agent.load_state_dict(full_checkpoint['model'])
            optimizer.load_state_dict(full_checkpoint['optimizer'])
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    agent, optimizer = fabric.setup(agent, optimizer)
    
    progress_bar = tqdm(
        range(0, num_collection_steps),
        initial=0,
        desc=f"COLLECTING DATA FOR BUFFER OF SIZE {local_buffer_size}",
        disable=not fabric.is_global_zero, 
        file=open(os.devnull, 'w')
    )
    
    for episode in range(args.num_episodes):
        
        fabric.barrier() # sync per episode
        logger.info(f'Starting Episode {episode+1} on all ranks...') if global_rank==0 else None
        rewards=[]
        batched_inputs={}
        steps_kept=0
        buffer_size=0
        
        obs, info = vector_env.reset()
        obs, info=process_environment_outputs(obs, info)
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad(): 
                
                logger.info(f'COLLECTING EXPERIENCE FOR RANK {global_rank}...\nESTIMATED NUMBER OF COLLECTION STEPS PER RANK: {num_collection_steps}')
                step=0

                while buffer_size<local_buffer_size: 
                    step+=1
                    try: 
                        batch_mapped_actions, actions, contexts, drop_indices=agent.sample_actions(obs, info)
                        # TODO: if action is dropped, then send in default actions but do not collect 
                        next_obs, reward, done, truncated, info = vector_env.step(batch_mapped_actions)
                        next_obs, info=process_environment_outputs(next_obs, info)
                        inputs=agent.compile_and_tokenize(next_obs, info)

                        if drop_indices:
                            _filtered_obs=[next_obs.pop(drop_idx) for drop_idx in drop_indices]
                            _filtered_info=[info.pop(drop_idx) for drop_idx in drop_indices]
                            filtered_rewards=[reward.pop(drop_idx) for drop_idx in drop_indices]
                            rewards.append(filtered_rewards)
                        else: 
                            rewards.append(reward)
                            # Merging with tensor concatenation
                            batched_inputs=add_inputs_to_dict(batched_inputs, inputs)
                        
                        obs = next_obs
                        logger.info(f'''Successfully parsed {args.collection_batch_size-len(drop_indices)} actions out of batch size {args.collection_batch_size} in step {step}, adding to replay buffer for rank {global_rank}...''')
                        steps_kept+=1
                    except Exception as e: 
                        logger.error(f'''Error when collecting experience from rank {global_rank}:\n{e}\nDropping Full Batch for Step {step}...''')
                        logger.debug(f'''PROPOSED OUTPUTS:\n{pp.pformat(agent.actor.outputs)}\n\n\n''')
                        continue
                    
                    progress_bar.update(1)
                    logger.info(str(progress_bar))
                    buffer_size = len(np.concatenate(rewards))
                    logger.info(f'''RANK BUFFER SIZE: {buffer_size} AT STEP {step} FOR RANK {global_rank}''')  
                    
        logger.info(f'''FOR RANK {global_rank} TOTAL STEPS:{step} STEPS KEPT:{steps_kept} GATHERING DATA FROM RANK {global_rank}...''')        
        fabric.barrier()
        
        # Truncate dataset in dim=0 if we oversample
        rewards=torch.from_numpy(np.stack(rewards)).flatten()
        if rewards.shape[0]>local_buffer_size: 
            logger.info(f'RANK {local_rank} OVERSAMPLED OVER LOCAL BUFFER SIZE LIMIT, TRUNCATING DATASET...')
            rewards=rewards[:local_buffer_size]
            batched_inputs['input_ids']=batched_inputs['input_ids'][:local_buffer_size,:]
            batched_inputs['attention_mask']=batched_inputs['attention_mask'][:local_buffer_size,:]
            
        local_data={
            'rewards':rewards, # shape (local_buffer_size,)
            'batched_inputs':batched_inputs # 'input_ids' = shape Tensor[B, Seq_len], 'attention_mask': Tensor[B, Seq_len]]
            }
        
        # flatten data across rank and batch size rewards = size [rank * B] and batched_inputs = [rank * B, T]
        gathered_data = fabric.all_gather(local_data)
        gathered_data['rewards']=gathered_data['rewards'].flatten()
        
        max_seq_len=gathered_data['batched_inputs']['input_ids'].shape[-1]
        gathered_data['batched_inputs']['input_ids']=gathered_data['batched_inputs']['input_ids'].reshape(-1, max_seq_len)  
        gathered_data["batched_inputs"]['attention_mask']=gathered_data["batched_inputs"]['attention_mask'].reshape(-1, max_seq_len) 
        
        logger.debug(f'REWARDS SHAPE: {gathered_data["rewards"].shape}')
        logger.debug(f'INPUT_IDS SHAPE: {gathered_data["batched_inputs"]["input_ids"].shape}')
        logger.debug(f'ATTENTION_MASK SHAPE: {gathered_data["batched_inputs"]["attention_mask"].shape}')
        
        fabric.log_dict({'TOTAL REWARD FOR ALL RANKS':torch.sum(rewards)})
        
        train(fabric, agent, optimizer, lr_scheduler, gathered_data, args, logger)  
        
        # saving the model 
        # if episode % args.checkpointing_steps == 0:
        #     if fabric.global_rank==0:
        #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        #         if args.checkpoints_total_limit is not None:
        #             checkpoints = os.listdir(args.output_dir)
        #             checkpoints = [d for d in checkpoints if d.startswith("critic_checkpoint")]
        #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if episode % args.checkpointing_steps == 0:
            if fabric.global_rank == 0:
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("critic_checkpoint") and d.endswith(".ckpt")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                        
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"critic_checkpoint-{episode}.ckpt")
                state = AttributeDict(model1=agent, optimizer=optimizer)
                fabric.save(save_path, state)
                logger.info(f"SAVING MODEL TO: {save_path}")
    
if __name__=="__main__": 
    
    main()