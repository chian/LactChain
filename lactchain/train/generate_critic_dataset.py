from __future__ import annotations

from argparse import ArgumentParser
import logging
import gymnasium as gym
import torch
from torch import nn, functional as F, Tensor
import pprint as pp
import numpy as np
from tqdm import tqdm
import sys, math

from lactchain.datasets.critic_dataset import CriticDataset
from lactchain.models.actor import ActorConfig, LoraConfigSettings, LactChain
from lactchain.models.critic import ValueFunctionConfig, ValueFunction
from lactchain.environments.grid_world import make_env, process_environment_outputs
from lactchain.utils.utils import configure_logger, add_inputs_to_dict, unfold_list_of_lists, filter_fake_tensors

ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
ACTOR_MODEL_TYPE='llama-3'

CRITIC_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11'

def argparse(): 
    parser=ArgumentParser()
    parser.add_argument(
        '--backend', 
        type=str, 
        default='vllm', 
        help='''Path to frozen causal model'''
        )
    parser.add_argument(
        '--actor_path', 
        type=str, 
        default=ACTOR_PATH, 
        help='''Path to frozen causal model'''
        )
    parser.add_argument(
        '--actor_model_type', 
        type=str, 
        default=ACTOR_MODEL_TYPE, 
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
        "--log_wandb_offline",
        type=bool,
        default=True,
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
        default=1000, 
        help='''The desired max length of the buffer after gathering from all local ranks'''
        )
    parser.add_argument(
        '--collection_batch_size', 
        type=int, 
        default=256, 
        help='''The batch size when collecting experience for buffer; also the same as the environment'''
        )
    
    return parser.parse_args()


def generate_critic_experience(args:ArgumentParser, 
                               logger:logging.Logger, 
                               environment:gym.Env, 
                               actor:LactChain, 
                               critic:ValueFunction
                               ): 
    
    num_collection_steps = args.global_buffer_size // args.collection_batch_size
    max_tries_per_rank=math.ceil(num_collection_steps * 2)
    
    progress_bar = tqdm(
        range(0, num_collection_steps),
        initial=0,
        leave=False,
        desc=f"COLLECTING DATA FOR BUFFER OF SIZE {args.global_buffer_size}",
        file=sys.stdout
    )
    
    for episode in range(args.num_episodes): 
        
        rewards=[]
        # batched_inputs={}
        steps_kept=0
        buffer_size=0
        
        observations_list=[]
        infos_list=[]
        # joint_prompts=[]
        
        observations, infos = environment.reset()
        observations, infos=process_environment_outputs(observations, infos)
        
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad(): 
                
                exit_early=False
                step=0
                
                while buffer_size<args.global_buffer_size: 
                    step+=1
                    try: 
                        
                        batch_mapped_actions, _actions, _contexts, drop_indices=actor.sample_actions(observations, infos)
                        next_observations, reward, _done, _truncated, infos = environment.step(batch_mapped_actions)
                        next_observations, infos=process_environment_outputs(next_observations, infos)
                        # inputs=critic.compile_and_tokenize(next_obs, info)                        
                        rewards.append(reward)
                        observations_list.append(observations)
                        infos_list.append(infos)
                        # joint_prompt=[str(observation) + '\n' + str(info) for observation, info in zip(observations, infos)]
                        # joint_prompts.append(joint_prompt)
                        # batched_inputs=add_inputs_to_dict(batched_inputs, inputs)
                        observations = next_observations
                        # logger.debug(f'''Successfully parsed {args.collection_batch_size-len(drop_indices)} actions out of batch size {args.collection_batch_size} in step {step}, adding to replay buffer...\n''')
                        steps_kept+=1
                    except Exception as e: 
                        # if args.show_parsing_errors:
                            # logger.error(f'''Error when collecting experience:\n{e}\nDropping Full Batch for Step {step}...\n''')
                        # logger.debug(f'''PROPOSED OUTPUTS:\n{pp.pformat(actor.outputs)}\n\n\n''')
                        print(f'Error in parsing, dropping full batch...')
                        continue
                    
                    buffer_size = len(np.concatenate(rewards))
                    logger.info(f'Buffer Size {buffer_size}')
                    progress_bar.update(1)
                    logger.info(str(progress_bar)+'\n')
                    
                    if step > max_tries_per_rank: 
                        # logger.warn(f'MAX NUM TRIES EXCEEDED, EXITING....')
                        exit_early=True
                        break
                    
        breakpoint()
            
        DATA={
            'observation':observations,
            'reward':reward
            }
        breakpoint()
        dataset=CriticDataset(DATA, './', tokenizer=critic.tokenizer)
                    
        breakpoint()
            
            
            
def main(): 
    args=argparse()
    
    logger = configure_logger(args.logging_level, args.logging_save_path)
        
    actor_config=ActorConfig(backend='vllm')
    lora_config=LoraConfigSettings()
    critic_config=ValueFunctionConfig()
    
    environment = gym.vector.AsyncVectorEnv([make_env for _ in range(args.collection_batch_size)])
    actor=LactChain(args.backend, args.actor_path, args.actor_model_type, actor_config, lora_config)
    critic=ValueFunction(args.critic_path, critic_config)
    
    generate_critic_experience(args, logger, environment, actor, critic)
    
    
    
    

    
    
    
if __name__=="__main__": 
    
    
    main()