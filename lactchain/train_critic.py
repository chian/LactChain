from typing import List, Callable, Dict, Any, Tuple, Union, Optional
from pathlib import Path
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import torch
import gymnasium as gym
from datasets import Dataset as HFDataset
from argparse import ArgumentParser
import os, uuid

from lactchain.environments.grid_world import GridEnvironment
from lactchain.models.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from lactchain.models.actor import LactChain, ActorConfig, Strategy
# from lactchain.utils import ...

PathLike=Union[str, Path]

def calc_returns(rewards:List[int], gamma:float) -> List[Tensor]:
    '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
    Input: Sequence[int] -> Output: Sequence[torch(int)]
    '''
    returns=[]
    R = 0
    for r in rewards[::-1]:
        R = r + gamma*R
        returns.insert(0, torch.tensor(R))
    return returns

def train_critic(actor:LactChain,
                 critic:ValueFunction,
                 critic_optimizer: torch.optim.AdamW,
                 critic_scheduler:torch.optim.lr_scheduler.CosineAnnealingLR,
                 env:GridEnvironment,
                 args:ArgumentParser,
                 ):
    print(f'STARTING TRAINING..., with model size {critic.total_params}')

    for episode in range(args.num_episodes):

        rewards=[]
        values=[]
        state, info = env.reset()
        done=0
        steps=0

        while not done and steps<=args.max_steps:
            try:
                action, context=actor.sample_action(state, info)
            except Exception as e: 
                print(f'action not validly parsed skipping...')
                continue
            
            value=critic(state, info)
            next_state, reward, done, info = env.step(action)

            print(f'STATE:{state}')
            print(f'ACTION: {action}')

            rewards.append(reward)
            values.append(value)

            state=next_state
            steps+=1

        cumulative_returns=calc_returns(rewards, args.gamma)
        cumulative_returns = torch.tensor(cumulative_returns)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                (cumulative_returns.std() + 1e-12)

        critic_losses=[]
        for R, value in zip(cumulative_returns, values):
            critic_losses.append(F.smooth_l1_loss(R, value)) # advantage = R - value

        critic_loss=torch.stack(critic_losses).mean()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if episode % 10 == 0:
            critic_scheduler.step()

        print(f'Episode total num steps: {steps}')
        print(f'Episode {episode+1} Loss: {critic_loss}')
        print(f'Episode {episode+1} Total Reward: {cumulative_returns.mean()}')

        # Save the final model and tokenizer
        checkpoint_dir = os.getcwd() + os.path.join(args.output_dir, f"checkpoint-{args.num_episodes}_episodes-{args.max_steps}_steps/")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(critic.state_dict(), checkpoint_dir + 'critic_checkpoint.pt')
        critic.tokenizer.save_pretrained(checkpoint_dir)

        print(f"Checkpoint saved to {checkpoint_dir}")
        print(f"Model and tokenizer saved to {args.output_dir}")
        
        
def argparse() -> Any:
    argparse=ArgumentParser()
    argparse.add_argument('--resume_from_checkpoint', type=str, default='/checkpoints/')
    argparse.add_argument('--output_dir', type=str, default='/checkpoints/')
    argparse.add_argument('--actor_path', type=str,
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--critic_path', type=str,
                          default='./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11')
    argparse.add_argument('--num_episodes', type=int, default=2)
    argparse.add_argument('--max_steps', type=int, default=3)
    argparse.add_argument('--gamma', type=float, default=0.99)
    args = argparse.parse_args()
    return args

def main():
    
    lora_config=LoraConfigSettings()
    actor_config=ActorConfig()
    critic_config=ValueFunctionConfig()

    args=argparse()
    # os.listdir()
    # '''
    # if args.resume_from_checkpoint: 
    #     lactchain_path = 
    #     list_dir = os.listdir(os.getcwd() + '')
    #     checkpoint_path = os.path.join()
    # '''

    # Check if resume_from_checkpoint is specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # Use the provided checkpoint path
            checkpoint_path = os.path.join(args.output_dir, os.path.basename(args.resume_from_checkpoint))
        else:
            # Find the most recent checkpoint in the output directory
            checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            
            if checkpoint_dirs:
                # Sort directories by checkpoint number
                checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(args.output_dir, checkpoint_dirs[-1])
            else:
                # No checkpoints found
                checkpoint_path = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            actor=LactChain(args.actor_path, actor_config, lora_config)
            critic = ValueFunction(args.critic_path, critic_config)
        else:
            print("No checkpoint found. Loading pretrained model.")
            actor=LactChain(args.actor_path, actor_config, lora_config)
            critic = ValueFunction(args.critic_path, critic_config)
    else:
        print("No checkpoint specified. Loading pretrained model.")
        

    env=GridEnvironment()

    actor=LactChain(args.actor_path, actor_config, lora_config)

    critic=ValueFunction(args.critic_path, critic_config)

    critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    critic_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=10)

    train_critic(actor, critic, critic_optimizer, critic_scheduler, env, args)


if __name__=="__main__":

    main()
