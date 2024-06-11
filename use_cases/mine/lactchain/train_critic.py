from peft import get_peft_model, LoraConfig, get_peft_config
from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
import sys, os
import numpy as np
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Any, Union, Dict, Tuple, List, Optional, Literal
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig
from transformers import BitsAndBytesConfig
from torch import Tensor
from lightning_fabric import Fabric
sys.path.append(os.getcwd()+'/../../../')
from classes.learning import LearningScheme
from use_cases.mine.lactchain.config import BaseConfig
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfig
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.lactchains import MyLactChain

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

def train_critic(actor:MyLactChain, 
                 critic:ValueFunction, 
                 critic_optimizer: torch.optim.AdamW, 
                 critic_scheduler:torch.optim.lr_scheduler.CosineAnnealingLR, 
                 env:GridEnvironment, 
                 fabric:Optional[Fabric]=None
                 ): 

    if fabric: 
        actor = fabric.setup(actor)
        critic, critic_optimizer = fabric.setup(critic, critic_optimizer)
    
    MAX_EPISODES=10000
    GAMMA=0.99
    MAX_STEPS=100
    
    for episode in range(MAX_EPISODES): 
        
        rewards=[]
        values=[]
        
        state, info = env.reset()
        done=0
        steps=0
        while not done and steps<=MAX_STEPS: 
            
            action, context=actor.sample_action(state, info)
            value=critic(state, info)
            next_state, reward, done, info = env.step(action)
        
            rewards.append(reward)
            values.append(value)
            
            state=next_state
            steps+=1
            
        cumulative_returns=calc_returns(rewards, GAMMA)
        cumulative_returns = torch.tensor(cumulative_returns)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                (cumulative_returns.std() + 1e-12)
        
        critic_losses=[]
        for R, value in zip(cumulative_returns, values): 
            advantage = R - value
            critic_losses.append(F.smooth_l1_loss(R, value))
            
        critic_loss=torch.stack(critic_losses).mean()
        
        critic_optimizer.zero_grad()
        # critic_loss.backward()
        fabric.backward(critic_loss)
        critic_optimizer.step()
        
        if episode % 10 == 0: 
            critic_scheduler.step()
        
        print(f'Episode {episode+1} Loss: {critic_loss}') if fabric.global_rank==0 else None
        print(f'Episode {episode+1} Total Reward: {cumulative_returns.mean()}') if fabric.global_rank==0 else None
    

if __name__=="__main__": 
    
    
    lactchain_config=PolicyConfig()
    lactchain=MyLactChain(lactchain_config, "mistralai/Mistral-7B-Instruct-v0.3", './')
    # print(list(lactchain.parameters()))
    # lactchain_optimizer=torch.optim.AdamW(lactchain.parameters(), lr=1e-4)
    

    critic_config=ValueFunctionConfig()
    critic=ValueFunction(critic_config)
    critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    critic_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=10)
    
    env=GridEnvironment()
    
    fabric=Fabric(accelerator='cuda', devices=[0, 1, 2])
    fabric.launch()
    
    train_critic(lactchain, critic, critic_optimizer, critic_scheduler, env, fabric)
    
    breakpoint()