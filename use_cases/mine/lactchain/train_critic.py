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
from pathlib import Path
sys.path.append(os.getcwd()+'/../../../')
from classes.learning import LearningScheme
from use_cases.mine.lactchain.config import BaseConfig
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfig, LoraConfigSettings
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.lactchains import MyLactChain

os.environ['HF_HOME']=os.getcwd()+'/../../../../'

PathLike=Union[str, Path]

import lightning as pl
        

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

def save_model(model:Any, save_path:PathLike): 
    assert isinstance(save_path, str), f'Path must be a string'
    model.save_pretrained(save_path, from_pt=True) 

def train_critic(actor:MyLactChain, 
                 critic:ValueFunction, 
                 critic_optimizer: torch.optim.AdamW, 
                 critic_scheduler:torch.optim.lr_scheduler.CosineAnnealingLR, 
                 env:GridEnvironment, 
                 fabric:Optional[Fabric]=None, 
                 save_path:Optional[PathLike]=None
                 ): 
    import time
    TOTAL_PARAMS=sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f'STARTING TRAINING..., with model size {TOTAL_PARAMS}')
    if fabric: 
        actor = fabric.setup(actor)
        critic, critic_optimizer = fabric.setup(critic, critic_optimizer)
    
    MAX_EPISODES=10000
    GAMMA=0.99
    MAX_STEPS=20
    
    for episode in range(MAX_EPISODES): 
        
        rewards=[]
        values=[]
        
        state, info = env.reset()
        done=0
        steps=0
        while not done and steps<=MAX_STEPS:         
     
            print(f'state:{state}')
            action, context=actor.sample_action(state, info)
            print(f'ACTION: {action}')
            time.sleep(3)
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
        # fabric.backward(critic_loss) if fabric else critic_loss.backward() 
        critic_loss.backward() 
        critic_optimizer.step()
        
        if episode % 10 == 0: 
            critic_scheduler.step()
        
        print(f'Episode {episode+1} Loss: {critic_loss}')
        print(f'Episode {episode+1} Total Reward: {cumulative_returns.mean()}')
        
    if save_path: 
        save_model(critic, save_path)
    

    
# class ActorCritic(pl.LightningModule): 
#     def __init__(self, 
#                  actor_model:PathLike, 
#                  actor_config:PolicyConfig, 
#                  critic_model:PathLike, 
#                  critic_config:ValueFunctionConfig,
#                  lora_config:Optional[LoraConfigSettings], 
#                  gamma:float
#                  ): 
#         super().__init__()
#         self.lactchain=MyLactChain(actor_model, actor_config, lora_config, './')
#         self.critic=ValueFunction(critic_model, critic_config)
#         self.gamma=gamma
        
#         # freeze model weights of lactchain 
#         for param in self.lactchain.generator.model.params():
#             param.requires_grad = False
            

#     def forward(self, state:str, info:str) -> Tuple[list[str], str, Tensor]: 
#         action, context=self.actor.sample_action(state, info)
#         value=critic(state, info)
#         return action, context, value
    
#     def training_step(self, batch:Tensor): 
        
#         return ...
    
#     def configure_optimizers(self) -> Optimizer | F.Sequence[Optimizer] | Tuple[F.Sequence[Optimizer] | F.Sequence[Any | ReduceLROnPlateau | LRSchedulerConfig]] | OptimizerLRSchedulerConfig | F.Sequence[OptimizerLRSchedulerConfig] | None:
#         return super().configure_optimizers()
    
    
    
if __name__=="__main__": 
    
    LACTCHAIN_PATH='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    CRITIC_PATH="./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
    
    lora_config=LoraConfigSettings()
    lactchain_config=PolicyConfig()
    lactchain=MyLactChain(LACTCHAIN_PATH, lactchain_config, lora_config, './')
    # print(list(lactchain.parameters()))
    # lactchain_optimizer=torch.optim.AdamW(lactchain.parameters(), lr=1e-4)

    critic_config=ValueFunctionConfig()
    critic=ValueFunction(CRITIC_PATH, critic_config)

    critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    critic_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=10)
    
    # actor_critic=ActorCritic(LACTCHAIN_PATH, lactchain_config, CRITIC_PATH, critic_config, lora_config, 0.99, False)
    
    env=GridEnvironment()

    # fabric=Fabric(accelerator='cuda', devices=[0, 1], strategy='FSDP')
    # fabric.launch()
    
    # actor_critic=fabric.setup(actor_critic)
    
    # obs, info = env.reset()
    
    # action, info, value = actor_critic(obs, info)
    
    train_critic(lactchain, critic, critic_optimizer, critic_scheduler, env)
    
    breakpoint()
