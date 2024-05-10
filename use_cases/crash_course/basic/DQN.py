from __future__ import annotations
import random, matplotlib.pyplot as plt, numpy as np
import pandas as pd, seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal
import gymnasium as gym 
from typing import Union, Tuple, Any
from collections import deque
from dataclasses import dataclass 
from pydantic import BaseModel, Field

@dataclass(frozen=True) # frozen --> immutable
class Transition: 
    state: Any
    action: Any
    reward: Any
    next_state: Any

class ReplayMemory: 
    '''Memory Bank'''
    def __init__(self, max_capacity:int): 
        self.max_capacity=max_capacity

        self.transitions=deque([], maxlen=self.max_capacity)

    def push(self, *args): 
        self.transitions.append(Transition(*args))
    
    def sample(self, batch_size:int):
        return random.sample(self.transitions, batch_size)

    def __len__(self): 
        return len(self.transitions)
    
    def __getitem__(self, idx): 
        return self.transitions[idx]
    
    def __repr__(self): 
        return f'Deque Max Capacity: {len(self.transitions)}, First item: {self.transitions[0]}, Last item: {self.transitions[-1]}'

class Model(BaseModel): 
    hidden1:int=Field(
        512
    )
    hidden2:int=Field(
        256
    )
    hidden3:int=Field(
        128
    )
    activation:str=Field(
        'ReLU'
    )

class Learn(BaseModel): 
    batch_size:int=Field(
        128
    )
    gamma:float=Field(
        0.99
    )
    eps_start:float=Field(
        0.9
    )
    eps_end:float=Field(
        0.05
    )
    eps_decay:float=Field(
        1000
    )
    tau:float=Field(
        0.005
    )
    lr:float=Field(
        1e-4
    )

class BaseConfig(BaseModel): 
    modelcfg:Model=Field(
        default_factory=Model
    )
    learncfg:Learn=Field(
        default_factory=Learn
    )

class DQN(nn.Module): 
    def __init__(self, n_obs:int, n_actions:int, config:BaseConfig): 
        super().__init__()
        self.config=config

        activations={
            'ReLU':nn.ReLU(), 
            'TanH':nn.Tanh()
                     }
        
        self.activation=activations[config.activation]
        self.layer1=nn.Linear(n_obs, self.config.hidden1)
        self.layer2=nn.Linear(self.config.hidden2, self.config.hidden3)
        self.layer3=nn.Linear(self.config.hidden3, n_actions)

        self.block=nn.Sequential(self.layer1, self.activation, self.layer2)

    def forward(self, x:Tensor) -> Tensor: 
        out = self.block(x)
        return out
    

def env_info(env:Any) -> Tuple[Any, Any]:
    '''return n_action, num_obs'''
    n_actions=env.action_space.shape
    n_obs=env.observation_space.sample
    return n_obs, n_actions

if __name__=="__main__": 

    memory=ReplayMemory(10000)



    memory.push(*[0, 1, 2, 3])
    memory.push(4, 5, 6, 7)

    sample=memory.sample(1)

    item=memory[0]

    length=len(memory)

    breakpoint()
