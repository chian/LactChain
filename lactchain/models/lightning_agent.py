from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import torch
import numpy as np
import torch.nn as nn, torch.nn.functional as F
import pprint as pp
import json
import lightning as pl
from torch import Tensor
from lactchain.models.actor import (ActorConfig, LactChain, 
                                    STRATEGY, PROMPT_TEMPLATE, 
                                    Strategy, ListOfMoves, 
                                    LoraConfigSettings)

from lactchain.models.critic import (ValueFunctionConfig, ValueFunction)
from lactchain.configs.base_config import BaseConfig
from lactchain.models.backends.langchain_backend import LangChainGenerator, GeneratorConfig
from lactchain.models.backends.vllm_backend import VLLMGeneratorConfig, VLLMGenerator
from lactchain.models.backends.huggingface_backend import (HuggingFaceGenerator, 
                                                           HuggingFaceGeneratorConfig, 
                                                           LoraConfigSettings)

class LightningA2C(pl.LightningModule): 
    def __init__(self, 
                 actor_model:str,
                 actor_config:ActorConfig,
                 lora_config:LoraConfigSettings,
                 critic_model:str,
                 critic_config:ValueFunctionConfig, 
                 gamma=float
                 ):
        '''Lightning Model that Joins Frozen Actor and Trainable Critic'''
        super().__init__()
        
        self.actor=LactChain(actor_model, actor_config, lora_config)
        self.critic=ValueFunction(critic_model, critic_config)
        
        self.gamma=gamma

        for param in self.actor.parameters():
            param.requires_grad = False
        
    def sample_actions(self,
                       states:Dict[str, Any],
                       infos:str
                       ) -> Tuple[list[str], str]:
        '''Actor samples actions'''
        mapped_actions, actions, contexts = self.actor.sample_actions(states, infos)
        return mapped_actions, actions, contexts
    
    def calculate_value(self,
                        states:Dict[str, Any] | list[Dict[str, Any]], 
                        info:Optional[str | list[str]]=None
                        ) -> Tensor: 
        '''Critic calculates value by batch'''
        pred_q_values = self.critic(states, info)
        return pred_q_values
    
    def _calc_returns_list(self, rewards:List[int]) -> List[np.ndarray]:
        '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
        Input: Sequence[int] -> Output: Sequence[torch(int)]
        '''
        returns=[]
        R = 0
        for r in rewards[::-1]:
            R = (r + self.gamma*R).clone()
            returns.insert(0, R)
        return returns
    
    def _calc_returns(self, rewards:Tensor) -> Tensor:
        '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
        Input: Sequence[int] -> Output: Sequence[torch(int)]
        '''
        # returns = torch.zeros_like(rewards, dtype=torch.float)
        returns = torch.zeros(rewards.shape).to(rewards.device)
        R = torch.tensor(0.0)
        for idx, r in enumerate(torch.flip(rewards, dims=[0])):
            R = r + self.gamma*R
            returns[idx]=R
        return returns
    
    def calculate_returns(self, rewards:Tensor) -> Tensor:
        
        cumulative_returns=self._calc_returns(rewards)
        # cumulative_returns=torch.stack(cumulative_returns)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                    (cumulative_returns.std() + 1e-12)

        return cumulative_returns
    
    def forward(self, states, info) -> Tensor: 
        return self.calculate_value(states, info)
    
    def training_step(self, batch: Dict[str, Tensor]):

        rewards=batch['rewards']
        observations=batch['observations']
        infos=batch['infos']
        
        values=self(observations, infos)
        cumulative_returns = self.calculate_returns(rewards).to(values.device)
        breakpoint()
        critic_loss = F.smooth_l1_loss(cumulative_returns, values)
        breakpoint()
        print(f"Critic loss: {critic_loss.item()}")  # Debug print
        
        return critic_loss
    
    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
    
    
    
if __name__=="__main__": 
    
    
    breakpoint()
