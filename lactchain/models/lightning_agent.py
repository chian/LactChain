from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import torch
import numpy as np
import torch.nn as nn
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
        
        # freeze weights of actor model 
        for param in self.actor.parameters():
            param.requires_grad = False
        
    @torch.no_grad()
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
    
    def _calc_returns(self, rewards:List[int]) -> List[Tensor]:
        '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
        Input: Sequence[int] -> Output: Sequence[torch(int)]
        '''
        returns=[]
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma*R
            returns.insert(0, torch.tensor(R))
        return returns
    
    @torch.no_grad()
    def calculate_returns_advantages(self, rewards:List[int], values:Any):
        cumulative_returns=self._calc_returns(rewards, self.gamma)
        cumulative_returns = torch.tensor(cumulative_returns)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                (cumulative_returns.std() + 1e-12)

        advantages=[]
        for R, value in zip(cumulative_returns, values):
            advantages.append((R - value)) # advantage = R - value

        return cumulative_returns, advantages
    
    
    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
