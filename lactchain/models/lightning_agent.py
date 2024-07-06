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
            
        self._total_params=sum(
            [p.numel() for p in self.actor.parameters()] + 
            [p.numel() for p in self.critic.parameters()] + 
            [p.numel() for p in self.critic.q_value_head.parameters()]
            )
            
        self._total_trainable_params=sum(
            [p.numel() for p in self.actor.parameters() if p.requires_grad] + 
            [p.numel() for p in self.critic.parameters() if p.requires_grad] + 
            [p.numel() for p in self.critic.q_value_head.parameters() if p.requires_grad]
            )
        
        self._model_trainable_params=dedent(f'''Total number of parameters: {self._total_params:,}\nTotal number of trainable parameters: {self._total_trainable_params:,}\nPercentage of parameters trained: {(self._total_trainable_params/self._total_params) * 100}%''')
        
    @property
    def model_trainable_params(self): 
        return self._model_trainable_params
        
    @torch.inference_mode()
    def sample_actions(self,
                       states:Dict[str, Any],
                       infos:str
                       ) -> Tuple[list[np.ndarray], list[str], list[str], list[int]]:
        '''Actor samples actions'''
        batch_mapped_actions, actions, contexts, drop_indices = self.actor.sample_actions(states, infos)
    
        return batch_mapped_actions, actions, contexts, drop_indices

    
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
        returns = torch.zeros(rewards.shape).to(rewards.device)
        R = torch.tensor(0.0)
        for idx, r in enumerate(torch.flip(rewards, dims=[0])):
            R = r + self.gamma*R
            returns[idx]=R
        return returns
    
    def calculate_returns(self, rewards:Tensor) -> Tensor:
        
        cumulative_returns=self._calc_returns(rewards)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                    (cumulative_returns.std() + 1e-12)

        return cumulative_returns
    
    def calculate_value(self,
                        states:Dict[str, Any] | list[Dict[str, Any]], 
                        info:Optional[str | list[str]]=None
                        ) -> Tensor: 
        '''Critic calculates value by batch'''
        pred_q_values = self.critic(states, info)
        return pred_q_values
    
    def forward(self, states, info) -> Tensor: 
        return self.calculate_value(states, info)
    
    def training_step(self, batch: Dict[str, Tensor]):

        rewards=batch['rewards']
        observations=batch['observations']
        infos=batch['infos']
        values=self(observations, infos)
        cumulative_returns = self.calculate_returns(rewards).to(values.device)
        critic_loss = F.smooth_l1_loss(cumulative_returns, values)

        return critic_loss
    
    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
    
    def state_dict(self):
        '''Overriding state dict to save only lora + q-value head'''
        state = super().state_dict()
        for name in list(state.keys()):
            if "lora" not in name and "q_value_head" not in name:  # <-- adapt the condition to your use case
                state.pop(name)
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        # Create a new state dict with only matching keys for LoRA and q_value_head
        lora_and_q_value_head_state = {k: v for k, v in state_dict.items() if "lora" in k or "q_value_head" in k}
        super().load_state_dict(lora_and_q_value_head_state, strict=strict)

