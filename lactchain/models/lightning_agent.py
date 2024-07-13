from textwrap import dedent
from typing import Any, List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
import torch
import numpy as np
import torch.nn as nn, torch.nn.functional as F
import lightning as pl
from torch import Tensor
from lactchain.models.actor import (ActorConfig, LactChain, LoraConfigSettings)
from peft import PeftModel
from lactchain.models.critic import (ValueFunctionConfig, ValueFunction)
from lactchain.configs.base_config import BaseConfig
from lactchain.models.backends.huggingface_backend import (HuggingFaceGenerator, 
                                                           HuggingFaceGeneratorConfig, 
                                                           LoraConfigSettings)

class LightningA2C(pl.LightningModule): 
    '''Lightning Class that contains Critic and Actor Models for Lactchain style inference'''
    MODEL_MAP:dict[str, str]={
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3', 
        'meta-llama/Meta-Llama-3-8B':'llama-3',
        'mistralai/Mistral-7B-v0.3': 'mistral-7b', 
        'mistralai/Mixtral-8x7B-Instruct-v0.1':'mistral-7b'
        }
    
    def __init__(self, 
                 actor_model:str,
                 actor_model_type:str,
                 actor_config:ActorConfig,
                 lora_config:LoraConfigSettings,
                 critic_model:str,
                 critic_config:ValueFunctionConfig, 
                 gamma=float
                 ):
        '''Lightning Model that Joins Frozen Actor and Trainable Critic'''
        super().__init__()
        
        assert actor_model_type in self.MODEL_MAP.values() , f'''Current supported models are only llama-3 8B models and mistral 7B-V0.3 and Mixtral 8x7B'''
        
        self.actor=LactChain(actor_model, actor_model_type, actor_config, lora_config)
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
        self._final_prompt_template=self.actor.compile_prompts('<STATE_GOES_HERE>', '<INFO_GOES_HERE>')
        
        self._actor_model=self.actor.generator.model # returns the type of actor model 
        
    @property
    def actor_model(self): 
        '''Returns the model structure of actor model'''
        return self._actor_model
    
    @property
    def model_trainable_params(self): 
        '''Returns total number of parameters of model'''
        return self._model_trainable_params
    
    @property
    def final_prompt_template(self):
        '''Return the final prompt template'''
        return self._final_prompt_template
    
    @torch.inference_mode()
    def sample_actions(self,
                       states:Dict[str, Any],
                       infos:str
                       ) -> Tuple[list[np.ndarray], list[str], list[str], list[int]]:
        '''Actor samples actions'''
        batch_mapped_actions, actions, contexts, drop_indices = self.actor.sample_actions(states, infos)
    
        return batch_mapped_actions, actions, contexts, drop_indices
    
    @torch.inference_mode()
    def compile_and_tokenize(self, 
                             states:Dict[str, Any] | list[Dict[str, Any]], 
                             infos:Dict[str, Any] | list[Dict[str, Any]]
                             ) -> Tensor: 
        '''compile and tokenize the strings'''
        return self.critic.compile_and_tokenize(states, infos)
    
    def decode(self, 
               inputs:Dict[str, Tensor]
               ) -> list[str]: 
        '''De-tokenize and return the original strings'''
        return self.critic.decode_tokens(inputs)

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
    
    def calculate_returns(self, rewards:Tensor | list[Tensor]) -> Tensor:
        '''Function that calculates tensor of returns from a tensor or list of batch tensor rewards'''

        if isinstance(rewards, Tensor): 
            cumulative_returns=self._calc_returns(rewards)
        elif isinstance(rewards, list): 
            cumulative_returns=torch.stack(self._calc_returns_list(rewards))
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                    (cumulative_returns.std() + 1e-12)

        return cumulative_returns
    
    def calculate_value(self,
                        **inputs:Dict[str, Any]
                        ) -> Tensor: 
        '''Critic calculates value by batch'''
        pred_q_values = self.critic(**inputs)
        return pred_q_values
    
    @torch.inference_mode()
    def calculate_advantages(self, 
                             rewards:Tensor | np.ndarray, 
                             inputs:Dict[str, Tensor]
                             ) -> Tensor: 
        '''
        Calculates Advantages Tensor Given a Tensor of Rewards shape [B] and inputs [B, T]
        '''
        if isinstance(rewards, np.ndarray): 
            rewards=torch.from_numpy(rewards)
            
        rewards=rewards.to(self.device)  
        inputs=inputs.to(self.device)
        values=self(**inputs)
        cumulative_returns=self.calculate_returns(rewards).to(values.device)
        advantages=(values-cumulative_returns)
        
        return advantages
    
    def forward(self, **inputs:Dict[str, Any]) -> Tensor: 
        return self.calculate_value(**inputs)
    
    def training_step(self, 
                      rewards:Tensor, 
                      inputs:Dict[str, Tensor]
                      ) -> Tensor:
        '''Function that takes in a reward tensor of shape B and a Dict of inputs where the input_ids are 
        of shape B x Seq_len
        
        Inputs: 
        ======
        Rewards: Tensor
            Tensor of rewards of shape B 
        Inputs: Dict[str, Tensor]
            Inputs dictionary of shape B x Seq_len that 
            contains batch inputs for model 
            
        Output: 
        ======
        critic_loss: Tensor 
            loss of shape B
        '''
        values=self(**inputs)
        cumulative_returns=self.calculate_returns(rewards).to(values.device)
        critic_loss=F.smooth_l1_loss(cumulative_returns, values)

        return critic_loss
    
    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
    
    # def save(self, save_path:str):
    #     self.peft_model.save_pretrained(save_path)
    #     torch.save(self.linear.state_dict(), f"{save_path}/linear_layer.pth")

    # def load(self, load_path:str):
    #     self.peft_model = PeftModel.from_pretrained(self.base_model, load_path)
    #     self.linear.load_state_dict(torch.load(f"{load_path}/linear_layer.pth"))

    
    def state_dict(self, *args, **kwargs):
        '''Overriding state dict to save only lora + q-value head'''
        state = super().state_dict(*args, **kwargs)
        for name in list(state.keys()):
            if "lora" not in name and "q_value_head" not in name:  # <-- adapt the condition to your use case
                state.pop(name)
        return state
    
    def load_state_dict(self, state_dict:Dict[str, Any], strict=True, *args, **kwargs):
        '''Overriding super class for loading in model state dicts for Lora and Q value Linear Head'''
        # Create a new state dict with only matching keys for LoRA and q_value_head
        lora_and_q_value_head_state = {k: v for k, v in state_dict.items() if "lora" in k or "q_value_head" in k}
        super().load_state_dict(lora_and_q_value_head_state, strict=strict, *args, **kwargs)

