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


class LightningCritic(pl.LightningModule): 
    '''Lightning Class that contains Critic and Actor Models for Lactchain style inference'''
    MODEL_MAP:dict[str, str]={
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3', 
        'meta-llama/Meta-Llama-3-8B':'llama-3',
        'mistralai/Mistral-7B-v0.3': 'mistral-7b', 
        'mistralai/Mixtral-8x7B-Instruct-v0.1':'mistral-7b'
        }
    
    def __init__(self, 
                 actor:LactChain,
                #  backend:str,
                #  actor_model:str,
                #  actor_model_type:str,
                #  actor_config:ActorConfig,
                 lora_config:LoraConfigSettings,
                 critic_model:str,
                 critic_config:ValueFunctionConfig, 
                 gamma=float
                 ):
        '''Lightning Model that Uses Actor to Generate'''
        super().__init__()