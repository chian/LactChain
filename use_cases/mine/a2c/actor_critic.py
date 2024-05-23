import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym 
import sys, os
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel, Field
from transformers.utils import ModelOutput
from typing import Dict
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')

class ActorConfig(BaseModel): 
    tokenizer_args:Dict=Field(
        {
        'padding':True, 
        'return_tensors':'pt', 
        'return_attention_mask':True
        }
    )

    model_args:Dict=Field(
        {
        'pretrained_model_name_or_path':'nvidia/Llama3-ChatQA-1.5-8B',
        'device_map':'auto', 
        'torch_dtype':torch.float32
        }
    )

class CriticConfig(BaseModel): 
    name:str=Field('nvidia/Llama3-ChatQA-1.5-8B')

class Actor(nn.Module): 
    def __init__(self, config:ActorConfig): 
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(config.model_args['pretrained_model_name_or_path'])
        self.model=AutoModel.from_pretrained(**config.model_args)

    def forward(self, state:str) -> ModelOutput: 

        inputs=self.tokenizer(state)(**self.config.tokenizer_args)

        return inputs

if __name__=="__main__": 

    # env=gym.make('')

    actorconfig=ActorConfig()
    actor=Actor(actorconfig)

    print('DONE')

