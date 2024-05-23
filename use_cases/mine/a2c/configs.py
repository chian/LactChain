from pydantic import BaseModel, Field
from typing import Dict
import torch 

class ActorConfig(BaseModel): 
    tokenizer_args:Dict=Field(
        {
        'return_tensors':'pt', 
        'return_attention_mask':True
        }
    )
    model_args:Dict=Field(
        {
        'pretrained_model_name_or_path':'microsoft/phi-2',
        # 'device_map':'auto', 
        'torch_dtype':torch.float32
        }
    )
    model_forward_args:Dict=Field(
        {
        'output_attentions':True, 
        'output_hidden_states':True, 
        }
    )
    pooling_type:str=Field(
        'last_token'
    )

class CriticConfig(BaseModel): 
    name:str=Field('nvidia/Llama3-ChatQA-1.5-8B')