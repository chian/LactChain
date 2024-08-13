from __future__ import annotations

'''class for Critic tokenizer aka value function'''

from typing import Any
from pydantic import Field
from torch import Tensor

from lactchain.configs import BaseConfig

class CriticTokenizerConfig(BaseConfig): 
    '''Base Config for tokenizer calling and decoding'''
    pretrained_model_name_or_path: str = ...
    
    call_kwargs: dict[str, Any] = Field(
        default={
            'return_tensors': 'pt',
            'padding': 'longest'
        }
    )
    decode_kwargs: dict[str, Any] = Field(
        default={
            'skip_special_tokens': True
        }
    )
    max_seq_length: int = Field(
        default=500, 
        description='the max sequence length for tokenization'
    )

class CriticTokenizer: 
    '''Tokenizer class for tokenization of strings'''
    
    def __init__(self, config: CriticTokenizerConfig, template: Any): 
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        tokenizer.model_max_length = config.max_seq_length
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
                
        self._call_kwargs = config.call_kwargs
        self._decode_kwargs = config.decode_kwargs
        self.tokenizer = tokenizer
        self._template = template
        
    @property
    def template(self): 
        return self._template
    
    @template.setter
    def template(self, new_template: Any): 
        self._template = new_template
        
    def tokenize(self, prompts: str | list[str]) -> dict[str, Tensor]: 
        '''Tokenizes a batch of prompts'''
        
        if isinstance(prompts, str): 
            prompts = [prompts]
        
        inputs = self.tokenizer(prompts, **self._call_kwargs)
        return inputs
    
    def decode(self, inputs: dict[str, Tensor]) -> list[str]: 
        '''Decodes a batch of inputs into list of outputs by llm'''

        input_ids = inputs['input_ids']
        
        outputs = self.tokenizer.batch_decode(
            input_ids, **self.tokenizer_decode_kwargs)
        
        return outputs
    
