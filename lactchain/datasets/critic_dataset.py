from __future__ import annotations

from typing import Literal
import sys, os 
from lactchain.configs.base_config import BaseConfig

from typing import Any, Dict, Union, Optional, Callable
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import PreTrainedTokenizer

class CriticDataset(Dataset): 
    
    '''VLLM Dataset for Collecting Data for Critic Training. Contains Per Row: 
    
    [obs, info, rewards, next_obs, done, truncated]
    '''
    
    def __init__(self, 
                 data:Dict[str, Any],
                 save_path:str,
                 tokenizer:PreTrainedTokenizer
                 ) -> None:
        super().__init__()
        from functools import partial
        
        raw_dataset=HFDataset.from_dict(data)
        self._column_names=self._get_string_fields(data)

        tokenizer.pad_token=tokenizer.eos_token
        self._tokenizer_kwargs={
            'return_tensors':'pt', 
            'padding':'longest'
        }
        self.tokenize_process=partial(self.tokenize_process, tokenizer=tokenizer)
        
        dataset=raw_dataset.with_transform(self.tokenize_process)
        # setting attributes
        self.save_path=save_path
        self.tokenizer=tokenizer
        self.dataset=dataset
        
    def _get_string_fields(self, data:Dict[str, Any]) -> list[str]: 
        '''returns the keys of the dict where the fields are strings'''
        column_names = []
        for field in data.keys(): 
            if isinstance(data[field][0], str): 
                column_names.append(field)
                
        return column_names
    
    def tokenize_process(self, examples:list[Any], tokenizer:Callable, is_train:bool=True) -> list[Any]: 
        '''Tokenize the text observations for each column in a dataset'''
        
        # column names
        text_data = [] # list that stores the text data
        for column_name in self._column_names:
            for text in examples[column_name]: # iterate through list of data 
                if isinstance(text, str): 
                    text_data.append(text)
                
            examples[column_name]=[self._tokenize_observations(self.tokenizer, example) 
                                      for example in text_data]
    
        return examples
    
    def _tokenize_observations(self, tokenizer:PreTrainedTokenizer, observations:str):
        '''Function for tokenizing captions per row to be used for preprocess mapping'''
        inputs = tokenizer(
            observations, 
            **self._tokenizer_kwargs
        )
        return inputs
    
    def save_dataset(self, save_path:str) -> None: 
        self.dataset.save_to_disk(save_path)
    
    
    def add_batch_to_dataset(self, batch_data:Dict[str, Any]) -> None: 
        '''Adds a New batch of data to an already made huggingface dataset'''
        temp_raw_dataset=HFDataset.from_dict(batch_data)
        temp_dataset=temp_raw_dataset.with_transform(self.tokenize_process)
        
        self.dataset=concatenate_datasets(self.dataset, temp_dataset)
        
    def __len__(self) -> int: 
        return len(self.dataset)
    
    def __getitem__(self, index:int) -> Any:
        return self.dataset[index]
    
    def __repr__(self) -> Any: 
        return repr(self.dataset)
    
    def __str__(self) -> str: 
        return f'Dataset: {self.dataset}'
    
    
if __name__=="__main__": 
    
    from lactchain.models.backends.vllm_backend import VLLMGenerator, VLLMGeneratorConfig
    from transformers import AutoTokenizer
    VLLM = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    
    config = VLLMGeneratorConfig(pretrained_model_name_or_path=VLLM)
    vllm = VLLMGenerator(config)
    
    tokenizer = AutoTokenizer.from_pretrained(VLLM)
    # test case
    obs = ['hello how are you doing']*50
    next_obs = vllm.generate(obs)
    reward = [-1]*50
    
    DATA={
        'observation':obs, 
        'joint_info':...,
        'reward':reward
        }
    
    dataset=CriticDataset(DATA, save_path='./', tokenizer=tokenizer)
    
    
    breakpoint()