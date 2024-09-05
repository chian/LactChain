from __future__ import annotations

'''Critic Dataset'''

from typing import Literal
import sys, os 
from pathlib import Path
from lactchain.configs.base_config import BaseConfig

from typing import Any, Dict, Union, Optional, Callable
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import PreTrainedTokenizer

PathLike = Union[Path, str]

class CriticDataset(Dataset): 
    
    '''VLLM Dataset for Collecting Data for Critic Training. Contains Per Row: 
    
    [obs, info, rewards, next_obs, done, truncated]
    '''
    
    def __init__(self, input_path: PathLike) -> None:
        super().__init__()
        from functools import partial
        
        self.dataset=HFDataset.load_from_disk(input_path)

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
        
    def __len__(self) -> int: 
        return len(self.dataset)
    
    def __getitem__(self, index:int) -> Any:
        return self.dataset[index]
    
    def __repr__(self) -> Any: 
        return repr(self.dataset)
    
    def __str__(self) -> str: 
        return f'Dataset: {self.dataset}'