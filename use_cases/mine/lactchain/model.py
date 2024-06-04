import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pydantic import BaseModel, Field
from typing import Optional, Union, Any, Dict, List
import sys, os
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')
from use_cases.mine.lactchain.lactchains import MyLactChain, Strategy
from use_cases.mine.lactchain.critic import ValueFunction
from use_cases.mine.lactchain.config import BaseConfig
from use_cases.mine.lactchain.my_huggingface_backend import HuggingFaceGeneratorConfig, MyHuggingFaceGenerator


class ActorConfig(BaseConfig): 
    generatorconfig:HuggingFaceGeneratorConfig=Field(
        default_factory=HuggingFaceGeneratorConfig
    )
    

    def __init__(self, model:'str'): 
        super().__init__(model=model)


class ActorModelForCausalLM(AutoModelForCausalLM): 
    def __init__(self, config:ActorConfig):
        super().__init__(config)
        self.model=AutoModelForCausalLM

    def forward(self, input_ids, attention_mask, **kwargs): 
        ...


if __name__=="__main__": 

    MODEL="mistralai/Mistral-7B-Instruct-v0.3"

    config=AutoConfig.from_pretrained(MODEL)

    model=ActorModelForCausalLM(config)

    breakpoint()
    ...


