from peft import get_peft_model, LoraConfig, get_peft_config
from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer
import torch
from torch import Tensor, nn, functional as F
from textwrap import dedent
from pydantic import Field
from typing import Any, Union, Dict, Tuple, List, Optional, Literal
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig
from transformers import BitsAndBytesConfig
from torch import Tensor

from lactchain.configs.base_config import BaseConfig

class LoraConfigSettings(BaseConfig): 
    r:int=Field(8)
    lora_alpha:int=Field(32)
    target_modules:List[str]=Field(["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout:float=Field(0.05)
    bias:str=Field('all')
    task_type:str=Field("SEQ_CLS")

class ValueFunctionConfig(BaseConfig): 
    bb_config:Any=Field(None)
    peft_type:Literal['lora', 'qlora']=Field('lora')
    lora_config_settings:LoraConfigSettings=Field(default_factory=LoraConfigSettings)
    printer:bool=Field(True)
    max_seq_length:int=Field(128)
    torch_dtype:str=Field('torch.float32')

class ValueFunction(nn.Module): 
    '''Config is type ValueFunctionConfig class and will dump sub-configs or attr into the model'''
    def __init__(self, 
                 model_name:str,
                 config:ValueFunctionConfig
                 ): 
        super().__init__()

        self.config=config
        self.tokenizer_call_kwargs={
            'return_tensors':'pt',
            'padding':'longest'
        }
        # model setup 
        self.lora_config=LoraConfig(**config.lora_config_settings.model_dump())
        if config.peft_type=='lora': 
            _trunk_model=AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            self.model=LoraModel(_trunk_model, self.lora_config, "default")
        elif config.peft_type=='qlora':
            _quant_config=BitsAndBytesConfig(load_in_8bit=True)
            _trunk_model=AutoModel.from_pretrained(model_name, 
                                                   torch_dtype=torch.float32, 
                                                   quantization_config=_quant_config)
            _model = prepare_model_for_kbit_training(_trunk_model)
            self.model = get_peft_model(_model, self.lora_config)
            
        self.q_value_head = nn.Linear(self.model.config.hidden_size, 1)

        # tokenizer setup
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = min(self.model.config.max_position_embeddings, 
                                              config.max_seq_length)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    @classmethod
    def load_from_checkpoint(cls, checkpoint:str, config:ValueFunctionConfig):
        critic=cls(checkpoint, config)
        return critic

    def forward(self, 
                states:Dict[str, Any] | list[Dict[str, Any]], 
                info:Optional[str | list[str]]=None
                ) -> Tensor: 
        states=[states] if isinstance(states, dict) else states
        states=[str(state) for state in states]
        if info is not None: 
            info=[info] if isinstance(info, str) else info
            states=[states+'\n'+info for states, info in zip(states, info)]
        
        inputs = self.tokenizer(states, **self.tokenizer_call_kwargs)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        q_values = self.q_value_head(last_hidden_states[:, 0, :])  # Using the first token's representation
        pred_q_values = q_values.mean(dim=-1)  # Take the mean of the first logit
        return pred_q_values # shape B x 1


if __name__=="__main__": 

    config=ValueFunctionConfig()
    valuefunction=ValueFunction(config)

    states=[{'x':3, 'y':4, 'orientation':'right'}, {'x':4, 'y':5, 'orientation':'left'}]
    info=['grid world is size 5', 'grid world is size 6']

    values=valuefunction(states, info)

    breakpoint()