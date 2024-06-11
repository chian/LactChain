from peft import get_peft_model, LoraConfig, get_peft_config
from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os
import numpy as np
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Any, Union, Dict, Tuple, List, Optional, Literal
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig
from transformers import BitsAndBytesConfig
from torch import Tensor
sys.path.append(os.getcwd()+'/../../../')
from classes.learning import LearningScheme
from use_cases.mine.lactchain.config import BaseConfig

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
    model_name:str=Field('meta-llama/Meta-Llama-3-8B')
    tokenizer_name:str=Field('meta-llama/Meta-Llama-3-8B')
    torch_dtype:str=Field('torch.float32')

class ValueFunction(nn.Module): 
    '''Config is type ValueFunctionConfig class and will dump sub-configs or attr into the model'''
    def __init__(self, 
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
            _trunk_model=AutoModel.from_pretrained(config.model_name, torch_dtype=torch.float32)
            self.model=LoraModel(_trunk_model, self.lora_config, "default")
        elif config.peft_type=='qlora':
            _quant_config=BitsAndBytesConfig(load_in_8bit=True)
            _trunk_model=AutoModel.from_pretrained(config.model_name, 
                                                   torch_dtype=torch.float32, 
                                                   quantization_config=_quant_config)
            _model = prepare_model_for_kbit_training(_trunk_model)
            self.model = get_peft_model(_model, self.lora_config)
            
        self.q_value_head = nn.Linear(self.model.config.hidden_size, 1)

        # tokenizer setup
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.model_max_length = min(self.model.config.max_position_embeddings, 
                                              config.max_seq_length)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __repr__(self):
        PEFT_CONFIG=self.config.lora_config if self.config.lora_config is not None else "None"
        return dedent(f'''
                      Model Type: {self.config.model_name}, 
                      PEFT Config: {PEFT_CONFIG},
                      Tokenizer: {self.config.tokenizer_name}, 
                      Trainable Parameters: {self.model.print_trainable_parameters()}
                      ''')

    def forward(self, 
                states:Dict[str, Any] | list[Dict[str, Any]], 
                info:Optional[str | list[str]]=None
                ) -> Tensor: 
        '''TODO: Should this be the whole strategy prompt, or just state + info stringified?'''
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


class ValueFunctionLearningScheme(LearningScheme):
    def __init__(self, model):
        self.model = model
        self.trainer = None
        self.training_args = TrainingArguments(
            output_dir="./model_output",
            num_train_epochs=1,  # Set to 1 for episodic training
            per_device_train_batch_size=16,
            learning_rate=1e-5,  # Adjusted for PEFT
            logging_dir='./logs',
            logging_steps=10,
        ) 

    def update_model(self, train_dataset):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            compute_loss=self.compute_loss
        )
        self.trainer.train()

    def compute_loss(self,batch):
        # Use the precomputed predicted Q-values from the dataset
        predicted_q_values = batch['predicted_q_value']
        # Extract the target Q-values from the batch
        target_q_values = batch['target_q_value']
        loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)
        return loss
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


if __name__=="__main__": 



    config=ValueFunctionConfig()
    valuefunction=ValueFunction(config)

    states=[{'x':3, 'y':4, 'orientation':'right'}, {'x':4, 'y':5, 'orientation':'left'}]
    info=['grid world is size 5', 'grid world is size 6']

    values=valuefunction(states, info)

    breakpoint()