from peft import get_peft_model, LoraConfig, get_peft_config
from transformers import TrainingArguments, Trainer, AutoModel, AutoTokenizer
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os
import numpy as np
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Any, Union, Dict, Tuple, List, Optional
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig
from transformers import BitsAndBytesConfig
from torch import Tensor
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.learning import LearningScheme
from use_cases.mine.lactchain.config import ValueFunctionConfig

class ValueFunction(nn.Module): 
    '''Config is type ValueFunctionConfig class and will dump sub-configs or attr into the model'''
    def __init__(self, 
                 config:ValueFunctionConfig
                 ): 
        super().__init__()

        self.config=config
        # model setup 
        self.lora_config=LoraConfig(**config.lora_config.model_dump())
        self.trunk_model=AutoModel.from_pretrained(config.model_name, torch_dtype=torch.float32)
        
        if config.peft_type=='lora': 
            _trunk_model=AutoModel.from_pretrained(config.model_name, torch_dtype=torch.float32)
            self.model=LoraModel(_trunk_model, self.lora_config, 'lora_adapter')
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
        self.tokenizer.pad_token = '[PAD]' if self.tokenizer.pad_token_id is None else None

    def __repr__(self):
        PEFT_CONFIG=self.config.lora_config if self.config.lora_config is not None else "None"
        return dedent(f'''
                      Model Type: {self.config.model_name}, 
                      PEFT Config: {PEFT_CONFIG},
                      Tokenizer: {self.config.tokenizer_name}, 
                      Trainable Parameters: {self.model.print_trainable_parameters()}
                      ''')

    def forward(self, 
                input_ids:Tensor, 
                attention_mask:Optional[Tensor]=None
                ) -> Tensor: 
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        q_values = self.q_value_head(last_hidden_states[:, 0, :])  # Using the first token's representation
        pred_q_value = q_values.mean()  # Take the mean of the first logit
        return pred_q_value


class ValueFunctionLearningScheme(LearningScheme): 
    def __init__(self, model:AutoModel, config): 

        ...



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

    breakpoint()