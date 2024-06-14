
import gymnasium as gym
from typing import List, Callable, Dict, Any, Tuple, Optional
import sys, os
from dataclasses import dataclass, field
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
from peft import LoraConfig, LoraModel
from datasets import load_from_disk
from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from argparse import ArgumentParser
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
from transformers import BitsAndBytesConfig
sys.path.append(os.getcwd()+'/../../../')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.dataset import DPODataset
from use_cases.mine.lactchain.SimPOTrainer import SimPOTrainer


@dataclass
class SimPOConfig(DPOConfig):
    gamma: Optional[float] = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    
    
def argparse() -> Tuple[Any]: 
    argparse=ArgumentParser()
    argparse.add_argument('--train_data_path', type=str, default='./')
    argparse.add_argument('--actor_path', type=str, 
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--bits_and_bytes', type=bool, default=True)
    argparse.add_argument('--lora', type=bool, default=True)
    argparse.add_argument('--half_precision', type=bool, default=True)
    args=argparse.parse_args()
    
    hfparser=H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    model_args, data_args, training_args = hfparser.parse()
    return args, model_args, data_args, training_args

def get_models(args:ArgumentParser, 
              actor_config:PolicyConfig, 
              lora_config:Optional[LoraConfigSettings]
              ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    model_kwargs={'device_map':'auto'}
    
    if args.bits_and_bytes: 
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )   
        model_kwargs['quantization_config'] = nf4_config
        
    model = AutoModelForCausalLM.from_pretrained(
        args.actor_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
    args.actor_path,
    trust_remote_code=True,
    )
    if args.lora: 
        lora_config=LoraConfig(**lora_config.model_dump())
        model=LoraModel(model, lora_config, adapter_name='default')

    tokenizer.model_max_length = model.config.max_position_embeddings
    tokenizer.pad_token=tokenizer.eos_token
    
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.half_precision:
        model.half()

    return model, tokenizer
    
    
def main():
    
    args, model_args, data_args, training_args = argparse()
    
    set_seed(training_args.seed)
    
    train_dataset=load_from_disk(args.train_data_path)
    
    lora_config=LoraConfigSettings()
    actor_config=PolicyConfig() 
    
    model, tokenizer = get_models(args, actor_config, lora_config)
    
    ref_model=None

    trainer = SimPOTrainer(model=model,
                        ref_model=ref_model, # pass in to bypass DPO Trainer check for ref model but is not actually used
                        # model_init_kwargs=model_kwargs,
                        args=training_args,
                        beta=training_args.beta,
                        train_dataset=train_dataset,
                        tokenizer=tokenizer,
                        max_length=training_args.max_length,
                        max_prompt_length=training_args.max_prompt_length,
                        peft_config=get_peft_config(model_args),
                        loss_type=training_args.loss_type,
                        )
    
    trainer.train()
    
    print(f'FINISHED TRAINING')
    
    
if __name__=="__main__": 
    
    main()