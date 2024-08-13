
# import gymnasium as gym
from typing import List, Callable, Dict, Any, Tuple, Optional
import sys, os
from dataclasses import dataclass, field
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
from peft import LoraConfig, LoraModel
from datasets import load_from_disk

from argparse import ArgumentParser
from peft import PeftConfig, PeftModel
from transformers import BitsAndBytesConfig

from lactchain.classes.base_lactchain import LactChain, Context, Component
from lactchain.environments.grid_world import GridEnvironment
from lactchain.models.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from lactchain.models.actor import LactChain, ActorConfig
from lactchain.datasets import dpo_dataset
from lactchain.simpo.simpo_trainer import SimPOTrainer
from lactchain.simpo.dpo_config import DPOConfig
from lactchain.simpo.dpo_collator import DPODataCollatorWithPadding
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig

@dataclass
class SimPOConfig(DPOConfig):
    gamma: Optional[float] = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    output_dir: Optional[str]=field(
        default='./'
    )

def get_models(args:ArgumentParser, 
              actor_config:ActorConfig, 
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
    
    
    
def argparse() -> Tuple[Any]: 
    argparse=ArgumentParser()
    argparse.add_argument('--train_data_path', type=str, default='./datasets/')
    argparse.add_argument('--actor_path', type=str, 
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--bits_and_bytes', type=bool, default=True)
    argparse.add_argument('--lora', type=bool, default=True)
    argparse.add_argument('--half_precision', type=bool, default=True)
    argparse.add_argument('--output_dir', type=str, default='./')
    args=argparse.parse_args()
    
    # hfparser=H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    # model_args, data_args, training_args = hfparser.parse()
    
    return args
    
def main():
    
    args = argparse()
    
    training_args = SimPOConfig()

    set_seed(training_args.seed)
    
    train_dataset=load_from_disk(args.train_data_path)
    # train_dataset.set_format("torch", columns=['prompt', 'chosen', 'rejected'], device="cuda")
    
    print(f'TYPE DATASET {type(train_dataset)}')

    lora_config=LoraConfigSettings()
    actor_config=ActorConfig() 
    
    model, tokenizer = get_models(args, actor_config, lora_config)
    # breakpoint()
    ref_model=None
    
    # model_kwargs = dict(
    # revision=model_args.model_revision,
    # trust_remote_code=model_args.trust_remote_code,
    # use_flash_attention_2=model_args.use_flash_attention_2,
    # torch_dtype=torch_dtype,
    # use_cache=False if training_args.gradient_checkpointing else True,
    # device_map=get_kbit_device_map() if quantization_config is not None else None,
    # quantization_config=quantization_config,
    # )
    lora_config_settings=LoraConfigSettings()
    lora_config=LoraConfig(**lora_config_settings.model_dump())
    data_collator=DPODataCollatorWithPadding()

    trainer = SimPOTrainer(model=model,
                        ref_model=ref_model, # pass in to bypass DPO Trainer check for ref model but is not actually used
                        model_init_kwargs=training_args.model_init_kwargs, # model_kwargs
                        args=training_args,
                        beta=training_args.beta,
                        train_dataset=train_dataset,
                        tokenizer=tokenizer,
                        max_length=training_args.max_length,
                        max_prompt_length=training_args.max_prompt_length,
                        # loss_type=training_args.loss_type,
                        peft_config=lora_config, 
                        # data_collator=data_collator
                        )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    print(f'FINISHED TRAINING')
    
    # pip install transformers==4.38.2
if __name__=="__main__": 
    
    main()