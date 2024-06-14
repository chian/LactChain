"""Module for the hugging face generator backend."""

from __future__ import annotations
import torch
from torch import nn
from typing import Literal, Union, TypeVar, Optional, Dict, Any, List
from pathlib import Path
from pydantic import Field
import sys, os
from peft import prepare_model_for_kbit_training, LoraModel, LoraConfig
sys.path.append(os.getcwd()+'/../../../')

from use_cases.mine.lactchain.config import BaseConfig

PathLike = Union[str, Path]

T = TypeVar('T')

def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size.

    Parameters
    ----------
    data : list[T]
        The data to batch.
    chunk_size : int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The batched data.
    """
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches

class LoraConfigSettings(BaseConfig): 
    r:int=Field(8)
    lora_alpha:int=Field(32)
    target_modules:List[str]=Field(["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout:float=Field(0.05)
    bias:str=Field('all')
    task_type:str=Field("CAUSAL_LM")

class HuggingFaceGeneratorConfig(BaseConfig):
    """Configuration for the HuggingFaceGenerator."""

    # name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]
    pretrained_model_name_or_path: str = Field(
        'None',
        description='The model id or path to the pretrained model.',
    )
    half_precision: bool = Field(
        False,
        description='Whether to use half precision.',
    )
    eval_mode: bool = Field(
        True,
        description='Whether to set the model to evaluation mode.',
    )
    compile_model: bool = Field(
        False,
        description='Whether to compile the model for faster inference.',
    )
    quantization: bool = Field(
        True,
        description='Whether to use quantization.',
    )
    top_p: float = Field(
        0.95,
        description='The top p for sampling.',
    )
    num_beams: int = Field(
        10,
        description='The number of beams for sampling.',
    )
    do_sample: bool = Field(
        True,
        description='Whether to use sampling.',
    )
    batch_size: int = Field(
        2,
        description='The number of prompts to process at once.',
    )


class MyHuggingFaceGenerator:
    """Language model generator using hugging face backend."""

    def __init__(self, 
                 config: HuggingFaceGeneratorConfig,
                 lora_config:Optional[LoraConfigSettings]=None, 
                 model_kwargs:Optional[Dict[str, Any]]=None, 
                 pipeline_kwargs:Optional[Dict[str, Any]]=None
                 ) -> None:
        
        super().__init__()
        
        """Initialize the HuggingFaceGenerator."""
        import torch
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import pipeline
        from langchain_core.prompts import PromptTemplate

        model_kwargs={'device_map':'auto'}

        self.tokenizer_call_kwargs={'return_tensors':'pt', 
                                    'padding':'longest'}

        if config.quantization:
            from transformers import BitsAndBytesConfig

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model_kwargs['quantization_config'] = nf4_config

        model = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model_name_or_path,
                trust_remote_code=True,
                **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        if lora_config: 
            lora_config=LoraConfig(**lora_config.model_dump())
            model=LoraModel(model, lora_config, adapter_name='default')

        # Set the model max length for proper truncation
        tokenizer.model_max_length = model.config.max_position_embeddings
        tokenizer.pad_token=tokenizer.eos_token
        # Convert the model to half precision
        if config.half_precision:
            model.half()
        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()
        # Load the model onto the device
        if not config.quantization:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu',
            )
            model.to(device)
        # Compile the model for faster inference
        if config.compile_model:
            model = torch.compile(model, fullgraph=True)
            
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        # Set persistent attributes
        self.model = model
        self.tokenizer = tokenizer
        self._config = config
        self._lora_config=lora_config
        
    @property
    def show_configs(self) -> Dict[str, Any]: 
        return {'model':self._config, 
                'lora':self._lora_config}

    def _generate_batch(self, prompts:list[str], **kwargs:Optional[Dict[str, Any]]) -> list[str]: 
        '''generates batch outputs and then filters out attached input prompt via token slicing'''

        _tokenizer_call_kwargs={'return_tensors':'pt', 'padding':'longest'}
        _model_call_kwargs={'num_return_sequences':1, 'max_new_tokens':500}

        batch_encoding=self.tokenizer(prompts, **_tokenizer_call_kwargs)
        batch_encoding = batch_encoding.to(self.model.device) # returns 
        input_tokens_len=batch_encoding['input_ids'].shape[-1]
        with torch.no_grad(): 
            generated_tokens=self.model.generate(**batch_encoding, **_model_call_kwargs)
        
        filtered_tokens=[row[input_tokens_len:] for row in generated_tokens]

        decoded_outputs=self.tokenizer.batch_decode(filtered_tokens, 
                                                    skip_special_tokens=True
                                                    )
        return decoded_outputs

    def generate(self, prompts:str | list[str]) -> list[str]: 
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        prompts=[prompts] if isinstance(prompts, str) else prompts
        responses = []
        for batch in batch_data(prompts, self._config.batch_size):
            responses.extend(self._generate_batch(batch))
        return responses

    
if __name__=="__main__": 

    config=HuggingFaceGeneratorConfig()
    config.pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
    generator=MyHuggingFaceGenerator(config)
    inputs=['can you give me a sci-fi story?', 
            'What is the difference between the stack and the heap in coding?',
            '''<s>[INST] You are an intelligent agi in grid-world that is of size 5x5. 
            You may only make one of the following moves to navigate: [move_forward, move_left]
            Propose a sequence of moves that will allow you to explore or get you to the goal. 
            All of your output must be stored in a json in the following format, and nothing else: 
            {{
            'explain': //Your explanation and logic goes here//
            'moves': // Your sequence of moves goes here// 
            }} 
            YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
            [/INST]''', 
            '''
            You are an intelligent agi in grid-world that is of size 5x5. 
            You may only make one of the following moves to navigate: [move_forward, move_left]
            Propose a sequence of moves that will allow you to explore or get you to the goal. 
            All of your output must be stored in a json in the following format, and nothing else: 
            {{
            'explain': //Your explanation and logic goes here//
            'moves': // Your sequence of moves goes here// 
            }} 
            YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
            '''
            ]
    outputs=generator.generate(inputs)
    breakpoint()