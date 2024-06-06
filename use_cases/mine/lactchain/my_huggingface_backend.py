"""Module for the hugging face generator backend."""

from __future__ import annotations

from typing import Literal, Union, TypeVar, Optional, Dict, Any
from pathlib import Path
from pydantic import Field
import sys, os
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')

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

    def __init__(self, config: HuggingFaceGeneratorConfig, 
                 model_kwargs:Optional[Dict[str, Any]]=None, 
                 pipeline_kwargs:Optional[Dict[str, Any]]=None) -> None:
        """Initialize the HuggingFaceGenerator."""
        import torch
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import pipeline
        from langchain_core.prompts import PromptTemplate

        model_kwargs={'device_map':'auto'}

        pipeline_kwargs={'device_map':'auto', 'max_new_tokens':1000}

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

        # Set the model max length for proper truncation
        tokenizer.model_max_length = model.config.max_position_embeddings
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

        chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, **pipeline_kwargs)
        hf_pipeline = HuggingFacePipeline(pipeline=chat_pipeline)
        prompt = PromptTemplate.from_template('{input}')
        chain = prompt | hf_pipeline

        # Set persistent attributes
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.chain = chain

    def _generate_batch(self, prompts:list[str]) -> list[str]: 
        '''generates batch outputs and then filters out attached input prompt via string slicing'''
        input_lengths=[len(prompt) for prompt in prompts]
        raw_outputs = self.chain.batch(prompts)
        breakpoint()
        outputs=[raw_output[input_length:] for raw_output, input_length in zip(raw_outputs, input_lengths)]
        return outputs

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
        for batch in batch_data(prompts, self.config.batch_size):
            responses.extend(self._generate_batch(batch))
        return responses
    
if __name__=="__main__": 

    config=HuggingFaceGeneratorConfig()
    config.pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"
    generator=MyHuggingFaceGenerator(config)
    inputs=['can you give me a sci-fi story?', 
            'What is the difference between the stack and the heap in coding?']
    outputs=generator.generate(inputs)
    breakpoint()