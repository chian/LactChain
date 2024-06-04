"""Module for the hugging face generator backend."""

from __future__ import annotations

from typing import Literal, Union, TypeVar, Optional, Dict
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


class HuggingFaceGenerator:
    """Language model generator using hugging face backend."""

    def __init__(self, config: HuggingFaceGeneratorConfig, model_type:Optional[str]=None) -> None:
        """Initialize the HuggingFaceGenerator."""
        import torch
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM
        
        model_kwargs = {'device_map':'auto'}

        # Use quantization
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

        # Set persistent attributes
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _generate_batch(self, prompts: str | list[str] | list[Dict[str, str]]) -> list[str]:
        """Generate text from a batch encoding.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[str]
            A list of generated responses.
        """
        if self.model_type=="causal": 
            batch_encoding=self.tokenizer.apply_chat_template(prompts, 
                                                              add_generation_prompt=True, 
                                                              return_tensors="pt", 
                                                              padding='longest'
                                                              )
            breakpoint()
            input_ids = batch_encoding  
        else:         
            # Tokenize the prompts
            batch_encoding = self.tokenizer(
                prompts,
                padding='longest',
                return_tensors='pt',
            )

        # Move the batch_encoding to the device
        batch_encoding = batch_encoding.to(self.model.device)
        if self.model_type=='causal': 
            generated_text=self.model.generate(batch_encoding, 
                                               top_p=self.config.top_p, 
                                               num_return_sequences=1, 
                                               num_beams=self.config.num_beams, 
                                               do_sample=self.config.do_sample, 
                                               max_new_tokens=1000)
        else: 
            # Generate text using top-p sampling
            generated_text = self.model.generate(
                **batch_encoding,
                top_p=self.config.top_p,
                num_return_sequences=1,
                num_beams=self.config.num_beams,
                do_sample=self.config.do_sample,
            )

        # Decode the generated text
        decoded_outputs = self.tokenizer.batch_decode(
            generated_text,
            skip_special_tokens=True,
        )

        breakpoint()
        if self.model_type == 'causal':
            input_lengths = [len(self.tokenizer.decode(input_id, skip_special_tokens=True)) for input_id in input_ids]
            responses = [decoded_output[input_length:].strip() for decoded_output, input_length in zip(decoded_outputs, input_lengths)]
        else:
            responses = decoded_outputs
        return responses

    def generate(self, prompts: str | list[str]) -> list[str]:
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
        # Ensure that the prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.model_type=='causal': # if statement for causal 
            prompts = [{'role':'user', 'content': prompt} for prompt in prompts]

        # Generate responses from the prompts
        responses = []
        for batch in batch_data(prompts, self.config.batch_size):
            responses.extend(self._generate_batch(batch))

        breakpoint()
        return responses
    


if __name__=="__main__": 
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from textwrap import dedent
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    from transformers import pipeline
    from langchain_core.prompts import PromptTemplate
    MODEL="mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                                 device_map='auto', 
                                                #  attn_implementation="flash_attention_2", 
                                                 torch_dtype=torch.bfloat16
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    chatbot = pipeline("text-generation", model=model, 
                        tokenizer=tokenizer, device_map='auto', max_new_tokens=1000)
    hf = HuggingFacePipeline(pipeline=chatbot)

    template="""{question}"""
    prompt = PromptTemplate.from_template(template)

    # chain = prompt | hf
    # question = "Can you write a short novel for me?"
    # output=chain.invoke({"question": question})

    breakpoint()

    # gpu_chain = prompt | hf.bind(stop=["\n\n"])

    gpu_chain=prompt | hf

    questions=['Give me a summary on how cancer develops in humans', 'User:\nwrite me a short story\nAI:']

    answers = gpu_chain.batch(questions)
    input_lengths=[len(question) for question in questions]
    answers_processed=[answer[input_length:] for answer, input_length in zip(answers, input_lengths)]

    breakpoint()





    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        # {"role": "user", "content": "Who are you?"},
    ]
    from transformers import pipeline
    MODEL="mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000)
    output=chatbot(messages)

    breakpoint()

    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000)
    hf = HuggingFacePipeline(pipeline=pipe)

    from langchain_core.prompts import PromptTemplate

    # template = dedent("""Question: {question}

    # Answer: Let's think step by step.""")

    template="""{question}"""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    question = "What is electroencephalography?"

    output=chain.invoke({"question": question})

    breakpoint()


    tokenizer=AutoTokenizer.from_pretrained('microsoft/phi-2')
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained('microsoft/phi-2', 
                                               trust_remote_code=True, 
                                               device_map='auto')

    device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu',
            )
    batch_encoding=tokenizer(
                list_prompts,
                padding='longest',
                return_tensors='pt',
            ).to(device)
    
    generated_text = model.generate(
    **batch_encoding,
    top_p=0.95,
    num_return_sequences=1,
    num_beams=10,
    do_sample=True)

    outputs=tokenizer.batch_decode(generated_text)
    outputs_skip=tokenizer.batch_decode(generated_text)



    breakpoint()

    dict_prompts = [
    {"role": "user", "content": "Hello how are you doing?"},
    {"role": "user", "content": "Can you write a poem for me?"}
    ]

    chat_tokenizer=AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
    chat_batch_encoding=chat_tokenizer.apply_chat_template(dict_prompts,
                                                            padding='longest',
                                                            return_tensors='pt',
                                                            return_dict=True
                                                            )

    # batch_encoding = tokenizer.apply_chat_template(
    # prompts,
    # add_generation_prompt=True,
    # return_tensors="pt",
    # padding='longest'
    # )

    # batch_encoding=Tokenizer.apply_chat_template(
    #     prompts, 
    #     add_generation
    # )

    breakpoint()

    other_prompts=[
        'hello how are you doing', 
        'write me a poem'
    ]

    llama_tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    llama_tokenizer.pad_token=llama_tokenizer.eos_token
    llama_batch_encoding=llama_tokenizer(
                other_prompts,
                padding='longest',
                return_tensors='pt',
            )

    breakpoint()
    ...

