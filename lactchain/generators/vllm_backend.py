"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations
from typing import Literal, Any
import sys
import os
from lactchain.configs.base_config import BaseConfig


class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the VLLMGenerator."""

    name: Literal['vllm'] = 'vllm'  # type: ignore[assignment]
    # The name of the vllm LLM model, see
    # https://docs.vllm.ai/en/latest/models/supported_models.html
    # llm_name: str = None
    # Whether to trust remote code
    trust_remote_code: bool = True
    # Temperature for sampling
    temperature: float = 0.0
    # Min p for sampling
    min_p: float = 0.1
    # Top p for sampling (off by default)
    top_p: float = 0.0
    # Max tokens to generate
    max_tokens: int = 500
    # Whether to use beam search
    use_beam_search: bool = False
    # The number of GPUs to use
    tensor_parallel_size: int = 2
    # pretrained model name or path
    pretrained_model_name_or_path: str = None
    # percentage of gpu to utilize
    gpu_memory_utilization: float = 0.9
    # no cuda graph
    enforce_eager: bool = True
    # quantization
    quantization: str = "bitsandbytes"
    # load format for quantization
    load_format: str = "bitsandbytes"


class VLLMGenerator:
    """Language model generator using vllm backend."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        """Initialize the VLLMGenerator.

        Parameters
        ----------
        config : vLLMGeneratorConfig
            The configuration for the VLLMGenerator.
        """
        from vllm import LLM
        from vllm import SamplingParams
        from openai import OpenAI
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
        # Create the sampling params to use
        sampling_kwargs = {}
        if config.top_p:
            sampling_kwargs['top_p'] = config.top_p
        else:
            sampling_kwargs['min_p'] = config.min_p

        # Create the sampling params to use
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            use_beam_search=config.use_beam_search,
            **sampling_kwargs,
        )

        # Create an LLM instance
        model = LLM(
            model=config.pretrained_model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            dtype='bfloat16',
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager
        )

        self.tokenizer_call_kwargs = {
            'return_tensors': 'pt',
            'padding': 'longest'
        }

        tokenizer = model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        self._config = config

    @property
    def config(self):
        return self._config

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

        # Generate responses from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt, generated text,
        # and other information.
        outputs = self.model.generate(prompts, self.sampling_params)

        # Extract the response from the outputs
        responses = [output.outputs[0].text for output in outputs]

        return responses

    def tokenize(self, prompts: str | list[str]) -> list[dict[str, Any]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = self.tokenizer(prompts, **self.tokenizer_call_kwargs)

        return inputs

# class VLLMGenerator:
#     """Language model generator using vllm backend."""

#     openai_api_key = "lactchain"
#     openai_api_base = "http://localhost:8000/v1"

#     def __init__(self,
#                  config: VLLMGeneratorConfig
#                  ) -> None:
#         """Initialize the VLLMGenerator.

#         Parameters
#         ----------
#         config : vLLMGeneratorConfig
#             The configuration for the VLLMGenerator.
#         """
#         from vllm import LLM
#         from vllm import SamplingParams
#         from openai import OpenAI
#         from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
#         # Create the sampling params to use
#         sampling_kwargs = {}
#         if config.top_p:
#             sampling_kwargs['top_p'] = config.top_p
#         else:
#             sampling_kwargs['min_p'] = config.min_p

#         if config.server:
#             model=OpenAI(api_key=config.api_key,
#                          base_url=config.host,
#                          )

#             tokenizer=AutoTokenizer.from_pretrained(
#                 config.pretrained_model_name_or_path,
#                 trust_remote_code=True
#                 )
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token=tokenizer.eos_token

#             self.call_kwargs={
#                 'top_p':config.top_p,
#                 'max_tokens':config.max_tokens,
#                 'temperature':config.temperature,
#             }

#         else:
#             # Create the sampling params to use
#             self.sampling_params = SamplingParams(
#                 temperature=config.temperature,
#                 max_tokens=config.max_tokens,
#                 use_beam_search=config.use_beam_search,
#                 **sampling_kwargs,
#             )

#             # Create an LLM instance
#             model = LLM(
#                 model=config.pretrained_model_name_or_path,
#                 trust_remote_code=config.trust_remote_code,
#                 dtype='bfloat16',
#                 tensor_parallel_size=config.tensor_parallel_size,
#                 gpu_memory_utilization=config.gpu_memory_utilization,
#                 enforce_eager=config.enforce_eager
#             )

#             tokenizer=model.get_tokenizer()
#             tokenizer.pad_token=tokenizer.eos_token

#         self.model=model
#         self.tokenizer=tokenizer
#         self._config=config

#     @property
#     def config(self):
#         return self._config


#     def generate(self, prompts: str | list[str]) -> list[str]:
#         if self.config.server:
#             responses=self.generate_server(prompts)
#         else:
#             responses=self.generate_local(prompts)
#         return responses

#     def generate_server(self, prompts: str | list[str]) -> list[str]:
#         '''Server Style responses from openai engine'''
#         # responses = self.model.chat.completions.create(
#         #     model=config.pretrained_model_name_or_path,
#         #     messages=[
#         #         {"role": "system", "content": "You are a helpful assistant."},
#         #         {"role": "user", "content": "Tell me a joke."},
#         #     ]
#         # )
#         breakpoint()
#         batch = self.model.completions.create(
#         model=self.config.pretrained_model_name_or_path,
#         prompt=prompts,
#         **self.call_kwargs
#         )
#         responses=[batch.choices[idx].text for idx in range(len(prompts))]

#         return responses

#     def generate_local(self, prompts: str | list[str]) -> list[str]:
#         """Generate response text from prompts.

#         Parameters
#         ----------
#         prompts : str | list[str]
#             The prompts to generate text from.

#         Returns
#         -------
#         list[str]
#             A list of responses generated from the prompts
#             (one response per prompt).
#         """
#         # Ensure that the prompts are in a list
#         if isinstance(prompts, str):
#             prompts = [prompts]

#         # Generate responses from the prompts. The output is a list of
#         # RequestOutput objects that contain the prompt, generated text,
#         # and other information.
#         outputs = self.model.generate(prompts, self.sampling_params)

#         # Extract the response from the outputs
#         responses = [output.outputs[0].text for output in outputs]

#         return responses


if __name__ == "__main__":

    from lactchain.models.prompts import Prompts

    prompt = Prompts('vllm', model_type='mistral-7b')

    ACTOR_PATH = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'

    config = VLLMGeneratorConfig()
    config.pretrained_model_name_or_path = ACTOR_PATH

    vllm = VLLMGenerator(config)
    tokenizer = vllm.return_tokenizer()
    prompts = [prompt.prompt_template]*2
    output = vllm.generate(prompts)

    breakpoint()
