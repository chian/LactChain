from __future__ import annotations

'''Base Value Function Chain'''

from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
import torch
from torch import Tensor, nn, functional as F
from textwrap import dedent
from pydantic import Field
from typing import Any, Union, Dict, Tuple, List, Optional, Literal
from peft import LoraModel, LoraConfig, PeftModel, get_peft_model

from lactchain.configs import BaseConfig

class LoraConfigSettings(BaseConfig):
    '''Base Config for Lora'''

    r: int = Field(8)
    lora_alpha: int = Field(32)
    target_modules: List[str] = Field(["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout: float = Field(0.05)
    bias: str = Field('all')
    task_type: str = Field("FEATURE_EXTRACTION")

class EmbedderFunctionConfig(BaseConfig):
    '''Value function Base Config'''

    pretrained_model_name_or_path: str = Field(
        default=...,
        description='what pretrained model name or path to use'
    )
    use_lora: bool = Field(
        True,
        description='whether or not to use lora adapters or not'
    )
    lora_config_settings: LoraConfigSettings = Field(
        default_factory=LoraConfigSettings,
        description='Default Lora config settings'
    )
    load_from_checkpoint: str = Field(
        None,
        description='Path to any previously loaded checkpoints'
    )
    max_seq_length: int = Field(
        128,
        description='total max length of sequences'
    )
    torch_dtype: str = Field(
        'torch.float32',
        description='dtype of model'
    )
    gradient_checkpointing_enable: bool = Field(
        True,
        description='Whether to enable gradient checkpointing or not'
    )
    quantization: bool = Field(
        True,
        description='Whether to use quantization.',
    )
    half_precision: bool = Field(
        True,
        description='Whether to use half precision.',
    )
    compile_model: bool = Field(
        False,
        description='Whether to compile the model for faster inference.',
    )
    enable_flash_attention: bool = Field(
        True,
        description='Whether to enable flash attention on model or not'
    )
    device_map_auto: bool = Field(
        False,
        description='Whether to enable auto device map'
    )
    enable_sdpa: bool = Field(
        True,
        description="Whether to enable sdpa attnetion or not via torch context manager"
    )


class EmbedderFunction(nn.Module):
    '''Model that is used for embedding text'''

    def __init__(self, config: EmbedderFunctionConfig):
        super().__init__()
        model_kwargs = {}

        # set device_map_auto if needed
        if config.device_map_auto:
            model_kwargs['device_map'] = 'auto'

        dtype = torch.float32
        if config.dtype == 'float16':
            dtype = torch.float16
        elif config.dtype == 'bfloat16':
            dtype = torch.bfloat16
        model_kwargs['torch_dtype'] = dtype

        if config.quantization:
            from transformers import BitsAndBytesConfig

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs['quantization_config'] = nf4_config

        if config.enable_flash_attention:
            model_kwargs['attn_implementation'] = "flash_attention_2"

        model = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path, **model_kwargs)

        if config.use_lora:
            lora_config = LoraConfig(**config.lora_config_settings.model_dump())
            model = get_peft_model(model, lora_config, adapter_name='default')

        if config.load_from_checkpoint:
            model = PeftModel.from_pretrained(
                model, config.load_from_checkpoint)

        if config.gradient_checkpointing_enable:
            gradient_checkpointing_kwargs = {}
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        # Compile the model for faster inference
        if config.compile_model:
            model = torch.compile(model, fullgraph=True)
        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # freeze non lora weights
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        self.model = model
        self.model_dtype = next(model.parameters()).dtype
        self.q_value_head = nn.Linear(model.config.hidden_size, 1, dtype=dtype)
        self._total_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad] +
            [p.numel() for p in self.q_value_head.parameters() if p.requires_grad]
        )

        self.config = config
        
    @property
    def total_params(self):
        return self._total_params

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, config: EmbedderFunctionConfig):
        critic = cls(checkpoint, config)
        return critic
    
    def forward(self,**inputs: Dict[str, Any]) -> Tensor:
        
        with torch.autocast(device_type="cuda"):
            
            if self.config.enable_sdpa:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, 
                                                    enable_math=False, 
                                                    enable_mem_efficient=False):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            q_values = self.q_value_head(last_hidden_states[:, 0, :])
            pred_q_values = q_values.mean(dim=-1)

        return pred_q_values  # shape B x 1
    
    

    # @torch.inference_mode()
    # def compile_and_tokenize(self,
    #                          states: Dict[str, Any] | list[Dict[str, Any]],
    #                          infos: Dict[str, Any] | list[Dict[str, Any]]
    #                          ) -> Tensor:
    #     '''Function that compiles the states + infos info one list, then tokenizes it'''

    #     states = [states] if isinstance(states, dict) else states
    #     infos = [infos] if isinstance(infos, dict) else infos
    #     states = [str(state) for state in states]
    #     infos = [str(info['info']) for info in infos]

    #     states = [states+'\n'+info for states, info in zip(states, infos)]
    #     inputs = self.tokenizer(
    #         states, **self.tokenizer_call_kwargs).to(self.model.device)

    #     return inputs

    # @torch.inference_mode()
    # def decode_tokens(self,
    #                   inputs: Dict[str, Tensor]
    #                   ) -> list[str]:
    #     '''Function that compiles the states + infos info one list, then tokenizes it'''

    #     input_ids = inputs['input_ids']
    #     decoded_strings = self.tokenizer.batch_decode(
    #         input_ids, **self.tokenizer_decode_kwargs)

    #     return decoded_strings
