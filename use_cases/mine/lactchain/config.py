from __future__ import annotations
from pydantic import Field, BaseModel
from typing import Any, Union, Dict, Tuple, List

import json
from pathlib import Path
from typing import Literal
from typing import TypeVar
from typing import Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

PathLike = Union[str, Path]

T = TypeVar('T')


class BaseConfig(BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, 'w') as fp:
            json.dump(self.model_dump(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, 'w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)

class LoraConfig(BaseConfig): 
    r:int=Field(8)
    lora_alpha:int=Field(32)
    target_modules:List[str]=Field(["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout:float=Field(0.05)
    bias:str=Field('all')
    task_type:str=Field("SEQ_CLS")

class ValueFunctionConfig(BaseConfig): 
    bb_config:Any=Field(None)
    peft_type:Literal['lora', 'qlora']=Field('qlora')
    lora_config:LoraConfig=Field(default_factory=LoraConfig)
    printer:bool=Field(True)
    max_seq_length:int=Field(128)
    model_name:str=Field('meta-llama/Meta-Llama-3-8B')
    tokenizer_name:str=Field('meta-llama/Meta-Llama-3-8B')
    torch_dtype:str=Field('torch.float32')


class HuggingfaceConfig(BaseConfig): 
    modelname:str=Field("meta-llama/Meta-Llama-3-70B")
    tokenizer_args:Dict=Field(
            {
            'padding':True,
            'return_tensors':'pt',
            'return_attention_mask':True 
            }
    )

if __name__=="__main__": 

    ...
