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


