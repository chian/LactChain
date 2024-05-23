import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym 
import sys, os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pydantic import BaseModel, Field
from transformers.utils import ModelOutput
from typing import Dict, Tuple

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')
from use_cases.mine.a2c.poolers import get_pooler, average_pool, last_token_pool
from use_cases.mine.a2c.configs import CriticConfig, ActorConfig

'''
=============
Possible plan: 
=============

Actor: wants to take the best action based on a strategy/policy (prompt-breeder goes here)
=====
- Takes in value assessment from critic (for now as a prompt)
- Takes in reward from environment (for now as a string)
- Outputs an Action as an str

Critic: wants to estimate how good the action taken is via advantage function or Q/V func aka value approx
======
- Consider learned token that encodes action from actor as a prefix token from prefix tuning
- Have it output an embedding or prompt ? That evaluates the reward of the action 

Environment: grid world, but outputs rewards in the form of strings 
===========
- Input is action 
- Output is state of the agent encoded as a string to feed into critic and actor
'''

STRATEGY_PROMPT='''
You are an AGI that lives in grid-world.

'''

CRITIC_PROMPT='''

'''


class Actor(nn.Module): 
    def __init__(self, config:ActorConfig): 
        super().__init__()
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(
            config.model_args['pretrained_model_name_or_path']
            )
        self.model=AutoModelForCausalLM.from_pretrained(
            **config.model_args
            ).to('cuda:1')

        poolers={'last_token':last_token_pool, 
                 'mean':average_pool}

        self.pooler=poolers[config.pooling_type]

    def select_action(self): 
        
        return None

    def forward(self, state:str) -> Tuple[str, ModelOutput]: 

        inputs=self.tokenizer(state, **self.config.tokenizer_args).to('cuda:1')
        outputs=self.model(**inputs|self.config.model_forward_args) 
        
        # embedding=outputs.last_hidden_state[-1].mean(dim=0)
        output_text=self.model.generate(**outputs)

        return output_text

if __name__=="__main__": 

    # env=gym.make('')

    actorconfig=ActorConfig()
    actor=Actor(actorconfig)

    decoded, outputs=actor('hello how are you doing?')

    print('DONE')

