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
from use_cases.mine.custom_environment import SimpleEnvironment, SimpleGridRewardFunction

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

STRATEGY_PROMPT=f'''
You are a smart navigator that lives in grid-world. Your goal is to help me explore
navigate grid-world and get to the green square in the least amount of steps: 

At each round of conversation, I can give you:

State: 
// This is the current state of the environment you are in. //

Previous Move: 
// This is the previous action that you have committed. //

Reward: 
// This is the reward you received from your previous move and the previous move you did. //

You must provide me with the following: 
Action:
// This is a single action that you can choose from in json format. You must return it as {{action}} and nothing more. //

Below are actions you can choose from: 
======================================
0           left            Turn Left 
1           right           Turn Right
2           forward         Move Forward
3           pickup          Unused
4           drop            Unused
5           toggle          Unused
6           done            Unused
'''

ACTOR_INPUT='''
State: 
{state}

Previous Move: 
{action}

Reward: 
{reward}
'''

CRITIC_PROMPT='''
'''

ENVIRONMENTS={
    'minigrid-empty':"MiniGrid-Empty-5x5-v0",
}



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

    env=SimpleEnvironment()

    

    observation = env.reset()

    env=gym.make("MiniGrid-Empty-5x5-v0")
    obs, info = env.reset()

    action=env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    actorconfig=ActorConfig()
    actor=Actor(actorconfig)

    output_text=actor(STRATEGY_PROMPT + ACTOR_INPUT.format({'state':None, 
                                                            'action':None, 
                                                            'reward':None}))

    print('DONE')

