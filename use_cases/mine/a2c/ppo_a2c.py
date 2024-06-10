from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
import gymnasium as gym 
from gymnasium.wrappers.frame_stack import FrameStack
import sys, os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pydantic import BaseModel, Field
from transformers.utils import ModelOutput
from typing import Dict, Tuple, List
import sys, os
sys.path.append(os.getcwd())
from models import Actor, Critic

def calc_returns(rewards:List[int], gamma:float) -> List[Tensor]:
    cumulative_rewards=[]
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        cumulative_rewards.insert(0, torch.tensor(R))
    return cumulative_rewards

def generate_trajectories(actor:Actor,
                          critic:Critic, 
                          env:gym.Env
                          ): 
    
    NUM_TRAJECTORIES=1000
    trajectories=[]

    while len(trajectories)<=NUM_TRAJECTORIES:

        actions=[]
        rewards=[]
        observations=[]
        values=[]

        trajectory={
            'actions':actions,
            'rewards':rewards,
            'observations':observations,
            'values':values
            }

        done=False
        obs, _ = env.reset()
        while not done: 
            action, _action_logits, process_obs = actor.choose_action(obs)
            value=critic.calculate_value(obs)
            next_obs, reward, done, trunc, _ = env.step(action)
            
            trajectory['rewards'].append(reward)
            trajectory['actions'].append(action)
            trajectory['observations'].append(process_obs)
            trajectory['values'].append(value)

            obs=next_obs

        trajectories.append(trajectory)

    return trajectories


def train(actor:Actor,
          critic:Critic, 
          env:gym.Env): 
    
    trajectories=generate_episode()
    rewards=buffer['rewards']
    # actions, rewards = torch.tensor(buffer['actions']), torch.tensor(buffer['rewards'])
    # observations, values = torch.tensor(buffer['observations']), torch.tensor(buffer['values'])

    cumulative_returns=calc_returns(rewards, gamma=0.99)
    breakpoint()

    ...

if __name__=="__main__": 

    env=gym.make("ALE/MarioBros-v5")
    env=FrameStack(env, 4)
    obs, info = env.reset()

    A = env.action_space.n
    actor=Actor(A, 4, 256, 3, 2, 128, 0.1)
    critic=Critic(4, 256, 3, 2, 128, 0.1)

    train(actor, critic, env)


    breakpoint()

    print('DONE')

