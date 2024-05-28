import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from torch import Tensor
from torchvision.transforms import v2
import numpy as np
from pydantic import BaseModel, Field
from torch.distributions.categorical import Categorical
from typing import Any, Tuple, Callable, List
from collections import deque
import sys
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')
from use_cases.mine.vision.data import Transform, Memory
from use_cases.mine.vision.encoder import Encoder

class Actor(nn.Module): 
    def __init__(self,
                in_channels:int, 
                out_channels:int, 
                kernel_size:int, 
                stride:int,
                out_features:int,
                dropout:float
                ): 
        super().__init__()

        self.action_encoder=Encoder(in_channels, out_channels, kernel_size, stride, out_features, dropout).to('cuda:0')
        self.alpha=nn.Parameter(torch.ones(1), requires_grad=True).to('cuda:0')

    def forward(self, state:Tensor) -> Tensor: 
        out=self.action_encoder(state)
        _softmax_actions=F.softmax(out, dim=-1)
        action_distro=Categorical(_softmax_actions)
        action=action_distro.sample()
        action_prob=action_distro.log_prob(action)
        return action, action_prob

class Critic(nn.Module): 
    def __init__(self,                 
                in_channels:int, 
                out_channels:int, 
                kernel_size:int, 
                stride:int,
                out_features:int,
                dropout:float, 
                gamma:float
                ): 
        super().__init__()
        self.critic_encoder=Encoder(in_channels, out_channels, kernel_size, stride, out_features, dropout).to('cuda:0')
        self.gamma=gamma

    def forward(self, state:Tensor, next_state:Tensor, reward:Tensor) -> Tensor: 
        value=self.critic_encoder(state)
        next_value=self.critic_encoder(next_state)
        advantage=reward+self.gamma*next_value + value
        return advantage, value
    

class ActorCritic(nn.Module): 
    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                kernel_size:int, 
                stride:int,
                out_features:int, 
                dropout:float, 
                gamma:float
                ): 
        super().__init__()
        self.actor=Actor(in_channels, out_channels, kernel_size, stride, out_features, dropout).to('cuda:0')
        self.critic=Critic(in_channels, out_channels, kernel_size, stride, out_features, dropout, gamma).to('cuda:0')

    def forward(): 

        return None
    
def setup_agents(env:gym.Env, transform:Transform, gamma:float) -> Tuple[object]: 
    obs, info = env.reset()
    obs=transform(obs)
    # action=env.action_space.sample()
    C, O, K, S, A, D = obs.size(0), 10, 3, 3, 6, 0.1
    actor=Actor(C, O, K, S, A, D)
    critic=Critic(C, O, K, S, A, D, gamma)

    return actor, critic

def collect_batch(memory:Memory, env:gym.Env, 
                   actor:Actor, critic:Critic, 
                   transform:Transform, batch_size:int
                   ) -> Tuple[Tensor, ]: 
    '''Each deque[idx] --> (trans_obs, action, reward, next_obs)'''
    obs, info = env.reset()
    terminated=False
    while not terminated and len(memory)<=batch_size: 
        obs=transform(obs)
        action=actor.sample_action(obs)
        next_obs, reward, terminated, _, _ = env.step(action)
        next_obs=transform(next_obs)
        memory.add_transition((obs, action, reward, next_obs, terminated))
        obs = next_obs
    return memory

def online_train(num_episodes:int, num_steps:int=100000): 

    transform=Transform()
    env=gym.make("ALE/AirRaid-v5")
    actor, critic = setup_agents(env, transform, 0.99)
    actor = actor.to('cuda:0')
    critic = critic.to('cuda:0')

    actor_optim=torch.optim.AdamW(actor.parameters(), lr=1e-4)
    critic_optim=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    for episode in range(num_episodes): 

        episode_reward=0
        t = 0
        episode_actor_loss=0
        episode_critic_loss=0
        terminated=False

        obs, info = env.reset()
        obs=transform(obs).to('cuda:0')

        while t<=num_steps and not terminated:
            action, action_prob=actor(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            next_obs=transform(next_obs).to('cuda:0')
            advantage, value=critic(obs, next_obs, reward)
            
            policy_loss=(actor.alpha*value*(-1)*action_prob).mean()
            actor_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_optim.step()

            critic_optim.zero_grad()
            mean_advantage=advantage.pow(2).mean()
            critic_loss=mean_advantage
            critic_loss.backward()
            critic_optim.step()

            obs=next_obs
            t+=1
            episode_reward+=reward
            episode_actor_loss+=policy_loss
            episode_critic_loss+=critic_loss

        print(f'Reward for Episode {episode+1}: {episode_reward}')
        print(f'Actor Loss for Episode {episode+1}: {episode_actor_loss/(t+1)}')
        print(f'Critic Loss for Episode {episode+1}: {episode_critic_loss/(t+1)}')


if __name__=="__main__": 

    online_train(1000)

    # transform=Transform()
    # env=gym.make("ALE/AirRaid-v5")
    # memory=Memory(100000)

    # transform=Transform()
    # env=gym.make("ALE/AirRaid-v5")
    breakpoint()
    # actor, critic = setup_agents(env, transform, 0.99)
    breakpoint()
    # memory = collect_batch(memory, env, actor, critic, transform, batch_size=64)
    breakpoint()
    # obs, info = env.reset()
    # action=env.action_space.sample()
    # obs, reward, terminated, truncated, info = env.step(action) # obs (250, 160, 3) and dict {'lives': 1, 'episode_frame_number': 12, 'frame_number': 20}
    # obs=transform(obs)
    # C, O, K, S, A, D = obs.size(0), 10, 3, 3, 6, 0.1
    # breakpoint()
    # actor=Actor(C, O, K, S, A, D)
    # action = actor.sample_action(obs)
    # breakpoint()


    print('DONE')