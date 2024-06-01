import sys, os
from typing import List, Callable
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym

sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction
from use_cases.mine.lactchain.lactchains import LactChainA
from use_cases.mine.lactchain.dataset import Memory

def calc_returns(rewards:List[int], cum_rewards:List[int], gamma:float) -> List[Tensor]:
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        cum_rewards.insert(0, torch.tensor(R))
    return cum_rewards

def sample_experience(env:gym.Env, num_episodes:int, lactchains:List[Callable]): 
    states=[]
    actions=[]
    next_states=[]
    dones=[]
    target_q_values=[]
    max_steps=100

    for episode in range(num_episodes): 
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps<=max_steps:
            action = env.lact_chains[0].propose_action() 
            next_obs, tot_reward, rewards, done  = env.step(action)  # Execute the action
            target_q_value = compute_target_q_value(next_obs, rewards, done)

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            dones.append(done)
            target_q_values.append(target_q_value)

            obs=next_obs
            steps += 1

    return states, actions, next_states, dones, 
    ...

def compute_advantage(): 
    ...

def compute_target_q_value(): 
    ...


def collect_experience(num_episodes:int, 
                       env:gym.Env, 
                       lactchain:LactChainA, 
                       critic:ValueFunction, 
                       memory:Memory, 
                       max_steps:int
                       ): 

    for episode in range(num_episodes): 
        done=False
        step=0
        obs = env.reset()

        while not done and step<=max_steps: 

            action = lactchain.propose_action(obs)
            next_obs, reward, done = env.step(action)
            memory.add_transition(obs, action, reward, next_obs)

            obs=next_obs
            step+=1

        compute_advantage(...)
    ...

def compute_advantage():
    ...


if __name__=="__main__": 

    env=GridEnvironment()
    obs=env.reset()

    cum_rewards=[]
    policy_losses=[]
    critic_losses=[]

    action_seq=['turn left', 'turn left', 'move forward', 'move forward', 'move forward', 'move forward']
    obs, tot_reward, rewards, done = env.step(action_seq)

    returns=calc_returns(rewards, cum_rewards, gamma=0.99)

    for (log_prob, value), R in zip(action_values, cum_rewards): 
        advantage = R - value
        critic_losses.append(F.smooth_l1_loss(R, value))
        policy_losses.append((-1)*log_prob*advantage)

        critic_loss=torch.stack(critic_losses).mean()
        policy_loss=torch.stack(policy_losses).mean()

        

    breakpoint()
    