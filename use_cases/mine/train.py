import sys, os
from typing import List
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain/')
from classes.lactchain import LactChain, Context, Component
from use_cases.mine.lactchain.environment import GridEnvironment



def calc_returns(rewards:List[int], cum_rewards:List[int], gamma:float) -> List[Tensor]:
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        cum_rewards.insert(0, torch.tensor(R))
    return cum_rewards

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
    