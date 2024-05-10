from __future__ import annotations
import random, matplotlib.pyplot as plt, numpy as np
import pandas as pd, seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal
import gymnasium as gym 
from typing import Union, Tuple


class Policy(nn.Module): 
    def __init__(self, obs_space_dims:int, hidden_dim1:int, hidden_dim2:int, action_space_dims:int): 
        super().__init__()
        self.obs_space_dims=obs_space_dims
        self.action_space_dims=action_space_dims
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2

        self.fc1=nn.Linear(self.obs_space_dims, self.hidden_dim1)
        self.tanh=nn.Tanh()
        self.fc2=nn.Linear(self.hidden_dim1, self.hidden_dim2)

        self.block=nn.Sequential(self.fc1, self.tanh, self.fc2, self.tanh)
        self.mean=nn.Sequential(nn.Linear(self.hidden_dim2, self.action_space_dims))
        self.std=nn.Sequential(nn.Linear(self.hidden_dim2, self.action_space_dims))

    def forward(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
        '''Input: current environment observation --> Output: Sampled action from distro constructed
        by mean and std of fc layer2
        
        Parameters:
        ==========
        obs: Tensor (obs_space_dim)

        Output: 
        ======
        mean: Tensor (action_space_dim)
        std: Tensor (action_space_dim)

        '''
        out=self.block(obs.float())

        mean=self.mean(out)
        std=torch.log(
            1 + torch.exp(self.std(out))
            )
        return (mean, std)


class REINFORCE(nn.Module): 
    def __init__(self, 
                 obs_space_dims:int, 
                 hidden_dim1:int, 
                 hidden_dim2:int, 
                 action_space_dims=int
                 ): 
        super().__init__()
        self.obs_space_dims=obs_space_dims
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.action_space_dims=action_space_dims

        self.hyperparams={
            'lr':1e-4, 
            'gamma':0.99, 
            'eps':1e-6
        }

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net=Policy(self.obs_space_dims, self.hidden_dim1, self.hidden_dim2, self.action_space_dims)
        self.optimizer=torch.optim.AdamW(self.net.parameters(), lr=self.hyperparams['lr'])

    def sample_action(self, state:np.ndarray) -> float: 
        state=Tensor(np.array([state]))
        action_means, action_stds = self.net(state)

        distro=Normal(action_means[0] + self.hyperparams['eps'], action_stds[0] + self.hyperparams['eps'])

        action=distro.sample()
        prob=distro.log_prob(action)
        action=action.numpy()

        self.probs.append(prob)

        return action
    

    def update(self): 
        '''update that happens as the end of an episode'''
        running_gamma=0 # discount factor 
        gammas=[]
        loss=0
        self.optimizer.zero_grad()
        # Discounted return (backwards) - [::-1] will return an array in reverse
        # self.rewards is collected over sample_action thorugh a whole episode
        for R in self.rewards[::-1]: # [start, end, step]; if -1, python assumes reverse
            running_gamma=R + self.hyperparams['gamma']*running_gamma
            gammas.append(running_gamma)

        deltas=Tensor(gammas)
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas): 
            loss+=log_prob.mean()*delta*(-1)

        loss.backward()
        self.optimizer.step()

        self.probs=[]
        self.gammas=[]

def train(env:str, seeds:list[int]): 

    env=gym.make(env)
    wrapped_env=gym.wrappers.RecordEpisodeStatistics(env, 50) # rec episode-reward
    tot_episodes=int(5e3)

    obs_space_dim=env.observation_space.shape[0]
    action_space_dim=env.action_space.shape[0]

    hidden1=512
    hidden2=256

    rewards_per_seed=[]
    seeds=[1, 2, 3, 5, 8]

    for seed in seeds: 
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        rewards_over_ep=[]

        agent=REINFORCE(obs_space_dim, hidden1, hidden2, action_space_dim)

        for episode in range(tot_episodes): 

            obs, info = wrapped_env.reset(seed=seed)
            done=False
            while not done: 
                action=agent.sample_action(obs)
                obs, reward, terminated, truncated, info  = wrapped_env.step(action)
                agent.rewards.append(reward)
                done=terminated or truncated

            agent.update()
            rewards_over_ep.append(wrapped_env.return_queue[-1]) # a deqeueu that stores the reward for the episode
            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_per_seed.append(rewards_over_ep)


    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for InvertedPendulum-v4"
    )
    plt.show()


if __name__=="__main__": 

    train('InvertedPendulum-v4', [1, 2, 3])

    breakpoint()
    net=Policy(768, 512, 256, 10)

    out=net(torch.randn(768))
    breakpoint()

    print('DONE')