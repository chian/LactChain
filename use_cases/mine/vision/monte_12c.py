import torch, torch.nn as nn, torch.nn.functional as F 
from encoder import VisionEncoder
import gymnasium as gym
from torch import Tensor 
from typing import Union, Dict, Tuple, Any, List
from torch.distributions import Categorical
from collections import namedtuple
import sys, os
sys.path.append('/nfs/lambda_stor_01/homes/bhsu/2024_research/LactChain')
from use_cases.mine.vision.data import VisionTransform, TensorTranform, FrameTransform
from use_cases.mine.vision.config import ACConfig
import numpy as np
from gymnasium.experimental.wrappers import RecordVideoV0
from gym.wrappers import FrameStack
from textwrap import dedent
import torchvision.transforms as transforms
from PIL import Image
import wandb

class ActorCritic(nn.Module): 
    def __init__(self,
                 config:ACConfig, 
                 transform:str
                 ): 
        super().__init__()
        models={
            'vision':VisionEncoder, 
            'other':nn.Linear
        }
        transforms={
            'image':VisionTransform(),
            'other':TensorTranform(), 
            'frame':FrameTransform()
        }
        self.transform=transforms[transform]
        model=models.get(config.game)
        self.encoder=model(**config.model.model_dump())

        _hidden_dim=int(config.model.out_features)
        _action_dim=int(config.actor.action_dim)
        _critic_dim=int(config.critic.critic_dim)

        self.action_head=nn.Linear(_hidden_dim, _action_dim)
        self.critic_head=nn.Linear(_hidden_dim, _critic_dim)
        self.softmax=nn.Softmax(dim=-1)

        self.alpha=nn.Parameter(Tensor(1), requires_grad=True)
        self.gamma=float(config.critic.gamma)

    def forward(self, state:Union[np.ndarray, Tensor]) -> Tuple[Dict[str, Any], Tensor]: 
        if type(state)!=Tensor: 
            state=self.transform(state)
        state=state.to(DEVICE) if DEVICE else 'cpu'
        '''Returns action, logits, and log prob along with critic value'''
        latent=self.encoder(state)
        _logits=self.action_head(latent)
        probs=self.softmax(_logits)
        self.m=Categorical(probs)
        action=self.m.sample()
        logp_action=self.m.log_prob(action)

        value=self.critic_head(latent)[0][0]
        return ({'action':action.item(), 
                 'log_prob':logp_action}, 
                 value)
    
def calc_returns(rewards:List[int], cum_rewards:List[int], gamma:float) -> List[Tensor]:
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        cum_rewards.insert(0, torch.tensor(R))
    return cum_rewards
    
if __name__=="__main__": 

    DEVICE='cuda:0'
    PROJECT='Atari-Test'
    NUM_EPISODES=1000
    VIDEO_PATH="./save_videos3"
    MODEL_PATH="./atari"
    VIDEO_LENGTH=1e4

    wandb.init(project=PROJECT)

    env=gym.make('ALE/AirRaid-v5', full_action_space=False, render_mode='rgb_array') # full_action_space=False
    env = RecordVideoV0(env, video_folder=VIDEO_PATH, video_length=VIDEO_LENGTH, disable_logger=True)
    env = FrameStack(env, num_stack=4)
    config=ACConfig()
    actorcritic=ActorCritic(config, 'frame').to(DEVICE)
    optimizer=torch.optim.AdamW(actorcritic.parameters(), lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)
    memory=namedtuple('ActionValue', ['log_prob', 'value'])

    for episode in range(NUM_EPISODES): 

        obs, info = env.reset()

        episode_reward=0
        action_values=[]
        rewards=[]
        terminated=False
        steps, num_steps=0, 1e6
        optimizer.zero_grad()

        while not terminated and steps<=1e6: 
            out, value = actorcritic(obs)
            obs, reward, terminated, truncated, info = env.step(out['action'])
            action_values.append(memory(out['log_prob'], value))
            rewards.append(reward)
            episode_reward+=reward

        cum_rewards=[]
        policy_losses=[]
        critic_losses=[]

        cum_rewards=calc_returns(rewards, cum_rewards, actorcritic.gamma)

        cum_rewards = torch.tensor(cum_rewards).to(DEVICE) if DEVICE else 'cpu'
        cum_rewards = (cum_rewards - cum_rewards.mean()) / (cum_rewards.std() + 1e-12)

        for (log_prob, value), R in zip(action_values, cum_rewards): 
            advantage = R - value
            critic_losses.append(F.smooth_l1_loss(R, value))
            policy_losses.append((-1)*log_prob*advantage)

        critic_loss=torch.stack(critic_losses).mean()
        policy_loss=torch.stack(policy_losses).mean()

        loss=critic_loss+policy_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if episode % 10 == 0: 
            print(dedent(f'''
                         For Episode: {episode}, 
                         Total Loss: {loss},
                         Policy Loss: {policy_loss},
                         Critic Loss: {critic_loss}, 
                         Reward: {episode_reward}\n
                        '''))

        del critic_losses
        del policy_losses
        del cum_rewards
        del action_values
        del rewards

    # at the end of everything 
    # wandb.log({"video": wandb.Video(VIDEO_PATH, fps=30, format="gif")})
    for filename in os.listdir(VIDEO_PATH):
        if filename.endswith(".mp4"): 
            video_path = os.path.join(VIDEO_PATH, filename)
            wandb.log({f"video_{filename}": wandb.Video(video_path, fps=4, format="mp4")})
    torch.save(actorcritic.state_dict(), MODEL_PATH)





    







