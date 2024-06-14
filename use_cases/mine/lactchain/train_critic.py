import torch, torch.nn as nn, torch.nn.functional as F
import sys, os
from typing import Any, Union, Dict, Tuple, List, Optional, Literal
from torch import Tensor
from lightning_fabric import Fabric
from pathlib import Path
from argparse import ArgumentParser
sys.path.append(os.getcwd()+'/../../../')
from use_cases.mine.lactchain.environment import GridEnvironment
from use_cases.mine.lactchain.critic import ValueFunction, ValueFunctionConfig, LoraConfigSettings
from use_cases.mine.lactchain.lactchains import MyLactChain, PolicyConfig
from use_cases.mine.lactchain.lactchains import MyLactChain

os.environ['HF_HOME']=os.getcwd()+'/../../../../'

PathLike=Union[str, Path]
        
def calc_returns(rewards:List[int], gamma:float) -> List[Tensor]:
    '''Takes a list of returns in trajectory and computes the return R_t for t in trajectory
    Input: Sequence[int] -> Output: Sequence[torch(int)]
    '''
    returns=[]
    R = 0
    for r in rewards[::-1]: 
        R = r + gamma*R
        returns.insert(0, torch.tensor(R))
    return returns

def save_model(model:Any, save_path:PathLike): 
    model.save_pretrained(save_path, from_pt=True) 

def train_critic(actor:MyLactChain, 
                 critic:ValueFunction, 
                 critic_optimizer: torch.optim.AdamW, 
                 critic_scheduler:torch.optim.lr_scheduler.CosineAnnealingLR, 
                 env:GridEnvironment, 
                 args:ArgumentParser,
                 fabric:Optional[Fabric]=None, 
                 save_path:Optional[PathLike]=None
                 ): 
    TOTAL_PARAMS=sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f'STARTING TRAINING..., with model size {TOTAL_PARAMS}')
    if fabric: 
        actor = fabric.setup(actor)
        critic, critic_optimizer = fabric.setup(critic, critic_optimizer)
    
    for episode in range(args.num_episodes): 
        
        rewards=[]
        values=[]
        state, info = env.reset()
        done=0
        steps=0
        
        while not done and steps<=args.max_steps:         
     
            action, context=actor.sample_action(state, info)
            value=critic(state, info)
            next_state, reward, done, info = env.step(action)
            
            print(f'STATE:{state}')
            print(f'ACTION: {action}')
        
            rewards.append(reward)
            values.append(value)
            
            state=next_state
            steps+=1
            
        cumulative_returns=calc_returns(rewards, args.gamma)
        cumulative_returns = torch.tensor(cumulative_returns)
        cumulative_returns = (cumulative_returns - cumulative_returns.mean()) /\
                                (cumulative_returns.std() + 1e-12)
        
        critic_losses=[]
        for R, value in zip(cumulative_returns, values): 
            critic_losses.append(F.smooth_l1_loss(R, value)) # advantage = R - value
            
        critic_loss=torch.stack(critic_losses).mean()
        
        critic_optimizer.zero_grad()
        critic_loss.backward() 
        critic_optimizer.step()
        
        if episode % 10 == 0: 
            critic_scheduler.step()
        
        print(f'Episode {episode+1} Loss: {critic_loss}')
        print(f'Episode {episode+1} Total Reward: {cumulative_returns.mean()}')
        
    if args.model_save_path: 
        save_model(critic, args.model_save_path)
    

def argparse() -> Any: 
    argparse=ArgumentParser()
    argparse.add_argument('--model_save_path', type=str, default='./')
    argparse.add_argument('--actor_path', type=str, 
                          default='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    argparse.add_argument('--critic_path', type=str, 
                          default='./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11')
    argparse.add_argument('--num_episodes', type=int, default=100)
    argparse.add_argument('--max_steps', type=int, default=15)
    argparse.add_argument('--gamma', type=float, default=0.99)
    args = argparse.parse_args()
    return args


def main():
    
    args=argparse()
    
    lora_config=LoraConfigSettings()
    actor_config=PolicyConfig() 
    critic_config=ValueFunctionConfig()
    
    env=GridEnvironment()
    
    actor=MyLactChain(args.actor_path, actor_config, lora_config)
    critic=ValueFunction(args.critic_path, critic_config)
    
    critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    critic_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=10)
    
    train_critic(actor, critic, critic_optimizer, critic_scheduler, env, args)
    

    
if __name__=="__main__": 
    
    main()
    
    # LACTCHAIN_PATH='./models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
    # CRITIC_PATH="./models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
    
    # lora_config=LoraConfigSettings()
    # lactchain_config=PolicyConfig()
    # lactchain=MyLactChain(LACTCHAIN_PATH, lactchain_config, lora_config, './')

    # critic_config=ValueFunctionConfig()
    # critic=ValueFunction(CRITIC_PATH, critic_config)

    # critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4)
    # critic_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=10)
    
    # env=GridEnvironment()
    
    # train_critic(lactchain, critic, critic_optimizer, critic_scheduler, env)

