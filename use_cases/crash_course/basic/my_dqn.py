import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym 
import copy 
from collections import deque
from typing import Tuple, Any, List, Dict
from torch import Tensor
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from random import random

class Memory(): 
    def __init__(self, capacity:int): 
        super().__init__()
        self.buffer=deque(maxlen=capacity)

    def add_transition(self, transition:Tuple[Any]): 
        self.buffer.append(transition)

    def add_batch(self, batch_transition:List[Tuple[Any]]): 
        self.buffer.extend(batch_transition)
    
    def __len__(self): 
        return len(self.buffer)
    
    def __getitem__(self, idx:int) -> Any:
        return self.buffer[idx]

class Transforms(object): 
    def __init__(self, transforms:Any=None):
        self.transforms=None 

    def __call__(self, observation:Dict[str, Any]) -> Tensor: 
        out=torch.tensor(observation['image']).type(torch.float32).permute(2, 0, 1)
        return out

class QNetwork(nn.Module): 
    def __init__(self, 
                 num_blocks:int, 
                 in_channels:int, 
                 out_channels:int,
                 kernel_size:int,
                 input_size:int, 
                 hidden_size:int, 
                 output_size:int, 
                 dropout:float, 
                 **kwargs
                 ): 
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool1=nn.MaxPool2d(kernel_size)
        _input_size=int(in_channels*out_channels*kernel_size)

        layer1=nn.Linear(input_size, hidden_size)
        layer2=nn.Linear(hidden_size, output_size)
        act=nn.GELU()
        drop=nn.Dropout(dropout)
        block=nn.Sequential(layer1, act, drop, layer2, act)

        self.encoder=nn.ModuleList([block for _ in range(num_blocks)])

        self.transform=Transforms()
    
    def forward(self, x:Dict[str, Any]) -> Tensor: 
        x = self.transform(x) # just grab the image

        x = self.conv1(x)
        x = self.pool1(x)
        x = x.flatten().unsqueeze(0)
        
        _x = x
        for block in self.encoder: 
            x = block(x)
        x+=_x
        return x

class Agent(object): 
    def __init__(self,
                 num_blocks:int, 
                 input_size:int, 
                 hidden_size:int, 
                 output_size:int, 
                 dropout:float, 
                 epsilon:float
                 ): 
        # policy networks
        self.Q_function=QNetwork(num_blocks, input_size, hidden_size, output_size, dropout)
        self.Q_target=copy.deepcopy(self.Q_function)

        # agent actions
        self.distro=Bernoulli(torch.tensor(epsilon))
        self.rand_actions=Categorical()


    def select_actions(self, state:Tensor) -> Tensor:
        '''Input states: (B, state_shape) 
        -> Q values: (B, action_dim) -> Output Actions: (B)'''

        B=state.size(0) 
        
        decisions=self.actions.get(self.distro.sample(sample_shape=B))
        output=self.Q_function(state)
        actions=torch.zeros(B)
        for idx in range(B): # 1 is greedy, 0 is random
            if decisions[idx]==1:
                actions[idx]=output[idx].argmax(dim=1).item()
            else: 
                actions[idx]=torch.randnint(0, output.size(1))
        return actions

    def gather_batch(self, states:Tensor, actions:Tensor, rewards:Tensor, next_states:Tensor) -> Any: 
        transitions=zip(states, actions, rewards, next_states)
        return transitions

    def forward(self, transition:List[Tuple[Any]]) -> Tensor: 
        return None
    

def train(env_name:str): 

    env=gym.make(env_name)


    return None


if __name__=="__main__": 

    env=gym.make("MiniGrid-Empty-5x5-v0")

    observation, info = env.reset(seed=42)

    transforms=Transforms()
    tensor=transforms(observation)

    conv1=nn.Conv2d(3, 10, 2)
    pool1=nn.MaxPool2d(3)

    breakpoint()

    net=QNetwork(1, 512, 50, 10, 0.1)

    breakpoint()

    out=net(observation)


    breakpoint()


    # net=QNetwork(3, 512, 50, 10, 0.1)

    # state=torch.randn(10, 512)

    # out=net(state) # 10, 5

    B = 5
    n = 10

    # out=torch.randn((B, n))
    # binary_tensor=torch.tensor([1, 0, 1, 0])
    # mask=torch.tensor()
    # breakpoint()
    # indices = torch.arange(n).expand(B, n)
    # random_indices = torch.randint(0, n, (B, n))
    # final_indices = torch.where(binary_tensor.view(-1, 1).expand(B, n).bool(), indices, random_indices)

    breakpoint()

    print('DONE')



