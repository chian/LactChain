import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym 
import copy 
from collections import deque
from typing import Tuple, Any, List
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

class QNetwork(nn.Module): 
    def __init__(self, 
                 num_blocks:int, 
                 input_size:int, 
                 hidden_size:int, 
                 output_size:int, 
                 dropout:float
                 ): 
        super().__init__()
        layer1=nn.Linear(input_size, hidden_size)
        layer2=nn.Linear(hidden_size, output_size)
        act=nn.GELU()
        drop=nn.Dropout(dropout)
        block=nn.Sequential(layer1, act, drop, layer2, act)

        self.encoder=nn.ModuleList([block for _ in range(num_blocks)])
    
    def forward(self, x:Tensor) -> Tensor: 
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



    # net=QNetwork(3, 512, 50, 10, 0.1)

    # state=torch.randn(10, 512)

    # out=net(state) # 10, 5

    B = 5
    n = 10

    out=torch.randn((B, n))
    binary_tensor=torch.tensor([1, 0, 1, 0])
    mask=torch.tensor()
    breakpoint()
    indices = torch.arange(n).expand(B, n)
    random_indices = torch.randint(0, n, (B, n))
    final_indices = torch.where(binary_tensor.view(-1, 1).expand(B, n).bool(), indices, random_indices)

    breakpoint()

    print('DONE')



