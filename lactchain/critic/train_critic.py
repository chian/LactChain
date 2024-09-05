from __future__ import annotations

'''File that trains the critic embedder function'''

from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F

from lactchain.critic.tokenizer import CriticTokenizer, CriticTokenizerConfig
from lactchain.critic.embedder import EmbedderFunction, EmbedderFunctionConfig
from lactchain.critic.critic_dataset import CriticDataset

def calculate_returns(rewards: Tensor, gamma: float = 0.99, normalize: Optional[bool] = True) -> Tensor:
    '''Takes a Tensor of rewards in trajectory and computes the return R_t for t in trajectory'''

    returns = torch.zeros(rewards.shape).to(rewards.device)
    R = torch.tensor(0.0)
    for idx, r in enumerate(torch.flip(rewards, dims=[0])):
        R = r + gamma*R
        returns[idx] = R

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-12)

    return returns


def main(): 

    tokenizer_config = CriticTokenizerConfig(
        pretrained_model_name_or_path='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de')
    tokenizer = CriticTokenizer(tokenizer_config)

    embedder_config = EmbedderFunctionConfig()
    embedder = EmbedderFunction(embedder_config)

    dataset = CriticDataset('./critic_dataset')
    dataloader = DataLoader(dataset, batch_size=4)
    
    
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch in dataloader:

            rewards, inputs = tokenizer(batch)

            q_values = embedder(**inputs)
            returns = calculate_returns(rewards, 0.99, True)
            
            loss = F.smooth_l1_loss(q_values, returns)
            
        
    ...


if __name__ == "__main__":

    main()