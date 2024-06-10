from __future__ import annotations
import torch
from torch import Tensor, nn, functional as F
from typing import Union, Optional, Any, Dict, Tuple
import sys, os
from torchvision.transforms import v2

sys.path.append(os.getcwd())
from dataset import VisionTransform, TensorTranform, FrameTransform


class VisionEncoder(nn.Module): 
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int, 
                 stride:int,
                 out_features:int,
                 dropout:float
                 ) -> None: 
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.out_features=out_features
        self.dropout=dropout

        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.maxpool1=nn.MaxPool2d(kernel_size, stride)
        self.act=nn.GELU()
        self.layer1=None
        self.dropout=nn.Dropout(dropout)

    def forward(self, obs:Tensor) -> Tensor: 
        x = self.conv1(obs)
        x = self.maxpool1(x)
        _in = x.unsqueeze(0).flatten(start_dim=1).size(1)
        if self.layer1 is None:
            self.layer1 = nn.Linear(_in, self.out_features).to(x.device)
        x = x.unsqueeze(0).flatten(start_dim=1)
        x = self.layer1(x)
        x = self.act(x)
        out = self.dropout(x)
        return out

class Actor(nn.Module): 
    def __init__(self, 
                 action_space:int,
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int, 
                 stride:int, 
                 out_features:int,
                 dropout:float
                 ): 
        super().__init__()

        self.transform=FrameTransform()
        self.encoder=VisionEncoder(in_channels, out_channels, kernel_size, 
                                   stride, out_features, dropout)
        self.action_head=nn.Linear(out_features, action_space)

    def forward(self, img:Any) -> Tuple[Tensor, Tensor]: 
        
        img=self.transform(img)
        out=self.encoder(img)
        action_logits=self.action_head(out)

        return action_logits, img
    

    def choose_action(self, img:Any) -> Tuple[int, Tensor, Tensor]: 

        action_logits, img=self.forward(img)
        action=action_logits.argmax(dim=1).item()

        return action, action_logits, img

    def process_observation(self, img:Any) -> Tensor: 
        return self.transform(img)
    

class Critic(nn.Module): 
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int, 
                 stride:int,
                 out_features:int,
                 dropout:float
                 ): 
        super().__init__()
        self.transform=FrameTransform()
        self.encoder=VisionEncoder(in_channels, out_channels, kernel_size, 
                                   stride, out_features, dropout)

        self.value_head=nn.Linear(out_features, 1)

    def forward(self, img:Any) -> Tensor: 
        img=self.transform(img)
        out=self.encoder(img)
        value=self.value_head(out)
        return value
    
    def calculate_value(self, img:Any) -> Tensor: 
        return self.forward(img)



