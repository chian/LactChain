import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms import transforms
from typing import Tuple, Any, List
from collections import deque

class VisionTransform(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.transforms=v2.Compose([
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image:np.ndarray) -> Tensor: 
        image=torch.Tensor(image).type(torch.float32).permute(-1, 0, 1)
        out=self.transforms(image)
        return out
    
class FrameTransform(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.preprocess = transforms.Compose([
                                    transforms.ToPILImage(),         # Convert numpy array to PIL Image
                                    transforms.Grayscale(),          # Convert to grayscale
                                    transforms.Resize((84, 84)),     # Resize to 84x84
                                    transforms.ToTensor(),           # Convert to tensor and reorder dimensions
                                    transforms.Normalize(0.0, 1.0)   # Normalize pixel values to [0, 1]
                                ])

    def preprocess_frame_stack(self, frame_stack:np.ndarray) -> Tensor:
        # Assuming frame_stack is of shape [4, 250, 160, 3]
        # Convert to grayscale by averaging the color channels
        gray_stack = np.mean(frame_stack, axis=-1).astype(np.uint8)  # Shape: [4, 250, 160]
        processed_frames = torch.cat([self.preprocess(frame) for frame in gray_stack]) # Apply the transform to the entire stack
        return processed_frames.unsqueeze(0)  # Add batch dimension
    
    def __call__(self, frames:np.ndarray) -> Tensor: 
        out = self.preprocess_frame_stack(frames)
        return out
    
class TensorTranform(nn.Module): 
    def __init__(self): 
        super().__init__()
    def __call__(self, obs:np.ndarray) -> Tensor: 
        obs = torch.from_numpy(obs).type(torch.float32)
        return obs

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