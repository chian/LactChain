from __future__ import annotations
from typing import Optional
import pkg_resources
import os, logging
from argparse import ArgumentParser
import torch

from lactchain.models.lightning_agent import LightningA2C

def configure_logger(level:str='debug', logging_save_path:Optional[str]=None) -> logging.Logger: 
    '''Function for creating a logger to write to file and terminal'''
    LEVELS={
        'debug':logging.DEBUG, 
        'info':logging.INFO, 
        'warning':logging.WARNING, 
        'error':logging.ERROR, 
        'critical':logging.CRITICAL
    }
        
    logger=logging.getLogger('Critic Training Logger')
    logger.setLevel(LEVELS.get(level))
    ch = logging.StreamHandler()
    ch.setLevel(LEVELS.get(level))
    fh = logging.FileHandler(logging_save_path)
    fh.setLevel(LEVELS.get(level))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def load_lactchain_path(checkpoint_dir:str='/checkpoints/') -> str: 
    """Return Trunk Path to Lactchain package"""
    file_path = pkg_resources.resource_filename("lactchain", "")
    checkpoint_dir_path = file_path + checkpoint_dir
    return str(checkpoint_dir_path)

# LEGACY / BACKUP UTILS
def save_only_trainable_weights(lightning_model:LightningA2C, path:str):
    '''Saves lora + q_value weights only'''
    torch.save(lightning_model.state_dict(), path)
    
def load_only_trainable_weights(lightning_model:LightningA2C, path:str):
    '''Loads lora + q_value weights only'''
    state_dict = torch.load(path)
    lightning_model.load_state_dict(state_dict, strict=False)
        




