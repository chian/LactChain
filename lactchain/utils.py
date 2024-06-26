import pkg_resources
import os 

def load_lactchain_path(checkpoint_dir:str='/checkpoints/') -> str: 
    """Return Trunk Path to Lactchain package"""
    file_path = pkg_resources.resource_filename("lactchain", "")
    checkpoint_dir_path = file_path + checkpoint_dir
    return str(checkpoint_dir_path)




