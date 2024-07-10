import pkg_resources
import os, logging

def configure_logger(level:str='debug') -> logging.Logger: 
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
    fh = logging.FileHandler("training.log")
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




