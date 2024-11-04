import os 
import errno
import json
import random 
import numpy as np
import torch
import torch.nn as nn
from PIL import Image 

def set_random_seed(seed):
    """Set random seed.
    
    Args:
        seed (int): seed to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def collect_env_info():
    """Collect environment information."""
    env_str = torch.utils.collect_env.get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(Image.__version__)
    return env_str
    
def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def check_path_exists(path):
    """Check if path exists."""
    if not os.path.exists(path):
        print(f"Warning: path '{path}' does not exist!")
        return False
    return True
                
def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj
    
def write_json(obj, fpath):
    """Write json object to a path."""
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
        
def resume_from_checkpoint(ckpt_path, model, optimizer=None):
    """Resume from saved checkpoint.
    
    Args:
        ckpt_path (str): path to checkpoint
        model (nn.Module): model
        optimizer (Optimizer): optimizer
        
    Returns:
        int: start_epoch
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "=> no checkpoint found at '{}'".format(ckpt_path)
        )
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch']
    
    # Load model state dict
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint['state_dict'].items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Load optimizer state dict
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print("=> loaded checkpoint '{}' (epoch {})".format(
        ckpt_path, checkpoint['epoch']
    ))
    
    return start_epoch