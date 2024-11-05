import os
import sys
import logging
import time

def setup_logger(name, save_dir, distributed_rank=0):
    """Setup logger.
    
    Args:
        name (str): name of logger
        save_dir (str): path to save log file
        distributed_rank (int): process rank in distributed training
        
    Returns:
        logging.Logger: a logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Don't log results for the non-master process
    if distributed_rank > 0:
        return logger
        
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if save_dir:
        fh = logging.FileHandler(
            os.path.join(save_dir, f"log_{time.strftime('%Y-%m-%d-%H-%M-%S')}.txt"),
            mode='w'
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger