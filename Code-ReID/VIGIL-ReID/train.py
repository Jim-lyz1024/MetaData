# train.py
import argparse
import torch
import os
from config import get_cfg_defaults
from utils import setup_logger, set_random_seed
from trainer import build_trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/reid_base.yml")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))

    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        
    data_dir = os.path.normpath(os.path.join(project_root, cfg.DATASETS.ROOT_DIR))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    cfg.DATASETS.ROOT_DIR = data_dir
    
    cfg.freeze()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup logger
    logger = setup_logger("reid", cfg.OUTPUT_DIR)
    logger.info(f"Running with config:\n{cfg}")
    
    # Set random seed
    set_random_seed(args.seed)

    # Build trainer and start training
    trainer = build_trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()