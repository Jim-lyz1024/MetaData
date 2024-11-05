# test.py
import argparse
import torch
import os
from config import get_cfg_defaults
from utils import setup_logger
from trainer import build_trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = setup_logger("reid", cfg.OUTPUT_DIR, training=False)
    logger.info(f"Running with config:\n{cfg}")

    trainer = build_trainer(cfg)
    trainer.load_model(args.checkpoint)
    trainer.test()

if __name__ == '__main__':
    main()