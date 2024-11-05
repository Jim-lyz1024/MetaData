# trainer.py

import time
import datetime
import torch
import os.path as osp
from collections import OrderedDict
from datasets import build_dataset, build_transforms
from datasets.base_dataset import ReIDDataset  
from datasets.samplers import RandomIdentitySampler
from evaluator import build_evaluator
from utils import AverageMeter, MetricMeter
from PIL import Image
import clip

class ReIDDatasetWrapper(torch.utils.data.Dataset):
    """A wrapper for ReID dataset that applies transforms."""
    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        item = self.data_source[index]
        img_path = item.img_path
        pid = item.pid
        camid = item.camid
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {
            'img': img,
            'pid': pid,
            'camid': camid,
            'img_path': img_path
        }

class TrainerBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Build data loader
        print("Building dataset...")
        dataset = build_dataset(cfg)
        transform_train = build_transforms(cfg, is_train=True)
        transform_test = build_transforms(cfg, is_train=False)

        print("Creating data loaders...")
        train_set = ReIDDatasetWrapper(dataset.train, transform=transform_train)
        query_set = ReIDDatasetWrapper(dataset.query, transform=transform_test)
        gallery_set = ReIDDatasetWrapper(dataset.gallery, transform=transform_test)

        # Stage 1 data loader - for text description generator
        self.train_loader_stage1 = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH if hasattr(cfg.SOLVER, 'STAGE1') else 1,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True
        )

        # Stage 2 data loader
        if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
            print("Using RandomIdentitySampler...")
            train_sampler = RandomIdentitySampler(
                dataset.train,
                cfg.DATALOADER.TRAIN.BATCH_SIZE,
                cfg.DATALOADER.NUM_INSTANCE
            )
        else:
            print("Using RandomSampler...")
            train_sampler = torch.utils.data.sampler.RandomSampler(train_set)

        self.train_loader_stage2 = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        self.query_loader = torch.utils.data.DataLoader(
            query_set,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True
        )

        self.gallery_loader = torch.utils.data.DataLoader(
            gallery_set,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True
        )

        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(train_set)}")
        print(f"Query samples: {len(query_set)}")
        print(f"Gallery samples: {len(gallery_set)}")

        self.num_classes = dataset.num_train_pids
        print(f"Number of classes: {self.num_classes}")

        # Build model
        print("\nBuilding model...")
        self.build_model()

        # Build evaluator
        print("Building evaluator...")
        self.evaluator = build_evaluator(cfg)

    def build_model(self):
        """Build model - to be implemented by subclass."""
        raise NotImplementedError

    def train(self):
        """Main training loop."""
        start_time = time.time()
        print("\nStart training")

        # Stage 1: Text Description Generator Training 
        if hasattr(self.cfg.SOLVER, 'STAGE1'):
            print("\nStage 1: Training Text Description Generator")
            self.train_stage1()

        # Stage 2: Main Model Training
        print("\nStage 2: Training Main Model") 
        self.train_stage2()

        print("\nTraining completed. Total time: {:.2f}s".format(time.time() - start_time))

    def train_stage1(self):
        """Train text description generator - to be implemented by subclass."""
        raise NotImplementedError

    def train_stage2(self):
        """Train main model - to be implemented by subclass."""
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        """Test process."""
        self.model.eval()
        self.evaluator.reset()

        print('\nExtracting features from query set ...')
        for batch_idx, data in enumerate(self.query_loader):
            imgs = data['img'].to(self.device)
            features = self.model(imgs)
            self.evaluator.process(features, data, is_query=True)

        print('\nExtracting features from gallery set ...')
        for batch_idx, data in enumerate(self.gallery_loader):
            imgs = data['img'].to(self.device)
            features = self.model(imgs)
            self.evaluator.process(features, data, is_gallery=True)

        print('Computing metrics ...')
        metrics = self.evaluator.evaluate()
        return metrics

    def save_model(self, epoch, optimizer, scheduler, is_best=False):
        state = {
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None
        }
        save_path = osp.join(self.cfg.OUTPUT_DIR, f'checkpoint_ep{epoch+1}.pth')
        torch.save(state, save_path)
        print(f'Saved model to {save_path}')
        
        if is_best:
            best_path = osp.join(self.cfg.OUTPUT_DIR, 'model_best.pth')
            torch.save(state, best_path)