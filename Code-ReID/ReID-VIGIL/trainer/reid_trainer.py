# reid_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer import TrainerBase
from .build_trainer import MODEL_REGISTRY
from loss import TripletLoss, CrossEntropyLabelSmooth, SupConLoss
import models
import clip
from collections import OrderedDict
from models.prompt import TextEncoder, PromptLearner
from utils import MetricMeter

@MODEL_REGISTRY.register()
class ReIDTrainer(TrainerBase):
    """Trainer for ReID tasks with two-stage CLIP training."""
    
    def build_model(self):
        """Build CLIP model and text generator."""
        print("Building CLIP model and text generator...")
        
        # Get input resolution from config
        self.input_resolution = self.cfg.INPUT.SIZE_TRAIN[0]
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(
            self.cfg.MODEL.BACKBONE, 
            device=self.device, 
            jit=False
        )
        
        # Convert model to float32 and ensure on correct device
        self.clip_model = self.clip_model.float().to(self.device)
        
        # Get encoders
        self.image_encoder = self.clip_model.visual.float().to(self.device)
        self.text_encoder = TextEncoder(self.clip_model).float().to(self.device)
        
        # Build loss functions first
        self.build_losses()

        # Text description generator
        self.prompt_learner = PromptLearner(
            self.num_classes, 
            self.cfg.DATASETS.NAMES,
            self.clip_model
        ).float().to(self.device)

        # Build optimizers
        self.build_optimizers()

    def build_optimizers(self):
        """Build optimizers for both stages."""
        # Stage 1 optimizer
        self.optimizer_stage1 = torch.optim.Adam(
            self.prompt_learner.parameters(),
            lr=self.cfg.SOLVER.STAGE1.BASE_LR
        )
        
        # Stage 2 optimizer
        params = []
        for key, value in self.image_encoder.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.cfg.SOLVER.STAGE2.BASE_LR}]
        self.optimizer_stage2 = torch.optim.Adam(params)

        # Schedulers
        self.scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_stage1, 
            T_max=self.cfg.SOLVER.STAGE1.MAX_EPOCHS
        )
        
        self.scheduler_stage2 = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_stage2,
            milestones=self.cfg.SOLVER.STAGE2.STEPS,
            gamma=self.cfg.SOLVER.GAMMA
        )

    def build_losses(self):
        """Build loss functions."""
        print(f"Building losses... num_classes = {self.num_classes}")
        self.criterion_xent = CrossEntropyLabelSmooth(self.num_classes).to(self.device)
        self.criterion_triplet = TripletLoss(margin=self.cfg.SOLVER.MARGIN)
        self.criterion_i2t = SupConLoss(self.device)

    def train_stage1(self):
        """Train text description generator."""
        self.prompt_learner.train()
        self.image_encoder.eval()
        self.text_encoder.eval()
        
        for epoch in range(self.cfg.SOLVER.STAGE1.MAX_EPOCHS):
            losses = MetricMeter()
            
            for batch_idx, data in enumerate(self.train_loader_stage1):
                imgs = data['img'].float().to(self.device)  # Ensure float32
                pids = data['pid'].to(self.device)
                
                with torch.no_grad():
                    image_features = self.image_encoder(imgs)
                
                text_features = self.get_text_features(pids, image_features)
                loss = self.compute_stage1_loss(image_features, text_features, pids)
                
                self.optimizer_stage1.zero_grad()
                loss.backward()
                self.optimizer_stage1.step()
                
                losses.update({'loss': loss.item()})
                
                if (batch_idx + 1) % self.cfg.SOLVER.STAGE1.LOG_PERIOD == 0:
                    print(f'Stage1 Epoch: [{epoch+1}][{batch_idx+1}/{len(self.train_loader_stage1)}]\t{losses}')
            
            self.scheduler_stage1.step()

    def train_stage2(self):
        """Train main model."""
        self.prompt_learner.eval()
        self.image_encoder.train()
        
        best_rank1 = 0
        for epoch in range(self.cfg.SOLVER.STAGE2.MAX_EPOCHS):
            losses = MetricMeter()
            
            for batch_idx, data in enumerate(self.train_loader_stage2):
                imgs = data['img'].float().to(self.device)
                pids = data['pid'].to(self.device)
                
                # Ensure pids are within range
                if torch.any(pids >= self.num_classes):
                    print(f"Warning: Found labels {pids} >= num_classes {self.num_classes}")
                    continue
                    
                image_features = self.image_encoder(imgs)
                text_features = self.get_text_features(pids)
                
                loss = self.compute_stage2_loss(image_features, text_features, pids)
                
                self.optimizer_stage2.zero_grad()
                loss.backward()
                self.optimizer_stage2.step()
                
                losses.update({'loss': loss.item()})
                
                if (batch_idx + 1) % self.cfg.SOLVER.STAGE2.LOG_PERIOD == 0:
                    print(f'Stage2 Epoch: [{epoch+1}][{batch_idx+1}/{len(self.train_loader_stage2)}]\t{losses}')
            
    @torch.no_grad()
    def test(self):
        """Test process."""
        self.image_encoder.eval()
        self.evaluator.reset()
        
        print('\nExtracting features from query set ...')
        for batch_idx, data in enumerate(self.query_loader):
            imgs = data['img'].float().to(self.device)  # Ensure float32
            features = self.image_encoder(imgs)
            self.evaluator.process(features, data, is_query=True)
            
        print('\nExtracting features from gallery set ...')
        for batch_idx, data in enumerate(self.gallery_loader):
            imgs = data['img'].float().to(self.device)  # Ensure float32
            features = self.image_encoder(imgs)
            self.evaluator.process(features, data, is_gallery=True)
            
        print('Computing metrics ...')
        metrics = self.evaluator.evaluate()
        return metrics        

    def get_text_features(self, pids, image_features=None):
        """Get text features from prompt learner."""
        if image_features is not None:
            prompts = self.prompt_learner(pids, image_features)
        else:
            prompts = self.prompt_learner(pids)
        return self.text_encoder(prompts, self.prompt_learner.tokenized_prompts.repeat(len(pids), 1))

    def compute_stage1_loss(self, image_features, text_features, pids):
        """Compute Stage 1 loss."""
        i2t_loss = self.criterion_i2t(text_features, image_features, pids, pids)
        t2i_loss = self.criterion_i2t(image_features, text_features, pids, pids)
        return i2t_loss + t2i_loss

    def compute_stage2_loss(self, image_features, text_features, pids):
        """Compute Stage 2 loss."""
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute logits
        logits = image_features @ text_features.t()
        
        # ID loss
        id_loss = self.criterion_xent(logits, pids)
        
        # Triplet loss
        triplet_loss = self.criterion_triplet(image_features, pids)[0]
        
        # Image-to-text loss
        i2t_loss = self.criterion_i2t(image_features, text_features, pids, pids)
        
        # Combine losses
        total_loss = (
            self.cfg.MODEL.ID_LOSS_WEIGHT * id_loss +
            self.cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet_loss + 
            self.cfg.MODEL.I2T_LOSS_WEIGHT * i2t_loss
        )
        
        return total_loss