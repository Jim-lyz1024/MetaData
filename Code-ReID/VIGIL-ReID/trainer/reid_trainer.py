# reid_trainer.py
import os
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
        """训练Stage 1"""
        self.prompt_learner.train()
        self.image_encoder.eval()
        self.text_encoder.eval()
        
        for epoch in range(self.cfg.SOLVER.STAGE1.MAX_EPOCHS):
            losses = MetricMeter()
            
            for batch_idx, data in enumerate(self.train_loader_stage1):
                imgs = data['img'].float().to(self.device)
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
        """训练Stage 2"""
        print("Starting Stage 2 training...")
        self.prompt_learner.eval()
        self.image_encoder.train()
        
        best_rank1 = 0.0
        best_map = 0.0
        
        with torch.no_grad():
            text_features = self.get_text_features()  # [num_classes, embed_dim]
        
        for epoch in range(self.cfg.SOLVER.STAGE2.MAX_EPOCHS):
            losses = MetricMeter()
            
            # Training
            self.image_encoder.train()
            for batch_idx, data in enumerate(self.train_loader_stage2):
                imgs = data['img'].float().to(self.device)
                pids = data['pid'].to(self.device)
                
                image_features = self.image_encoder(imgs)
                loss = self.compute_stage2_loss(image_features, text_features, pids)
                
                self.optimizer_stage2.zero_grad()
                loss.backward()
                self.optimizer_stage2.step()
                
                losses.update({'loss': loss.item()})
                
                if (batch_idx + 1) % self.cfg.SOLVER.STAGE2.LOG_PERIOD == 0:
                    print(f'Stage2 Epoch: [{epoch+1}][{batch_idx+1}/{len(self.train_loader_stage2)}]\t{losses}')
            
            # Evaluation
            if (epoch + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
                print(f"\nEvaluating epoch {epoch+1}...")
                metrics = self.test()
                
                mAP = metrics['mAP']
                rank1 = metrics['rank1']
                rank5 = metrics['rank5']
                rank10 = metrics['rank10']
                
                print(f'Validation Results - Epoch: {epoch+1}')
                print(f"mAP: {mAP:.1%}")
                print(f"Rank-1: {rank1:.1%}")
                print(f"Rank-5: {rank5:.1%}")
                print(f"Rank-10: {rank10:.1%}")
                
                # Save best model
                is_best = rank1 > best_rank1
                if is_best:
                    best_rank1 = rank1
                    best_map = mAP
                    self.save_model(
                        epoch,
                        self.optimizer_stage2,
                        self.scheduler_stage2,
                        is_best=True,
                        best_rank1=best_rank1,
                        best_map=best_map
                    )
                
                # Regular checkpoint saving
                if (epoch + 1) % self.cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD == 0:
                    self.save_model(
                        epoch,
                        self.optimizer_stage2,
                        self.scheduler_stage2,
                        is_best=False
                    )
            
            self.scheduler_stage2.step()
        
        print("\nTraining completed!")
        print(f"Best Rank-1: {best_rank1:.1%}")
        print(f"Best mAP: {best_map:.1%}")
        
    def save_model(self, epoch, optimizer, scheduler, is_best=False, best_rank1=None, best_map=None):
        state = {
            'state_dict': {
                'image_encoder': self.image_encoder.state_dict(),
                'prompt_learner': self.prompt_learner.state_dict(),
                'text_encoder': self.text_encoder.state_dict()
            },
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'best_rank1': best_rank1,
            'best_map': best_map
        }
        
        save_path = os.path.join(self.cfg.OUTPUT_DIR, f'checkpoint_ep{epoch+1}.pth')
        torch.save(state, save_path)
        print(f'Model saved to {save_path}')
        
        if is_best:
            best_path = os.path.join(self.cfg.OUTPUT_DIR, 'model_best.pth')
            torch.save(state, best_path)
            print(f'Best model saved to {best_path}')    
        
    @torch.no_grad()
    def test(self):
        """Test process."""
        print('\nRunning evaluation...')
        self.image_encoder.eval()
        self.evaluator.reset()
        
        # Get text features for all classes
        with torch.no_grad():
            text_features = self.get_text_features()  # [num_classes, embed_dim]
        
        print('\nExtracting features from query set ...')
        for batch_idx, data in enumerate(self.query_loader):
            imgs = data['img'].float().to(self.device)
            pids = data['pid']
            camids = data['camid']
            
            with torch.no_grad():
                image_features = self.image_encoder(imgs)
                if self.cfg.TEST.FEAT_NORM:
                    image_features = F.normalize(image_features, p=2, dim=1)
            
            self.evaluator.process(image_features, data, is_query=True)
                
        print('\nExtracting features from gallery set ...')
        for batch_idx, data in enumerate(self.gallery_loader):
            imgs = data['img'].float().to(self.device)
            pids = data['pid']
            camids = data['camid']
            
            with torch.no_grad():
                image_features = self.image_encoder(imgs)
                if self.cfg.TEST.FEAT_NORM:
                    image_features = F.normalize(image_features, p=2, dim=1)
            
            self.evaluator.process(image_features, data, is_gallery=True)
                
        print('Computing metrics ...')
        metrics = self.evaluator.evaluate()
        return metrics      

    def get_text_features(self, pids=None, image_features=None):
        if pids is not None and image_features is not None:
            batch_size = image_features.shape[0]
            prompts = self.prompt_learner.forward_stage1(pids, image_features)
            tokenized_prompts = self.prompt_learner.tokenized_prompts.expand(batch_size, -1)
            text_features = self.text_encoder(prompts, tokenized_prompts)
        else:
            prompts = self.prompt_learner.forward_stage2()
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        return text_features
    
    def compute_stage1_loss(self, image_features, text_features, pids):
        """Compute Stage 1 loss."""
        i2t_loss = self.criterion_i2t(text_features, image_features, pids, pids)
        t2i_loss = self.criterion_i2t(image_features, text_features, pids, pids)
        return i2t_loss + t2i_loss

    def compute_stage2_loss(self, image_features, text_features, pids):
        image_features = F.normalize(image_features, p=2, dim=1)  # [batch_size, embed_dim]
        text_features = F.normalize(text_features, p=2, dim=1)  # [num_classes, embed_dim]
        
        sim_matrix = image_features @ text_features.t()  # [batch_size, num_classes]
        
        id_loss = self.criterion_xent(sim_matrix, pids)
        triplet_loss = self.criterion_triplet(image_features, pids)[0]
        
        loss = (
            self.cfg.MODEL.ID_LOSS_WEIGHT * id_loss +
            self.cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet_loss
        )
        
        return loss
