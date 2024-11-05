import torch
import torch.nn as nn
from torchvision import models
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        # Build backbone
        backbone = models.resnet50(pretrained=True)
        
        # Remove the final fc layer
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2]
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Bottleneck
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        
        # Classifier
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        # Extract features
        feat = self.backbone(x)
        global_feat = self.gap(feat)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        
        # BN
        bn_feat = self.bottleneck(global_feat)
        
        # Classification
        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, global_feat
        else:
            return global_feat