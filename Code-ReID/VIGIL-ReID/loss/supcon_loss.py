# loss/supcon_loss.py

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.temperature = 1.0

    def forward(self, text_features, image_features, t_label, i_targets):
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        
        # Create label matrix
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N),
                       i_targets.unsqueeze(0).expand(batch_size, batch_size_N))
        mask = mask.float().to(self.device)
        
        # Calculate similarities
        logits = torch.div(torch.matmul(text_features, image_features.T),
                          self.temperature)
        
        # Calculate loss
        loss = torch.exp(logits)
        return loss