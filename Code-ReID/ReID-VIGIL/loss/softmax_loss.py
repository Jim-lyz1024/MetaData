import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    
    Args:
        num_classes (int): number of classes
        epsilon (float, optional): smoothing value. Default is 0.1.
    """
    
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        
        # Create one-hot encoding of targets
        # targets_onehot = torch.zeros(inputs.size(), device=inputs.device)
        targets_onehot = torch.zeros(log_probs.size(), device=inputs.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_onehot = (1 - self.epsilon) * targets_onehot + \
                        self.epsilon / self.num_classes
        
        loss = (- targets_onehot * log_probs).sum(dim=1).mean()
        return loss