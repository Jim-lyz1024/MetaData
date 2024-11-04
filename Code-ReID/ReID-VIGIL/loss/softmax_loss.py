import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        device = inputs.device
        batch_size = inputs.size(0)

        log_probs = self.logsoftmax(inputs)  # [batch_size, num_classes]
        
        targets = torch.clamp(targets, 0, self.num_classes - 1)
        
        targets_onehot = torch.zeros(batch_size, self.num_classes, device=device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        targets_onehot = (1 - self.epsilon) * targets_onehot + \
                        self.epsilon / self.num_classes
                        
        loss = (- targets_onehot * log_probs).mean(0).sum()
        
        return loss