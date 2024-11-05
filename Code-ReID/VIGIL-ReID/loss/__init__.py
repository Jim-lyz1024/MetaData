from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth
from .supcon_loss import SupConLoss

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SupConLoss'
]