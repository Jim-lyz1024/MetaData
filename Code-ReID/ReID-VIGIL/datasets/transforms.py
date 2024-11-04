from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomCrop,
    ColorJitter,
    CenterCrop
)

def build_transforms(cfg, is_train=True):
    """Build transforms for training/testing."""
    normalize = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, 
        std=cfg.INPUT.PIXEL_STD
    )
    
    if is_train:
        transform = Compose([
            Resize(cfg.INPUT.SIZE_TRAIN),
            RandomHorizontalFlip(p=cfg.INPUT.PROB),
            CenterCrop(cfg.INPUT.SIZE_TRAIN),
            ToTensor(),
            normalize,
        ])
    else:
        transform = Compose([
            Resize(cfg.INPUT.SIZE_TEST),
            CenterCrop(cfg.INPUT.SIZE_TEST),
            ToTensor(),
            normalize
        ])
    return transform