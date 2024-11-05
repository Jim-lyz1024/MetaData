from utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODEL")

def build_trainer(cfg):
    """Build trainer.
    
    Args:
        cfg (CfgNode): Config containing trainer settings.
    """
    avai_trainers = MODEL_REGISTRY.registered_names()
    check_availability(cfg.MODEL.NAME, avai_trainers)
    print(f"Building trainer: {cfg.MODEL.NAME}")
    
    return MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)