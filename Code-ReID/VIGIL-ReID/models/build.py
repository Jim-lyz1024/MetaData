from utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODELS")

def build_model(cfg, num_classes):
    """Build a model.
    """
    avai_models = MODEL_REGISTRY.registered_names()
    print(f"Available models: {avai_models}")
    print(f"Requested model: {cfg.MODEL.NAME}")
    check_availability(cfg.MODEL.NAME, avai_models)
    print(f"Building model: {cfg.MODEL.NAME}")
    
    return MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg, num_classes)