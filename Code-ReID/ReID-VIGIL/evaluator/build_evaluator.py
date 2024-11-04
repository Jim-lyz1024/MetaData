from utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry("EVALUATOR")

def build_evaluator(cfg):
    """Build evaluator.
    
    Args:
        cfg (CfgNode): Config containing evaluator settings.
    """
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    print(f"Available evaluators: {avai_evaluators}")
    print(f"Requested evaluator: {cfg.TEST.EVALUATOR}")
    check_availability(cfg.TEST.EVALUATOR, avai_evaluators)
    print(f"Building evaluator: {cfg.TEST.EVALUATOR}")
    
    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg)