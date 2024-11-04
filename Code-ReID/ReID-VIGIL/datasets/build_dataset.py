from utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")

def build_dataset(cfg):
    """Build a dataset.
    
    Args:
        cfg (CfgNode): Config containing dataset settings
    """
    avai_datasets = DATASET_REGISTRY.registered_names()
    print("Available datasets:", avai_datasets)
    print("Requested dataset:", cfg.DATASETS.NAMES)
    check_availability(cfg.DATASETS.NAMES, avai_datasets)
    print("Building dataset: {}".format(cfg.DATASETS.NAMES))
    
    return DATASET_REGISTRY.get(cfg.DATASETS.NAMES)(cfg)