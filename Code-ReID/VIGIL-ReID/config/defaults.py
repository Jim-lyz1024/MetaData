from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "ReIDTrainer"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.BACKBONE = "ViT-B/16"
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.NECK = 'bnneck'
_C.MODEL.NECK_FEAT = 'after'
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.IF_LABELSMOOTH = 'on'
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'

# Loss weights
_C.MODEL.ID_LOSS_WEIGHT = 0.25
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [224, 224]  # CLIP default size
_C.INPUT.SIZE_TEST = [224, 224]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PADDING = 10
_C.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]  # CLIP normalization
_C.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = 'market1501'
_C.DATASETS.ROOT_DIR = '../data'

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = 'softmax'
_C.DATALOADER.NUM_INSTANCE = 4

# For train
_C.DATALOADER.TRAIN = CN()
_C.DATALOADER.TRAIN.BATCH_SIZE = 32
_C.DATALOADER.TRAIN.SAMPLER = 'RandomSampler'

# For test
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.BATCH_SIZE = 32
_C.DATALOADER.TEST.SAMPLER = 'SequentialSampler'

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 120
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3
_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)

# Stage 1
_C.SOLVER.STAGE1 = CN()
_C.SOLVER.STAGE1.IMS_PER_BATCH = 1
_C.SOLVER.STAGE1.BASE_LR = 0.00055
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.00001
_C.SOLVER.STAGE1.LR_MIN = 1e-6
_C.SOLVER.STAGE1.WARMUP_METHOD = 'linear'
_C.SOLVER.STAGE1.WEIGHT_DECAY = 1e-4
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 1e-4
_C.SOLVER.STAGE1.MAX_EPOCHS = 40
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 40
_C.SOLVER.STAGE1.LOG_PERIOD = 250
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"

# Stage 2
_C.SOLVER.STAGE2 = CN()
_C.SOLVER.STAGE2.IMS_PER_BATCH = 32
_C.SOLVER.STAGE2.BASE_LR = 0.000008
_C.SOLVER.STAGE2.WARMUP_METHOD = 'linear'
_C.SOLVER.STAGE2.WARMUP_ITERS = 10
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.1
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0001
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0001
_C.SOLVER.STAGE2.MAX_EPOCHS = 60
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 60
_C.SOLVER.STAGE2.LOG_PERIOD = 10
_C.SOLVER.STAGE2.EVAL_PERIOD = 60
_C.SOLVER.STAGE2.STEPS = [30, 50]
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = 'after'
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.RE_RANKING = False
_C.TEST.EVALUATOR = "ReIDEvaluator"
_C.TEST.EVAL_PERIOD = 10
_C.TEST.MAX_RANK = 50
_C.TEST.REMOVE_IDENTICAL = False
_C.TEST.EVAL = True

# ---------------------------------------------------------------------------- 
# Misc options
# ---------------------------------------------------------------------------- 
_C.OUTPUT_DIR = "./output"
_C.SEED = -1
_C.LOG_PERIOD = 50

def get_cfg_defaults():
    """Get default configs."""
    return _C.clone()