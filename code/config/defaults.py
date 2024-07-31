from yacs.config import CfgNode as CN

# -------------------------
_C = CN(new_allowed=True)
_C.SYSTEM = CN(new_allowed=True)
_C.EXPERIMENT = CN(new_allowed=True)
_C.OPTIMIZER = CN(new_allowed=True)
_C.LOSS = CN(new_allowed=True)
_C.INPUT = CN(new_allowed=True)
_C.DATASET = CN(new_allowed=True)
_C.MODEL = CN(new_allowed=True)

# -------------------------
_C.SYSTEM.NUM_CPU = 4     
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.GPU_IDS = [0]                     # which gpus to use for training - list of int, e.g. [0, 1], CURRENTLY ONLY SINGLE GPU IS SUPPORTED
_C.SYSTEM.RNG_SEED = 42                     # to make the code (almost) deterministic

# -------------------------
_C.EXPERIMENT.NAME = None                   # None == Auto name from date and time 
_C.EXPERIMENT.OUT_DIR = None                # where to store the training experiment (default: _out/experiments)
_C.EXPERIMENT.EPOCHS = 400                  # number of training epochs
_C.EXPERIMENT.RESUME_CHECKPOINT = None      # specify the checkpoint to resume (path to the .pth file)
_C.EXPERIMENT.EVAL_INTERVAL = 1             # eval on val data every X epoch
_C.EXPERIMENT.EVAL_METRIC = None            
_C.EXPERIMENT.USE_EVAL_METRIC_FOR_CHCK = False            
_C.EXPERIMENT.SKIP_EPOCHS = False           # skips training, just save the checkpoint (for CLIP or other pre-trained models)
_C.EXPERIMENT.RECOMPUTE_NM = True

# -------------------------
_C.OPTIMIZER.METHOD = "adamw"               # ["sgd", "adamw", "adam"]
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.NESTEROV = False
_C.OPTIMIZER.SCHEDULER = "cosine"           # ["cosine", "step", "multistep"]
_C.OPTIMIZER.CLIP_GRAD = -1.0

# -------------------------
_C.LOSS.TYPE = ["CrossEntropyLoss"]         # one or more losses
_C.LOSS.WEIGHTS = [1.0]                     # weights of individual losses
_C.LOSS.IGNORE_LABEL = 255

# -------------------------
_C.DATASET.AUGMENT = "NoAugmentation"       # name of a class from dataloaders/augmentations.py
_C.DATASET.SELECTED_LABELS = None           # None - use all labels; [L1, L2, L3, ...] - specify labels; [-1, minL, maxL] - range
_C.DATASET.OOD_SELECTED_LABELS = None       # None - use all remaining labels; [L1, L2, L3, ...] - specify labels; [-1, minL, maxL] - range

# -------------------------
_C.MODEL.FILENAME = ""                      # net/models/ filename
_C.MODEL.NET = ""                           # a class in FILENAME
_C.MODEL.EMB_SIZE = 512                     # embeding size
_C.MODEL.NUM_CLASSES = 10                   # number of classes the model is trained for
_C.MODEL.PATCH_SIZE = 14                    # vit net image -> token patch size 
_C.MODEL.FEATURE_RESIZE_FACTOR = 1 
_C.MODEL.INIT_TAU = 1 
_C.MODEL.TAU_NORM = False
_C.MODEL.PDF_DIST2NP = False
_C.MODEL.KNN_TYPE = "condensation"
_C.MODEL.KNN_REINIT = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

