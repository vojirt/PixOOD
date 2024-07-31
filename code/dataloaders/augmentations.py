from torchvision.transforms import Compose
from dataloaders.transforms import NormalizeSN, ToTensorSN, ResizeLongestSideDivisible

class DINOv2Augmentation():
    # Transform input to DINOv2 format:
    #  - values are normalized by mean and std of imagenet (see IMAGENET_DEFAULT_{MEAN, STD} 
    #    from https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py) 
    #  - can have arbitrary size, but it must be divisible by 14 (patch size of the used VIT backbone)
    #  - keeps aspect ratio, no padding needed

    def train(self, cfg):
        augment = Compose([
            ToTensorSN(),
            ResizeLongestSideDivisible(cfg.INPUT.IMG_SIZE, cfg.MODEL.PATCH_SIZE, randomcrop=cfg.INPUT.RANDOMCROP_AUG, hflip=cfg.INPUT.HFLIP_AUG),
            NormalizeSN(mean=cfg.INPUT.NORM_MEAN, std=cfg.INPUT.NORM_STD)
        ])
        return augment

    def val(self, cfg):
        augment = Compose([
            ToTensorSN(),
            ResizeLongestSideDivisible(cfg.INPUT.IMG_SIZE, cfg.MODEL.PATCH_SIZE, eval_mode=False, randomcrop=False),
            NormalizeSN(mean=cfg.INPUT.NORM_MEAN, std=cfg.INPUT.NORM_STD)
        ])
        return augment
    
    def test(self, cfg):
        augment = Compose([
            ToTensorSN(),
            ResizeLongestSideDivisible(cfg.INPUT.IMG_SIZE, cfg.MODEL.PATCH_SIZE, eval_mode=True, randomcrop=False),
            NormalizeSN(mean=cfg.INPUT.NORM_MEAN, std=cfg.INPUT.NORM_STD)
        ])
        return augment

