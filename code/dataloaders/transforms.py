import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor, Normalize, InterpolationMode, PILToTensor, RandomCrop

class ToTensorSN():
    def __init__(self, normalize01=True):
        if normalize01:
            self._op = ToTensor()
        else:
            self._op = PILToTensor()
        self._op_target = PILToTensor()
    def __call__(self, x_sn):
        x_sn.image = self._op(x_sn.image)
        if x_sn.label is not None:
            x_sn.label = self._op_target(x_sn.label).long().squeeze()
        return x_sn 

class NormalizeSN():
    def __init__(self, mean, std):
        self._op = Normalize(mean=mean, std=std)

    def __call__(self, x_sn):
        x_sn.image = self._op(x_sn.image)
        return x_sn

class ResizeLongestSideDivisible():
    def __init__(self, img_size, divider, eval_mode=False, randomcrop=False, hflip=False):
        self.divider = divider
        self.eval_mode = eval_mode
        self.randomcrop = randomcrop
        self.hflip = hflip

        # Transform input to DINO format:
        #  - can have arbitrary size, but it must be divisible by 14 (patch size of the used VIT backbone)
        #  - keeps aspect ratio, no padding needed
        if isinstance(img_size, list) and len(img_size) == 2:
            self.img_sz = img_size
            if self.img_sz[0] % self.divider > 0 or self.img_sz[1] % self.divider > 0:
                raise RuntimeError(f"INPUT.IMG_SIZE has to be divisible by 14")
        elif isinstance(img_size, int) and img_size % self.divider == 0:
            # longest side stored in IMG_SIZE
            self.img_sz = img_size
        else:
            raise RuntimeError(f"INPUT.IMG_SIZE has to be list[2] or int and divisible by {divider}!")

    def __call__(self, x_sn):
        x_size = x_sn.image.shape[-2:]

        if not self.eval_mode:
            if self.randomcrop:
                # i, j, h, w = RandomResizedCrop.get_params(x_sn.image, scale=[0.75, 1.0], ratio=[0.75, 4.0/3.0])

                if x_size[0] >= x_size[1]:
                    factor = x_size[0] / float(self.img_sz)
                    size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
                else:
                    factor = x_size[1] / float(self.img_sz)
                    size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 

                i, j, h, w = RandomCrop.get_params(x_sn.image, size)
                x_sn.image = F.crop(x_sn.image, i, j, h, w)
                x_sn.label = F.crop(x_sn.label, i, j, h, w)

            if self.hflip and torch.rand(1) < 0.5:
                x_sn.image = F.hflip(x_sn.image)
                x_sn.label = F.hflip(x_sn.label)
        else:
            # assuming x is tensor of [..., h, w] shape
            self.img_sz = int((np.max(x_size) // self.divider) * self.divider)

        if isinstance(self.img_sz, list):
            size = self.img_sz
        else:
            if x_size[0] >= x_size[1]:
                factor = x_size[0] / float(self.img_sz)
                size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
            else:
                factor = x_size[1] / float(self.img_sz)
                size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 
        x_sn.image = torchvision.transforms.functional.resize(x_sn.image, size, antialias=True)
        if x_sn.label is not None:
            x_sn.label = torchvision.transforms.functional.resize(x_sn.label[None, ...], size, interpolation=InterpolationMode.NEAREST)[0, ...]
        return x_sn

