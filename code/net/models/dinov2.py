import torch
from torch import nn
from types import SimpleNamespace
from einops import rearrange


class DINOv2NetMultiScale(nn.Module):
    """
    A wrapper around a pre-trained DINOv2 network (https://github.com/facebookresearch/dinov2)
    """
    def __init__(self, cfg):
        super(DINOv2NetMultiScale, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.feature_resize_factor = cfg.MODEL.FEATURE_RESIZE_FACTOR

        if isinstance(cfg.MODEL.MULTISCALE, int):
            self.blocks_ids =(23 - torch.arange(cfg.MODEL.MULTISCALE)).tolist()
        elif isinstance(cfg.MODEL.MULTISCALE, list):
            self.blocks_ids = cfg.MODEL.MULTISCALE
        else:
            raise TypeError

        self.patch_sz = cfg.MODEL.PATCH_SIZE
        assert self.patch_sz == 14, "MODEL.PATCH_SIZE has to be set to 14 for base DINOv2 model!"

        # load dino model 
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', cfg.MODEL.ARCH).to(self.device)

        # froze parameters
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.dinov2.eval()

    def forward(self, x, norm=True):
        with torch.no_grad():
            # list of [B (H W) C]
            out_ms = self.dinov2.get_intermediate_layers(x, n=self.blocks_ids, norm=norm)

        # NOTE: commented out, seems dinov2 already cuts the register tokens in the get_intermediate_layers function
        # out_ms = [out_ms[i][:, self.dinov2.num_register_tokens:, :] for i in range(0, len(out_ms))]

        return out_ms

class  DINOv2NetMultiScaleBaseline(DINOv2NetMultiScale):
    def __init__(self, cfg):
        super(DINOv2NetMultiScaleBaseline, self).__init__(cfg)

        num_layers = len(self.blocks_ids)
        self.decoder = torch.nn.Sequential(
                nn.Conv2d(num_layers*cfg.MODEL.EMB_SIZE, num_layers*cfg.MODEL.EMB_SIZE, kernel_size=1, stride=1),
                nn.GELU(), 
                nn.Conv2d(num_layers*cfg.MODEL.EMB_SIZE, cfg.MODEL.NUM_CLASSES, kernel_size=1, stride=1)
            )

    def forward(self, x):
        # list of [B (H W) C]
        emb_list = DINOv2NetMultiScale.forward(self, x, norm=True)
        emb = rearrange(torch.cat(emb_list, dim=-1), "b (h w) c -> b c h w", h=int(x.shape[2]/self.patch_sz))
        emb = torch.nn.functional.interpolate(emb, scale_factor = self.feature_resize_factor, mode="bilinear")
        
        # [B, num_classes, xH, xW]
        logits_embshape = self.decoder(emb)
        logits = torch.nn.functional.interpolate(logits_embshape, size=x.shape[-2:], mode="bilinear")

        logits = rearrange(logits, "b c xh xw -> b xh xw c")
        emb = rearrange(emb, "b c ph pw -> b ph pw c")
        logits_embshape = rearrange(logits_embshape, "b c ph pw -> b ph pw c")
        return SimpleNamespace(logits = logits, emb = emb, logits_embshape=logits_embshape)

