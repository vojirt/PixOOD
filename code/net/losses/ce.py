import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossEntropyLoss(object):
    def __init__(self, cfg, **kwargs):
        self.nll_loss = nn.NLLLoss(reduction="mean", ignore_index=cfg.LOSS.IGNORE_LABEL)

    def __call__(self, res, target):
        log_softmax = F.log_softmax(rearrange(res.logits, "b h w c -> (b h w) c"), dim=-1)
        ce_loss = self.nll_loss(log_softmax, rearrange(target.long(), "b h w -> (b h w)"))
        return ce_loss
