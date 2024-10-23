import os
import sys
import torch
import importlib
import numpy as np
from types import SimpleNamespace
from torchvision.transforms import ToPILImage
from einops import rearrange

class PixOOD():
    def __init__(self, exp_dir, **kwargs_global) -> None:
        self.exp_dir = exp_dir
        self.code_dir = os.path.join(self.exp_dir, "code")

        # store all loaded modules so it can be later restored to the same state
        pre_modules_keys = []
        for k, _ in sys.modules.items():
            pre_modules_keys.append(k)

        cfg_local = get_experiment_cfg(self.exp_dir)

        cfg_local.EXPERIMENT.RESUME_CHECKPOINT = os.path.join(self.exp_dir, "checkpoints", "checkpoint-latest.pth")
        if not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT):
            raise RuntimeError(f"Experiment dir does not contain valid checkpoint!\n \t ==> file {cfg_local.EXPERIMENT.RESUME_CHECKPOINT} not found!")

        # CUDA
        if not torch.cuda.is_available():
            print ("GPU is disabled")
            cfg_local.SYSTEM.USE_GPU = False

        self.cfg = cfg_local

        self.device = torch.device("cuda" if cfg_local.SYSTEM.USE_GPU else "cpu")

        # define the network
        sys.path.insert(0, self.code_dir)
        kwargs = {'cfg': cfg_local}
        spec = importlib.util.spec_from_file_location(cfg_local.MODEL.FILENAME, os.path.join(self.code_dir, "net", "models", cfg_local.MODEL.FILENAME + ".py"))
        model_module = spec.loader.load_module()
        print (self.code_dir, model_module)
        self.model = getattr(model_module, cfg_local.MODEL.NET)(**kwargs)
        # load input preprocessing for the network
        spec = importlib.util.spec_from_file_location("augmentations", os.path.join(self.code_dir, "dataloaders", "augmentations.py"))
        augment_module = spec.loader.load_module()
        self.transforms = getattr(augment_module, cfg_local.DATASET.AUGMENT)().test(cfg_local)
        # clean up the inserted code path
        sys.path = sys.path[1:]

        # load the model paraters
        checkpoint = torch.load(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")
        for key in list(checkpoint['state_dict'].keys()):
            if '_orig_mod.' in key:
                checkpoint['state_dict'][key.replace('_orig_mod.', '')] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]
        
        strict = not checkpoint.get("save_trainable_only", False)
        if not strict:
            print ("Saved model stores only tranable weights of model --> disabling strict model loading")
            model_state = self.model.state_dict()
            no_match = { k:v.size() for k,v in checkpoint['state_dict'].items() if (k in model_state and v.size() != model_state[k].size()) or (k not in model_state)}
            print("    Number of not matched parts: ", len(no_match))
            print("-----------------")
            print(no_match)
            print("-----------------")

        self.model.load_state_dict(checkpoint['state_dict'], strict=strict)
        custom_data = checkpoint.get("custom_data", {})
        if hasattr(self.model, "custom_data"):
            self.model.custom_data = custom_data
        
        print("=> loaded checkpoint '{}' (epoch {})".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, checkpoint['epoch']))
        del checkpoint

        # Using cuda
        self.model.to(self.device)
        self.model.eval()

        # clean-up imported modules
        to_del = []
        for k, _ in sys.modules.items():
            if k not in pre_modules_keys and any(m in k for m in ["config", "dataloaders", "helpers", "net"]):
                to_del.append(k)
        for k in to_del:
            del sys.modules[k]

        # Cityscapes labels
        # 0:road 1:sidewalk 2:building 3:wall 4:fence
        # 5:pole 6:traffic light 7:traffic sign
        # 8:vegetation 9:terrain 10:sky 11:person
        # 12:rider 13:car 14:truck 15:bus
        # 16:train 17:motorcycle 18:bicycle
        if "eval_labels" not in kwargs_global.keys():
            self.eval_labels = [0, 1] 
            print(f"Using default road+sidewalk labels for anomaly detection!")
        elif len(kwargs_global["eval_labels"]) == 0:
            self.eval_labels = np.arange(self.cfg.MODEL.NUM_CLASSES).tolist()
            print(f"Using all labels for anomaly detection: {*self.eval_labels,}")
        else:
            self.eval_labels =  kwargs_global["eval_labels"]
            print(f"Using labels {*self.eval_labels,} for anomaly detection!")

        self.eval_scale_factor = kwargs_global.get("eval_scale_factor", 1)
        print(f"Using emb scale factor {self.eval_scale_factor}")

    def evaluate(self, input_pil_image, return_anomaly_score=True):
        if torch.is_tensor(input_pil_image):
            if len(input_pil_image.shape) > 3:
                orig_size = input_pil_image.shape[-2:]
                #assumes tensor [B, 3, H, W] in range (0, 1)
                assert ((input_pil_image.min().item() >= 0.0) and (input_pil_image.max().item() <= 1.0)), f"The input tensor is not in range (0, 1). ({input_pil_image.min().item()}, {input_pil_image.max().item()})"
                x = []
                for b in range(0, input_pil_image.shape[0]):
                    pi = ToPILImage()(input_pil_image[b, ...])
                    x.append(self.transforms(SimpleNamespace(image=pi, label=None, image_name="")).image) 
                x = torch.stack(x, dim=0).to(self.device)
            else:
                assert False, f"No batch dimension of input tensor: {input_pil_image.shape}."
        else:
            orig_size = [input_pil_image.height, input_pil_image.width]
            #assumes single pil image
            input_sn = self.transforms(SimpleNamespace(image=input_pil_image, label=None, image_name=""))
            x = input_sn.image.to(self.device)[None, ...]

        with torch.no_grad():
            out = self.model(x, eval_scale_factor=self.eval_scale_factor)

        # convert outputs to the original pil image resolution 
        # "b h w" 
        pred_score_hires = torch.nn.functional.interpolate(out.pred_score[:, None, ...], size=orig_size, mode="nearest")
        pred_score_hires = pred_score_hires.squeeze().cpu()

        pred_score_hires_all = torch.nn.functional.interpolate(rearrange(out.pred_score_all, "b h w c -> b c h w"), size=orig_size, mode="bilinear")
        pred_score_hires_all = rearrange(pred_score_hires_all, "b c h w -> b h w c").squeeze().cpu()

        pred_y_hires = torch.nn.functional.interpolate(out.pred_y.float()[:, None, ...], size=orig_size, mode="nearest").long()
        pred_y_hires = pred_y_hires.squeeze().cpu()

        # for the custom benchmarks evaluator
        if return_anomaly_score:
            p = torch.max(pred_score_hires_all[..., self.eval_labels], dim=-1)[0]
            return (1.0-p)
        else:
            return SimpleNamespace(pred_y=pred_y_hires,
                                   pred_score=pred_score_hires,
                                   pred_y_orig=out.pred_y.squeeze().cpu(),
                                   pred_score_orig=out.pred_score.squeeze().cpu(),
                                   pred_score_all=pred_score_hires_all,
                                   out=out)

def get_experiment_cfg(exp_dir):
    code_dir = os.path.join(exp_dir, "code")
    #from config import get_cfg_defaults
    config_module = importlib.util.spec_from_file_location("get_cfg_defaults", os.path.join(code_dir, "config", "defaults.py")).loader.load_module()
    cfg_fnc = getattr(config_module, "get_cfg_defaults")
    cfg_local = cfg_fnc()

    # read the experiment parameters
    if os.path.isfile(os.path.join(exp_dir, "parameters.yaml")):
        with open(os.path.join(exp_dir, "parameters.yaml"), 'r') as f:
            cc = cfg_local._load_cfg_from_yaml_str(f)
        cfg_local.merge_from_file(os.path.join(exp_dir, "parameters.yaml"))
        cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME
    else:
        raise RuntimeError(f"Experiment directory does not contain parameters.yaml: {exp_dir}")
    return cfg_local

