import os
import torch
from config import get_cfg_defaults

class Saver(object):
    def __init__(self, cfg):
        self.experiment_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME)
        self.experiment_checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.experiment_code_dir = os.path.join(self.experiment_dir, "code")
        os.makedirs(self.experiment_checkpoints_dir, exist_ok=True)
        os.makedirs(self.experiment_code_dir, exist_ok=True)
        os.system("rsync -avm --exclude='_*/' --exclude='out/' --exclude='data/' --include='*/' --include='*.py' --exclude='*' ./ " + self.experiment_code_dir)

    def save_checkpoint(self, state, is_best, filename="checkpoint-latest.pth", model=None, save_trainable_only=False):
        if is_best:
            filename = "checkpoint-best.pth"
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a') as f:
                f.write(str(state["epoch"]) + ", " + str(best_pred) + "\n")
        if model is not None and save_trainable_only:
            trainable_params = {n: state["state_dict"][n] for n, p in model.named_parameters() if p.requires_grad}
            reg_buffers = {n: state["state_dict"][n] for n, _ in model.named_buffers()}
            state["state_dict"] = trainable_params 
            state["state_dict"].update(reg_buffers)
           # print("===========")
           # print("Saved state_dict:")
           # print(state["state_dict"])
           # print("===========")
            state["save_trainable_only"] = True
        else:
            state["save_trainable_only"] = False

        torch.save(state, os.path.join(self.experiment_checkpoints_dir, filename))

    def save_metrics(self, epoch, metrics):
        out_dir = os.path.join(self.experiment_dir, 'metrics')

        for (m, v) in metrics.items():
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"metric_{m}.txt"), 'a') as f:
                f.write(str(epoch) + ", " + str(v) + "\n")
    
    @staticmethod
    def load_checkpoint(checkpoint_file, model, optimizer=None, device="cuda", map_location="cpu"):
        state = {}
        if checkpoint_file is not None:
            if not os.path.isfile(checkpoint_file):
                raise RuntimeError(f"=> Resume checkpoint does not exist! ({checkpoint_file})")

            checkpoint = torch.load(checkpoint_file, map_location=map_location)

            # NOTE: I think that because of torch.optimize, the trainable weights are saved with the "_orig_mod." prefix, so remove it
            for key in list(checkpoint['state_dict'].keys()):
                if '_orig_mod.' in key:
                    checkpoint['state_dict'][key.replace('_orig_mod.', '')] = checkpoint['state_dict'][key]
                    del checkpoint['state_dict'][key]
            
            strict = not checkpoint.get("save_trainable_only", False)
            if not strict:
                print ("Saved model stores only tranable weights of model --> disabling strict model loading")
                model_state = model.state_dict()
                no_match = { k:v.size() for k,v in checkpoint['state_dict'].items() 
                            if (k in model_state and v.size() != model_state[k].size()) or (k not in model_state) }
                print("    Number of not matched parts: ", len(no_match))
                print("-----------------")
                print(no_match)
                print("-----------------")

            try:
                model.load_state_dict(checkpoint['state_dict'], strict=strict)
                state["start_epoch"] = checkpoint['epoch']
                if optimizer is not None:
                    print("Loading optimizer state ...")
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for state_opt in optimizer.state.values():
                        for k, v in state_opt.items():
                            if torch.is_tensor(v):
                                state_opt[k] = v.to(device)
                if "best_pred" in checkpoint: 
                    state["best_pred"] = checkpoint['best_pred']
            except:
                # finetuning or using some part of pretrained model
                print(f"Failed to load original model {checkpoint_file}") 
                print("    Loading only matching layers ...")
                print("    Not loading saved optimizer ...")
                pretrained_state = { k:v for k,v in checkpoint['state_dict'].items() if k in model_state and v.size() == model_state[k].size() }
                no_match = { k:v.size() for k,v in checkpoint['state_dict'].items() if (k in model_state and v.size() != model_state[k].size()) or (k not in model_state) }
                print("    Not matched parts: \n", no_match)
                model_state.update(pretrained_state)
                model.load_state_dict(model_state, strict=False)

            custom_data = checkpoint.get("custom_data", {})
            if hasattr(model, "custom_data"):
                model.custom_data = custom_data
            
            print(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")
        return state

    def save_experiment_config(self, cfg):
        with open(os.path.join(self.experiment_dir, 'parameters.yaml'), 'w') as f:
            f.write(cfg.dump())


def load_experiment_cfg(exp_dir):
    cfg_local = get_cfg_defaults() 
    # read the experiment parameters
    if os.path.isfile(os.path.join(exp_dir, "parameters.yaml")):
        with open(os.path.join(exp_dir, "parameters.yaml"), 'r') as f:
            cc = cfg_local._load_cfg_from_yaml_str(f)
        cfg_local.merge_from_file(os.path.join(exp_dir, "parameters.yaml"))
        cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME
    else:
        raise RuntimeError(f"Experiment directory does not contain parameters.yaml: {exp_dir}")
    return cfg_local
