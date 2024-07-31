import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import random
import time
from datetime import datetime
import importlib
from types import SimpleNamespace

from config import get_cfg_defaults
from dataloaders import make_data_loader
from helpers.saver import Saver
from helpers.logger import Logger, with_debugger 


class Trainer(object):
    '''
    A class which defines the training/validation steps and storing of the progress.
    '''
    def __init__(self, cfg):
        self.cfg = cfg.clone()

        # Saver: before running the experiment, it copies the python code and saves the cfg to yaml to the output directory
        #   That is useful when running many experiments with different versions of the code
        self.saver = Saver(cfg)
        self.saver.save_experiment_config(cfg)

        self.device = torch.device("cuda" if cfg.SYSTEM.USE_GPU else "cpu")

        # build the dataloader
        kwargs = {'num_workers': cfg.SYSTEM.NUM_CPU, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(cfg, **kwargs)

        self.nclass = cfg.MODEL.NUM_CLASSES
        assert self.nclass == len(cfg.DATASET.SELECTED_LABELS), "The number of selected labels must be same as MODEL.NUM_CLASSES"

        # build the network
        kwargs = {'cfg': cfg}
        model_module = importlib.import_module("net.models." + cfg.MODEL.FILENAME)
        self.model = getattr(model_module, cfg.MODEL.NET)(**kwargs)
        
        # define the optimizer
        print (f"OPTIMIZER {cfg.OPTIMIZER.METHOD}: Training the model with LR: {cfg.OPTIMIZER.LR:0.5f}")

        if hasattr(self.model, "get_trainable_parameters"):
            train_params = self.model.get_trainable_parameters()
            print(f"Different training param groups: {len(train_params)}")
            for i in range(0, len(train_params)):            
                lr = train_params[i].get("lr", self.cfg.OPTIMIZER.LR)
                print(f"\t lr: {lr}")
        else:
            train_params = self.model.parameters()

        if cfg.OPTIMIZER.METHOD == "sgd":
            self.optimizer = torch.optim.SGD(train_params, lr=cfg.OPTIMIZER.LR, momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, 
                                    nesterov=cfg.OPTIMIZER.NESTEROV)
        elif cfg.OPTIMIZER.METHOD == "adamw":
            self.optimizer = torch.optim.AdamW(train_params, lr=cfg.OPTIMIZER.LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
        elif cfg.OPTIMIZER.METHOD == "adam":
            self.optimizer = torch.optim.Adam(train_params, lr=cfg.OPTIMIZER.LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
        else:
            raise NotImplementedError

        # losses
        kwargs = {'cfg': cfg, 'device': self.device}
        criterion_module = importlib.import_module("net.losses")
        self.criterions = []
        self.criterion_weights = cfg.LOSS.WEIGHTS
        for loss_type in cfg.LOSS.TYPE:
            self.criterions.append(getattr(criterion_module, loss_type)(**kwargs))

        # LR scheduler
        if cfg.OPTIMIZER.SCHEDULER == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.EXPERIMENT.EPOCHS)
        elif cfg.OPTIMIZER.SCHEDULER == "cosinerestart":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        elif cfg.OPTIMIZER.SCHEDULER == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.EXPERIMENT.EPOCHS, gamma=0.1)
        elif cfg.OPTIMIZER.SCHEDULER == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90, 120])
        elif cfg.OPTIMIZER.SCHEDULER == "none":
            class OptimizerDummy():
                def get_last_lr(self):
                    return [cfg.OPTIMIZER.LR]
                def step(self):
                    pass
            self.scheduler = OptimizerDummy()
        else:
            raise NotImplementedError

        self.epochs = cfg.EXPERIMENT.EPOCHS 

        # Define Evaluator ?
        if cfg.EXPERIMENT.EVAL_METRIC is not None:
            evaluation_module = importlib.import_module("helpers.metrics")
            self.evaluator = getattr(evaluation_module, cfg.EXPERIMENT.EVAL_METRIC)(cfg)
        else:
            self.evaluator = None

        # resuming checkpoint
        checkpoint_state = self.saver.load_checkpoint(cfg.EXPERIMENT.RESUME_CHECKPOINT, self.model, optimizer=self.optimizer, device=self.device)
        self.start_epoch = checkpoint_state.get("start_epoch", 0)
        if self.evaluator is None:
            self.best_pred = checkpoint_state.get("best_pred", 1e9)
        else:
            self.best_pred = checkpoint_state.get("best_pred", 0.0)
        
        print (f"Names of trainable parameters: {[p[0] for p in self.model.named_parameters() if p[1].requires_grad]}")

        self.model.to(self.device)
        # self.model = torch.compile(self.model)

    def training(self, epoch):
        '''
        Training step
        '''
        lr_str = "[" + ", ".join([f"{lr:.3e}" for lr in self.scheduler.get_last_lr()]) + "]"
        print(f"\n=>Epochs {epoch}, learning rate = {lr_str}, previous best = {self.best_pred:0.4f}, EXP_NAME {self.cfg.EXPERIMENT.NAME}")

        train_loss = 0.0
        self.model.train()
        start_time = time.time()
        if hasattr(self.model, "training_step"):
            # custom training step method inside the model
            time_measure_dict, train_count, partial_loss, train_loss = self.model.training_step(
                    epoch, self.train_loader, self.criterions, self.criterion_weights, self.optimizer)
        else:
            time_measure_dict = {"forward": 0.0, "loss": 0.0, "backward": 0.0, "batch": 0.0}
            train_count = 0

            partial_loss = [0.0 for _ in range(len(self.criterions))]
            tbar = tqdm(self.train_loader)
            for i, sample in enumerate(tbar):
                tmp_time_batch = time.time()

                image, target = sample[0].to(self.device), sample[1].to(self.device)
                self.optimizer.zero_grad()

                tmp_time = time.time()
                output = self.model(image, target)
                time_measure_dict["forward"] += time.time() - tmp_time

                tmp_time = time.time()
                # add metadata
                if isinstance(output, dict):
                    output["__epoch__"] = epoch
                if isinstance(output, SimpleNamespace):
                    output.__epoch__ = epoch

                loss = 0.0
                for t in range(len(self.criterions)):
                    loss_val = self.criterions[t](output, target)
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        print(f"Loss NaN/Inf ERROR: partial loss {t} = {loss_val.item()}")
                        assert False
                    loss += self.criterion_weights[t] * loss_val
                    partial_loss[t] += self.criterion_weights[t] * loss_val.item()
                time_measure_dict["loss"] += time.time() - tmp_time
                tmp_time = time.time()
                loss.backward()

                if self.cfg.OPTIMIZER.CLIP_GRAD > 0:
                    trainable_params_names = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
                    for item in trainable_params_names:
                        if torch.any(torch.isnan(item[1].grad)) or torch.any(torch.isinf(item[1].grad)):
                            print(item[0], ":-----------")
                            print(item[1].grad)
                        
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.OPTIMIZER.CLIP_GRAD, norm_type=2.0, error_if_nonfinite=True)

                self.optimizer.step()               
                time_measure_dict["backward"] += time.time() - tmp_time

                train_loss += loss.item()
                train_count += 1
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                time_measure_dict["batch"] += time.time() - tmp_time_batch

                if hasattr(self.model, "post_batch"):
                    self.model.post_batch(image, target, len(self.train_loader), epoch)

        self.scheduler.step()
        print('Train: [Epoch: %d, Loss: %.3f, Time: %.3f(min)]' % (
            epoch, 
            train_loss / train_count, 
            (time.time()-start_time)/60.0))
        print(f"Partial train losses: {[L/train_count for L in partial_loss]}")

        print('Train avg times: [forward: %.3f(sec), loss: %.3f(sec), backward: %.3f(sec), batch: %.3f(sec)]' % (
            time_measure_dict["forward"] / train_count, 
            time_measure_dict["loss"] / train_count, 
            time_measure_dict["backward"] / train_count, 
            time_measure_dict["batch"] / train_count))

        self.save_checkpoint(epoch, False)
        self.saver.save_metrics(epoch+1, {'train_loss': train_loss / train_count})

    def validation(self, epoch):
        '''
        Test the current model.
        '''

        if len(self.val_loader) == 0:
            print(f"Validation: [Epoch: {epoch}, num. samples: 0], skipping ...")
            self.saver.save_metrics(epoch+1, {'val_loss': 0.0})
            return

        self.model.eval()
        if self.evaluator is not None:
            self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        start_time = time.time()
        test_count = 0
        for i, sample in enumerate(tbar):
            with torch.no_grad():
                image, target = sample[0].to(self.device), sample[1].to(self.device)

                output = self.model(image, target)

                # add metadata
                if isinstance(output, dict):
                    output["__epoch__"] = epoch
                if isinstance(output, SimpleNamespace):
                    output.__epoch__ = epoch

                loss = 0.0
                for t in range(len(self.criterions)):
                    loss += self.criterion_weights[t] * self.criterions[t](output, target)
                test_loss += loss.item()
                test_count += 1
                tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))

                # Add batch sample into evaluator
                if self.evaluator is not None:
                    self.evaluator.add_batch(target, output)

        test_loss /= test_count
        print('Validation: [Epoch: %d, num. samples: %5d, Loss: %.3f, Time: %.3f(min)]' % (
            epoch, 
            i * self.cfg.INPUT.BATCH_SIZE + image.data.shape[0], 
            test_loss, 
            (time.time()-start_time)/60.0))
        
        eval_metric = None
        if self.evaluator is not None:
            eval_metric = self.evaluator.compute_stats(printflag=True, epoch=epoch)

        if self.cfg.EXPERIMENT.USE_EVAL_METRIC_FOR_CHCK and self.evaluator is not None:
            if eval_metric > self.best_pred:
                self.best_pred = eval_metric
                self.save_checkpoint(epoch, True)
        else:
            if test_loss < self.best_pred:
                self.best_pred = test_loss
                self.save_checkpoint(epoch, True)

        self.saver.save_metrics(epoch+1, {'val_loss': test_loss, 'eval_metric': eval_metric})

    def save_checkpoint(self, epoch, best_flag):
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'custom_data': getattr(self.model, "custom_data", {}),
        }, best_flag, model=self.model, save_trainable_only=True)


@with_debugger
def main(args, unknown):
    # get default configuration
    cfg = get_cfg_defaults() 

    # for better performance of matmuls as a trade of for bit of precison
    torch.set_float32_matmul_precision('high')

    # overwrite the user specified config values
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    cfg.merge_from_list(unknown)

    if not torch.cuda.is_available() and cfg.SYSTEM.USE_GPU: 
        raise RuntimeError

    # generate a data-based experiment name when needed
    if cfg.EXPERIMENT.NAME is None:
        cfg.EXPERIMENT.NAME = datetime.now().strftime(r'%Y%m%d_%H%M%S.%f').replace('.','_')

    # default output directory
    if cfg.EXPERIMENT.OUT_DIR is None:
        cfg.EXPERIMENT.OUT_DIR = "./_out/experiments/"

    sys.stdout = Logger(os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in cfg.SYSTEM.GPU_IDS])
    
    if cfg.DATASET.SELECTED_LABELS == None or len(cfg.DATASET.SELECTED_LABELS) == 0:
        print("SELECTED_LABELS are empty => using all labels (0...MODEL.NUM_CLASSES)!")
        cfg.DATASET.SELECTED_LABELS = np.arange(cfg.MODEL.NUM_CLASSES).tolist()
    elif len(cfg.DATASET.SELECTED_LABELS) == 3 and cfg.DATASET.SELECTED_LABELS[0] < 0:
        print(f"SELECTED_LABELS are range => using labels from {cfg.DATASET.SELECTED_LABELS[1]} to {cfg.DATASET.SELECTED_LABELS[2]} including boundary!")
        cfg.DATASET.SELECTED_LABELS = np.arange(cfg.DATASET.SELECTED_LABELS[1], cfg.DATASET.SELECTED_LABELS[2]).tolist() + [cfg.DATASET.SELECTED_LABELS[2]]
        print(f"Total number of SELECTED_LABELS {len(cfg.DATASET.SELECTED_LABELS)}.")
    
    # to avoid accidental overwrite of the settings
    cfg.freeze()

    # output the full setup (gets stored in the log)
    print("CMD: python3", " ".join(sys.argv))
    print(cfg)

    if cfg.SYSTEM.RNG_SEED is not None:
        print (f"Setting rng. seed to {cfg.SYSTEM.RNG_SEED}")
        # fix rng seeds 
        torch.manual_seed(cfg.SYSTEM.RNG_SEED)
        np.random.seed(cfg.SYSTEM.RNG_SEED)
        random.seed(cfg.SYSTEM.RNG_SEED)
        # Note that training is still non-deterministic because of cudnn implementation
        # you can set torch.backends.cudnn.deterministic = True
        # but it slows down training and may cause problems when restarting training from checkpoints

    trainer = Trainer(cfg)
    print(trainer.model)

    print("Saving experiment to:", trainer.saver.experiment_dir) 

    if hasattr(trainer.model, "pre_training"):
        trainer.model.pre_training(trainer)

    # if no training is needed, just create a checkpoint and exit
    if cfg.EXPERIMENT.SKIP_EPOCHS:
        print("Skipping epochs, saving model only.")
        epoch = 0
    else:
        print('Starting Epoch:', trainer.start_epoch)
        print('Total Epochs:', trainer.epochs)
        print (f"Number of trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}") 

        for epoch in range(trainer.start_epoch, trainer.epochs):
            print(datetime.now())

            if hasattr(trainer.model, "pre_epoch"):
                trainer.model.pre_epoch(epoch, trainer)

            trainer.training(epoch)

            if (epoch % cfg.EXPERIMENT.EVAL_INTERVAL) == (cfg.EXPERIMENT.EVAL_INTERVAL - 1):
                trainer.validation(epoch)

            if hasattr(trainer.model, "post_epoch"):
                trainer.model.post_epoch(epoch, trainer)

    if hasattr(trainer.model, "post_training"):
        trainer.model.post_training(trainer)
        trainer.save_checkpoint(epoch, False)

    print("Experiment {} done.".format(cfg.EXPERIMENT.NAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='', help='YAML Configuration file for experiment (it overrides the default settings).')
    args, unknown = parser.parse_known_args()
    main(args, unknown)

