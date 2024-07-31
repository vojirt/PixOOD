import numpy
import importlib
import random
import torch
from torch.utils.data import DataLoader

from dataloaders.cityscapes import CityscapesDataset

# Contains paths to root dirs of different datasets
def get_root_dir(dataset_name):
    if dataset_name == "cityscapes":
        assert False, "You need to set path to the cityscapes dataset here"
        return "<ABSOLUTE PATH TO DATASET>/CityScapes/"
    return ""

def make_datasets(cfg):
    # load the specified augmentation
    augment_module = importlib.import_module("dataloaders.augmentations")
    aug_maker = getattr(augment_module, cfg.DATASET.AUGMENT)()
    augment = aug_maker.train(cfg)
    test_augment = aug_maker.test(cfg)
    val_augment = aug_maker.val(cfg) if hasattr(aug_maker, "val") else test_augment

    if cfg.DATASET.TRAIN == "cityscapes":
        train_set = CityscapesDataset(cfg, root=get_root_dir(cfg.DATASET.TRAIN), split="train", transforms=augment)
    else:
        raise NotImplementedError
 
    if cfg.DATASET.VAL == "cityscapes":
        val_set = CityscapesDataset(cfg, root=get_root_dir(cfg.DATASET.VAL), split="val", transforms=val_augment)
    else:
        raise NotImplementedError

    if cfg.DATASET.TEST == "cityscapes":
        test_set = CityscapesDataset(cfg, root=get_root_dir(cfg.DATASET.TEST), split="test", transforms=test_augment)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set

def make_data_loader(cfg, **kwargs):
    train_set, val_set, test_set = make_datasets(cfg)

    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() + worker_id) % 2**32 
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def custom_collate(original_batch):
        imgs = [] 
        targets = []
        names = []
        for item in original_batch:
            imgs.append(item.image)
            targets.append(item.label)
            names.append(item.image_name)
        imgs = torch.stack(imgs, dim=0)
        targets = torch.stack(targets, dim=0)
        return [imgs, targets, names]

    train_loader = DataLoader(train_set, batch_size=cfg.INPUT.BATCH_SIZE, drop_last=True, shuffle=True, worker_init_fn=seed_worker, collate_fn=custom_collate, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=custom_collate, **kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, collate_fn=custom_collate, **kwargs)

    return train_loader, val_loader, test_loader
