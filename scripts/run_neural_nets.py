from fluxrgnn import dataloader, utils
from fluxrgnn.models import *
import torch
from torch.utils.data import random_split, Subset
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import to_dense_adj
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
import pickle
import os.path as osp
import os
import numpy as np
#import ruamel.yaml
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

# map model name to implementation
MODEL_MAPPING = {'LocalMLP': LocalMLP,
                 'LocalLSTM': LocalLSTM,
                 'FluxRGNN': FluxRGNN}

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """
    Run training and/or testing for neural network model.

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which all outputs are written to
    :param log: log file
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    trainer = instantiate(cfg.trainer)

    if isinstance(cfg.trainer.logger, WandbLogger):
        # save config to wandb
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        wandb.config = cfg_resolved
        # trainer.logger.experiment.config.update(cfg_resolved)

    utils.seed_all(cfg.seed + cfg.get('job_id', 0))

    model = instantiate(cfg.model, n_env=len(cfg.datasource.env_vars))

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print('------------------------------------------------------')


    if 'train' in cfg.task.name:
        training(trainer, model, cfg)
    if 'eval' in cfg.task.name:
        # if hasattr(cfg, 'importance_sampling'):
        #     cfg.importance_sampling = False

        cfg['fixed_t0'] = True
        testing(trainer, model, cfg, ext='_fixedT0')
        # cfg['fixed_t0'] = False
        # testing(trainer, model, cfg, output_dir, log)

        # if cfg.get('test_train_data', False):
        #     # evaluate performance on training data
        #     training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
        #     cfg.model.test_horizon = cfg.model.horizon
        #     for y in training_years:
        #         cfg.datasource.test_year = y
        #         testing(trainer, model, cfg, output_dir, log, ext=f'_training_year_{y}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_training_data(cfg):
    data = dataloader.load_dataset(cfg, cfg.output_dir, training=True)[0]
    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    # split data into training and validation set
    n_val = max(1, int(cfg.datasource.val_train_split * n_data))
    n_train = n_data - n_val

    if cfg.verbose:
        print('------------------------------------------------------')
        print('-------------------- data sets -----------------------')
        print(f'total number of sequences = {n_data}')
        print(f'number of training sequences = {n_train}')
        print(f'number of validation sequences = {n_val}')

    train_data, val_data = random_split(data, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = instantiate(cfg.dataloader, train_data)
    val_loader = instantiate(cfg.dataloader, val_data, batch_size=1)

    return train_loader, val_loader

def training(trainer, model, cfg: DictConfig):
    """
    Run training of a neural network model.

    :param trainer: pytorch_lightning Trainer object
    :param model: pytorch_lightning model object
    :param cfg: DictConfig specifying model, data and training details
    """

    if cfg.debugging: torch.autograd.set_detect_anomaly(True)

    dl_train, dl_val = load_training_data(cfg)

    n_params = count_parameters(model)

    if cfg.verbose:
        print('initialized model')
        print(f'number of model parameters: {n_params}')
        print(f'environmental variables: {cfg.datasource.env_vars}')

    trainer.fit(model, dl_train, dl_val)


def testing(trainer, model, cfg: DictConfig, ext=''):
    """
    Test neural network model on unseen test data.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which test results are written to
    """

    # cfg.datasource.bird_scale = float(model_cfg['datasource']['bird_scale'])

    # load test data
    test_data, input_col, context, seq_len = dataloader.load_dataset(cfg, training=False)
    test_data = test_data[0]
    test_loader = instantiate(cfg.dataloader, test_data, batch_size=1, shuffle=False)

    trainer.test(model, test_loader)

    if cfg.get('save_prediction', False):
        results = trainer.predict(model, test_loader, return_predictions=True)

if __name__ == "__main__":
    run()