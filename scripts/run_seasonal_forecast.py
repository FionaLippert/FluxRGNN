from fluxrgnn import dataloader, utils
from fluxrgnn.models import *
import torch
from torch.utils.data import random_split, Subset
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb
import pickle
import os.path as osp
import os
import numpy as np
#import ruamel.yaml
import yaml
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from evaluate_models import summarize_performance
import transforms

def merge_lists(*lists):
    merged = []
    for l in lists:
        merged += l
    return merged

OmegaConf.register_new_resolver("sum", sum)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("merge", merge_lists)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """
    Setup and evaluate seasonality forecasting model.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which all outputs are written to
    :param log: log file
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(osp.join(cfg.output_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(OmegaConf.to_yaml(cfg), f)

    trainer = instantiate(cfg.trainer)

    if isinstance(cfg.trainer.logger, WandbLogger):
        # save config to wandb
        print('use wandb for logging')
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        # wandb.config = cfg_resolved
        wandb.config.update(cfg_resolved)

        wandb.savefile(osp.join(cfg.output_dir, 'full_config.yaml'))


        # trainer.logger.experiment.config.update(cfg_resolved)

    utils.seed_all(cfg.seed + cfg.get('job_id', 0))

    model = instantiate(cfg.model)

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print('------------------------------------------------------')


    if 'train' in cfg.task.task_name:
        training(trainer, model, cfg)
    if 'eval' in cfg.task.task_name:
        testing(trainer, model, cfg)

    if isinstance(cfg.trainer.logger, WandbLogger):
        wandb.finish()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transform(cfg):

    transform_list = [instantiate(t) for t in cfg.model.transforms]
    transform = T.Compose(transform_list)

    return transform
    

def load_training_data(cfg):

    transform = get_transform(cfg)
    #data = dataloader.load_seasonal_dataset(cfg, cfg.output_dir, split='train',
    #                               transform=transform)
    
    data = dataloader.load_dataset(cfg, cfg.output_dir, split='train', transform=transform)[0]
    data = torch.utils.data.ConcatDataset(data)
    train_loader = instantiate(cfg.dataloader, data, batch_size=1)

    return train_loader


def training(trainer, model, cfg: DictConfig):
    """
    Extract seasonal patterns from training data.

    :param trainer: pytorch_lightning Trainer object
    :param model: pytorch_lightning model object
    :param cfg: DictConfig specifying model, data and training details
    """

    dl_train = load_training_data(cfg)
    if trainer.accelerator == 'gpu' and torch.cuda.is_available():
        model.cuda()

    max_tidx = 31 * 12 * 24
    seasonal_patterns_x = None # TODO: initialize all zeros and then index according to batch tidx
    counts_x = None

    seasonal_patterns_uv = None
    counts_uv = None

    for nidx, data in enumerate(dl_train):
        data = data.to(model.device)

        tidx = data['cell'].tidx

        if seasonal_patterns_x is None:
            seasonal_patterns_x = torch.zeros(data['radar'].num_nodes, max_tidx, device=model.device)
            seasonal_patterns_uv = torch.zeros(data['radar'].num_nodes, 2, max_tidx, device=model.device)
            counts_x = torch.zeros(data['radar'].num_nodes, max_tidx, device=model.device)
            counts_uv = torch.zeros(data['radar'].num_nodes, max_tidx, device=model.device)

        mask = torch.logical_not(data['radar'].missing_x)
        seasonal_patterns_x[:, tidx] += data['radar'].x * mask
        counts_x[:, tidx] += mask

        mask = torch.logical_not(data['radar'].missing_bird_uv)
        seasonal_patterns_uv[:, :, tidx] += data['radar'].bird_uv * mask.unsqueeze(1)
        counts_uv[:, tidx] += mask

    
    seasonal_patterns_x = seasonal_patterns_x / (counts_x + 1e-8)
    print('x size:', seasonal_patterns_x.size())
    model.seasonal_patterns['x'] = seasonal_patterns_x

    seasonal_patterns_uv = seasonal_patterns_uv / (counts_uv.unsqueeze(1) + 1e-8)
    print('uv size:', seasonal_patterns_uv)
    print((counts_uv > 0).sum())
    print((counts_x > 0).sum())
    model.seasonal_patterns['bird_uv'] = seasonal_patterns_uv

    # save model
    model_path = osp.join(cfg.output_dir, 'models')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    if isinstance(cfg.trainer.logger, WandbLogger):
        # save as artifact for version control
        artifact = wandb.Artifact(f'model', type='models')
        artifact.add_dir(model_path)
        wandb.run.log_artifact(artifact)


def testing(trainer, model, cfg: DictConfig, ext=''):
    """
    Test seasonal forecasting model on unseen test data.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which test results are written to
    """

    # load test data
    transform = get_transform(cfg)
    test_data = dataloader.load_dataset(cfg, cfg.output_dir, split='test', transform=transform)[0]
    test_data = torch.utils.data.ConcatDataset(test_data)

    test_loader = instantiate(cfg.dataloader, test_data, batch_size=1, shuffle=False)

    model.horizon = cfg.model.test_horizon
    model.config['ignore_day'] = True

    #model.seasonal_patterns.to(model.device)
    trainer.test(model, test_loader)

    eval_path = osp.join(cfg.output_dir, 'evaluation')

    summarize_performance(model.test_results, cfg, var='x', groupby=['observed'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'radar'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'night'], path=eval_path)

    if 'bird_uv' in model.test_vars:
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'bird_bin'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'radar'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'night'], path=eval_path)

    if cfg.task.get('store_test_results', True):
        print('save evaluation artifact')
        # utils.dump_outputs(model.test_metrics, eval_path)
        utils.dump_outputs(model.test_results, eval_path)
        artifact = wandb.Artifact(f'evaluation-{trainer.logger.version}', type='evaluation')
        artifact.add_dir(eval_path)
        wandb.run.log_artifact(artifact)

    if cfg.get('save_prediction', False):
        results = trainer.predict(model, test_loader, return_predictions=True)

        pred_path = osp.join(cfg.output_dir, 'prediction')
        utils.dump_outputs(results, pred_path)

        if isinstance(trainer.logger, WandbLogger):
            print('add prediction artifact')
            # save as artifact for version control
            artifact = wandb.Artifact(f'prediction-{trainer.logger.version}', type='prediction')
            artifact.add_dir(pred_path)
            wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    run()
