import fluxrgnn
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
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
import wandb
import pickle
import os.path as osp
import os
import numpy as np
#import ruamel.yaml
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import xgboost

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

OmegaConf.register_new_resolver("eval", eval)

# trade precision for performance
torch.set_float32_matmul_precision('medium')


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """
    Run training and/or testing for XGBoost model.

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which all outputs are written to
    :param log: log file
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f'output_dir = {cfg.output_dir}')

    trainer = instantiate(cfg.trainer)

    if isinstance(trainer.logger, WandbLogger):
        # save config to wandb
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        wandb.config = cfg_resolved
        # trainer.logger.experiment.config.update(cfg_resolved)
        print(trainer.logger.version)

    utils.seed_all(cfg.seed + cfg.get('job_id', 0))

    cfg.model.xgboost.random_state = cfg.seed + cfg.get('job_id', 0)

    model = instantiate(cfg.model)
    print(model)
    
    if 'train' in cfg.task.task_name:
        training(model, cfg)
    if 'eval' in cfg.task.task_name:
        testing(trainer, model, cfg)
    if 'predict' in cfg.task.task_name:
        prediction(trainer, model, cfg)

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


def get_transform(cfg):

    # transform_list = []
    #
    # if cfg.model.use_log_transform:
    #     transform_list.extend([transforms.LogTransform('x', offset=cfg.model.log_offset),
    #                            transforms.LogTransform('y', offset=cfg.model.log_offset)])
    #
    # transform_list.extend([transforms.Rescaling('x', factor=cfg.model.scale),
    #                        transforms.Rescaling('y', factor=cfg.model.scale)])

    transform_list = [instantiate(t) for t in cfg.model.transforms]
    transform = T.Compose(transform_list)

    return transform
    

def load_training_data(cfg, split='train'):

    transform = get_transform(cfg)

    data = dataloader.load_xgboost_dataset(cfg, cfg.output_dir, transform=transform, split=split)
    data = torch.utils.data.ConcatDataset(data)

    dynamic_cell_features = cfg.model.get('dynamic_cell_features', {})
    static_cell_features = cfg.model.get('static_cell_features', {})

    X = []
    y = []
    print(f'number of sequences in training dataset: {len(data)}')
    for d in data:

        cell_data = d['cell']
        radar_data = d['radar']
        T = cell_data.tidx.size(-1)

        print(radar_data['local_night'].size())

        print(f'coord size = {cell_data.coords.size()}')
        print(f'num nodes = {cell_data.num_nodes}')

        assert cell_data.num_nodes == radar_data.num_nodes

        # dynamic features for all time points
        dynamic_features = [cell_data.get(feature).reshape(cell_data.num_nodes, -1, T) for
                                      feature in dynamic_cell_features]

        # static graph features
        node_features = [cell_data.get(feature).reshape(cell_data.num_nodes, -1, 1).repeat(1, 1, T) for
                                   feature in static_cell_features]

        # combined features
        inputs = torch.cat(node_features + dynamic_features, dim=1) #.detach().numpy()
        inputs = inputs.permute(0, 2, 1) # [nodes, time, features]

        # masks
        missing = radar_data.missing_x
        if split == 'train':
            radar_mask = radar_data.train_mask
        else:
            radar_mask = radar_data.test_mask

        if cfg.model.get('force_zeros', False):
            local_mask = torch.logical_and(radar_data.local_night, torch.logical_not(missing))
        else:
            local_mask = torch.logical_not(missing)

        local_mask = local_mask[radar_mask].reshape(-1)

        # ground truth
        gt = radar_data.x[radar_mask].reshape(-1)

        # features
        inputs = inputs[radar_mask].reshape(radar_mask.sum() * T, -1)

        X.append(inputs[local_mask])
        y.append(gt[local_mask])

    X = torch.cat(X, dim=0).detach().numpy()
    y = torch.cat(y, dim=0).detach().numpy()

    print(f'mean density = {y.mean()}')
    print(f'max density = {y.max()}')
    print(f'min density = {y.min()}')
    print(f'95% density = {np.percentile(y, 95)}')
    print(f'90% density = {np.percentile(y, 90)}')
    print(f'99% density = {np.percentile(y, 99)}')

    return X, y

def training(model, cfg: DictConfig):
    """
    Run training of a neural network model.

    :param model: XGBoost model
    :param cfg: DictConfig specifying model, data and training details
    """

    X_train, y_train = load_training_data(cfg, split='train')
    X_val, y_val = load_training_data(cfg, split='val')

    # early_stop = xgboost.callback.EarlyStopping(tolerance=cfg.model.xgboost.tol)

    model.fit_xgboost(X_train, y_train, eval_set=[(X_val, y_val)], **cfg.model.train_settings)


def testing(trainer, model, cfg: DictConfig, ext=''):
    """
    Test XGBoost model on unseen test data.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which test results are written to
    """

    # load test data
    transform = get_transform(cfg)
    print('######### testing #########')

    test_data = dataloader.load_dataset(cfg, cfg.output_dir, split='test', transform=transform)[0]
    test_data = torch.utils.data.ConcatDataset(test_data)

    test_loader = instantiate(cfg.dataloader, test_data, batch_size=1, shuffle=False)

    model.horizon = cfg.model.test_horizon
    model.config['ignore_day'] = True
    trainer.test(model, test_loader)

    eval_path = osp.join(cfg.output_dir, 'evaluation')

    summarize_performance(model.test_results, cfg, var='x', groupby=['observed'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 150, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 50, 250, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 50, 200, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'radar'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'night'], path=eval_path)

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


def prediction(trainer, model, cfg: DictConfig, ext=''):
    """
    Run neural network model on unseen test data.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which test results are written to
    """

    # load test data
    transform = get_transform(cfg)
    test_data = dataloader.load_dataset(cfg, cfg.output_dir, split='test', transform=transform)[0]
    test_data = torch.utils.data.ConcatDataset(test_data)

    test_loader = instantiate(cfg.dataloader, test_data, batch_size=1, shuffle=False)

    model.horizon = cfg.model.test_horizon
    trainer.predict(model, test_loader)

    pred_path = osp.join(cfg.output_dir, 'prediction')
    utils.dump_outputs(model.predict_results, pred_path)

    if isinstance(trainer.logger, WandbLogger):
        artifact = wandb.Artifact(f'prediction-{trainer.logger.version}', type='prediction')
        artifact.add_dir(pred_path)
        wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    run()
