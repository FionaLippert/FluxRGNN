import fluxrgnn
from fluxrgnn import dataloader, utils, transforms
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
import yaml
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from evaluate_models import summarize_performance

#import transforms

def merge_lists(*lists):
    merged = []
    for l in lists:
        merged += l
    return merged

OmegaConf.register_new_resolver("sum", sum)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("merge", merge_lists)

OmegaConf.register_new_resolver("eval", eval)

# map model name to implementation
MODEL_MAPPING = {'LocalMLP': LocalMLP,
                 'LocalLSTM': LocalLSTM,
                 'FluxRGNN': FluxRGNN}

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# trade precision for performance
torch.set_float32_matmul_precision('medium')


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """
    Run training and/or testing for neural network model.

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
        wandb.config.update(cfg_resolved)

    utils.seed_all(cfg.seed + cfg.get('job_id', 0))

    model = instantiate(cfg.model)
    
    if (cfg.model.load_states_from is not None) and isinstance(trainer.logger, WandbLogger):
        # load model checkpoint
        # model_checkpoint has form 'user/project/model-runID:version' where version is vX, latest or best
        artifact_dir = trainer.logger.download_artifact(cfg.model.load_states_from, artifact_type='model')
        model_path = osp.join(artifact_dir, 'model.ckpt')

        if torch.cuda.is_available():
            model = eval(cfg.model._target_).load_from_checkpoint(model_path)
        else:
            model = eval(cfg.model._target_).load_from_checkpoint(model_path, map_location=torch.device('cpu'))

    
    if 'train' in cfg.task.task_name:
        training(trainer, model, cfg)

    if 'eval' in cfg.task.task_name:
        testing(trainer, model, cfg)

    if 'predict' in cfg.task.task_name:
        prediction(trainer, model, cfg)

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transform(cfg):

    transform_list = [instantiate(t) for t in cfg.model.transforms]
    transform = T.Compose(transform_list)

    return transform
    

def load_training_data(cfg):

    transform = get_transform(cfg)
    # data = dataloader.load_dataset(cfg, cfg.output_dir, training=True,
    #                                transform=transform)[0]
    #
    # data = torch.utils.data.ConcatDataset(data)
    # n_data = len(data)
    #
    # # split data into training and validation set
    # n_val = max(1, int(cfg.datasource.val_train_split * n_data))
    # n_train = n_data - n_val
    #
    # if cfg.verbose:
    #     print('------------------------------------------------------')
    #     print('-------------------- data sets -----------------------')
    #     print(f'total number of sequences = {n_data}')
    #     print(f'number of training sequences = {n_train}')
    #     print(f'number of validation sequences = {n_val}')
    #
    # # TODO: use a consecutive chunk as validation data instead of random sequences?
    #
    # train_data, val_data = random_split(data, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed))

    train_data = dataloader.load_dataset(cfg, cfg.output_dir, split='train', transform=transform)[0]
    train_data = torch.utils.data.ConcatDataset(train_data)

    val_data = dataloader.load_dataset(cfg, cfg.output_dir, split='val', transform=transform)[0]
    val_data = torch.utils.data.ConcatDataset(val_data)

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

    trainer.fit(model, dl_train, dl_val)


def testing(trainer, model, cfg: DictConfig, ext=''):
    """
    Test neural network model on unseen test data.

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
    trainer.test(model, test_loader)

    eval_path = osp.join(cfg.output_dir, 'evaluation')
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 150, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 50, 250, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'bird_bin'], bins=[1, 50, 200, 1500], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'radar'], path=eval_path)
    summarize_performance(model.test_results, cfg, var='x', groupby=['observed', 'night'], path=eval_path)

    if 'bird_uv' in model.test_vars:
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 50, 250, 1500])
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 150, 1500])
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 1500])
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'radar'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='bird_uv', groupby=['observed', 'night'], path=eval_path)
    
        def uv_to_speed(uv):
            # assume uv to have shape [sequences, radars, 2, horizon]
            return torch.linalg.vector_norm(uv, dim=-2).unsqueeze(-2)

        def uv_to_direction(uv):
            # assume uv to have shape [sequences, radars, 2, horizon]
            angles = (torch.rad2deg(torch.arctan2(uv[..., 0, :], uv[..., 1, :])) + 360) % 360

            return angles.unsqueeze(-2)

        # compute speed and direction
        model.test_results['test/predictions/direction'] = uv_to_direction(model.test_results['test/predictions/bird_uv'])
        model.test_results['test/predictions/speed'] = uv_to_speed(model.test_results['test/predictions/bird_uv'])
        model.test_results['test/measurements/direction'] = uv_to_direction(model.test_results['test/measurements/bird_uv'])
        model.test_results['test/measurements/speed'] = uv_to_speed(model.test_results['test/measurements/bird_uv'])
        model.test_results['test/masks/direction'] = model.test_results['test/masks/bird_uv']
        model.test_results['test/masks/speed'] = model.test_results['test/masks/bird_uv']


        summarize_performance(model.test_results, cfg, var='direction', groupby=['observed'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='direction', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 50, 250, 1500])
        summarize_performance(model.test_results, cfg, var='direction', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 150, 1500])
        summarize_performance(model.test_results, cfg, var='direction', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 1500])
        summarize_performance(model.test_results, cfg, var='direction', groupby=['observed', 'radar'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='direction', groupby=['observed', 'night'], path=eval_path)


        summarize_performance(model.test_results, cfg, var='speed', groupby=['observed'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='speed', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 50, 250, 1500])
        summarize_performance(model.test_results, cfg, var='speed', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 150, 1500])
        summarize_performance(model.test_results, cfg, var='speed', groupby=['observed', 'bird_bin'], path=eval_path, bins=[1, 1500])
        summarize_performance(model.test_results, cfg, var='speed', groupby=['observed', 'radar'], path=eval_path)
        summarize_performance(model.test_results, cfg, var='speed', groupby=['observed', 'night'], path=eval_path)
    
    if cfg.task.get('store_test_results', True):

        # utils.dump_outputs(model.test_metrics, eval_path)
        utils.dump_outputs(model.test_results, eval_path)

    if isinstance(trainer.logger, WandbLogger):
        print('add evaluation artifact')
        artifact = wandb.Artifact(f'evaluation-{trainer.logger.version}', type='evaluation')
        artifact.add_dir(eval_path)
        wandb.run.log_artifact(artifact)

    if cfg.get('save_prediction', False):
        trainer.predict(model, test_loader, return_predictions=True)

        pred_path = osp.join(cfg.output_dir, 'prediction')
        utils.dump_outputs(model.predict_results, pred_path)

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
    model.store_fluxes = cfg.model.store_fluxes
    
    if cfg.task.predict_cells:
        model.observation_model = None
    
    trainer.predict(model, test_loader)

    pred_path = osp.join(cfg.output_dir, 'prediction')
    utils.dump_outputs(model.predict_results, pred_path)

    if isinstance(trainer.logger, WandbLogger):
        artifact = wandb.Artifact(f'prediction-{trainer.logger.version}', type='prediction')
        artifact.add_dir(pred_path)
        wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    run()
