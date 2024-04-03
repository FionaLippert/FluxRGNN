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
        # if hasattr(cfg, 'importance_sampling'):
        #     cfg.importance_sampling = False

        # cfg['fixed_t0'] = True
        # testing(trainer, model, cfg, ext='_fixedT0')
        # cfg['fixed_t0'] = False
        testing(trainer, model, cfg)

        # if cfg.get('test_train_data', False):
        #     # evaluate performance on training data
        #     training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
        #     cfg.model.test_horizon = cfg.model.horizon
        #     for y in training_years:
        #         cfg.datasource.test_year = y
        #         testing(trainer, model, cfg, output_dir, log, ext=f'_training_year_{y}')

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
    data = dataloader.load_seasonal_dataset(cfg, cfg.output_dir, split='train',
                                   transform=transform)
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

    seasonal_patterns = []
    missing_patterns = []

    for nidx, data in enumerate(dl_train):
        data = data.to(model.device)

        seasonal_patterns.append(data.x)
        missing_patterns.append(data.missing)

    seasonal_patterns = torch.stack(seasonal_patterns, dim=0)  # shape [years, radars, timepoints]
    missing_patterns = torch.stack(missing_patterns, dim=0)  # shape [years, radars, timepoints]

    mask = torch.logical_not(missing_patterns)
    print(f'number of zeros = {(mask.sum(0) == 0).sum()}')
    #seasonal_patterns = (mask * seasonal_patterns).sum(0) / mask.sum(0)  # shape [radars, timepoints]
    seasonal_patterns = seasonal_patterns.mean(0)
    model.seasonal_patterns = seasonal_patterns

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
    # test_data = test_data[0]
    test_data = torch.utils.data.ConcatDataset(test_data)

    test_loader = instantiate(cfg.dataloader, test_data, batch_size=1, shuffle=False)

    model.horizon = cfg.model.test_horizon

    print(model.device, model.seasonal_patterns.device)
    model.seasonal_patterns.to(model.device)
    trainer.test(model, test_loader)

    eval_path = osp.join(cfg.output_dir, 'evaluation')
    utils.dump_outputs(model.test_metrics, eval_path)
    utils.dump_outputs(model.test_results, eval_path)

    if isinstance(trainer.logger, WandbLogger):
        print('add evaluation artifact')
        artifact = wandb.Artifact(f'evaluation-{trainer.logger.version}', type='evaluation')
        artifact.add_dir(eval_path)
        wandb.run.log_artifact(artifact)

    if cfg.get('save_prediction', False):
        results = trainer.predict(model, test_loader, return_predictions=True)

        # save results
        #os.makedirs(result_path, exist_ok=True)
        #with open(osp.join(result_path, 'results.pickle'), 'wb') as f:
        #    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        pred_path = osp.join(cfg.output_dir, 'prediction')
        utils.dump_outputs(results, pred_path)
        
        if isinstance(cfg.trainer.logger, WandbLogger):
            # save as artifact for version control
            artifact = wandb.Artifact(f'prediction', type='prediction')
            artifact.add_dir(pred_path)
            wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    run()
