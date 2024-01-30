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
import yaml
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

import transforms

OmegaConf.register_new_resolver("sum", sum)
OmegaConf.register_new_resolver("len", len)


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

    #with open(osp.join(cfg.output_dir, 'full_config.yaml'), 'w') as f:
    #    yaml.dump(OmegaConf.to_yaml(cfg), f)

    trainer = instantiate(cfg.trainer)

    if isinstance(trainer.logger, WandbLogger):
        # save config to wandb
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        # wandb.config = cfg_resolved
        # trainer.logger.experiment.config.update(cfg_resolved)
        wandb.config.update(cfg_resolved)

        #wandb.savefile(osp.join(cfg.output_dir, 'full_config.yaml'))

    #if cfg.device.accelerator == 'gpu' and torch.cuda.is_available():
    #    print('Use GPU')
    #    # all newly created tensors go to GPU
    #    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    utils.seed_all(cfg.seed + cfg.get('job_id', 0))

    model = instantiate(cfg.model)#, n_env=len(cfg.datasource.env_vars))
    
    if 'model_checkpoint' in cfg and isinstance(trainer.logger, WandbLogger):
        # load model checkpoint
        # model_checkpoint has form 'user/project/model-runID:version' where version is vX, latest or best
        artifact_dir = trainer.logger.download_artifact(cfg.model_checkpoint, artifact_type='model')
        model_path = osp.join(artifact_dir, 'model.ckpt')
        #model.load_state_dict(torch.load(model_path))

        model = eval(cfg.model._target_).load_from_checkpoint(model_path)

    
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
    if 'predict' in cfg.task.task_name:
        prediction(trainer, model, cfg)

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()
        # save config to wandb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    

def load_training_data(cfg):

    transform = get_transform(cfg)
    data = dataloader.load_dataset(cfg, cfg.output_dir, training=True, 
                                   transform=transform)[0]
    
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

    # TODO: use a consecutive chunk as validation data instead of random sequences

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

    # save model
    #model_path = osp.join(cfg.output_dir, 'models')
    #os.makedirs(model_path, exist_ok=True)
    #torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    #if isinstance(trainer.logger, WandbLogger):
    #    # save as artifact for version control
    #    print('add model artifact')
    #    artifact = wandb.Artifact(f'model', type='models')
    #    artifact.add_dir(model_path)
    #    wandb.run.log_artifact(artifact)


def testing(trainer, model, cfg: DictConfig, ext=''):
    """
    Test neural network model on unseen test data.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which test results are written to
    """

    # cfg.datasource.bird_scale = float(model_cfg['datasource']['bird_scale'])

    # load test data
    transform = get_transform(cfg)
    # test_data, input_col, context, seq_len = dataloader.load_dataset(cfg, cfg.output_dir, training=False, transform=transform)
    test_data, context, seq_len = dataloader.load_dataset(cfg, cfg.output_dir, training=False, transform=transform)
    test_data = test_data[0]

    test_loader = instantiate(cfg.dataloader, test_data, batch_size=1, shuffle=False)

    model.horizon = cfg.model.test_horizon
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
    test_data, context, seq_len = dataloader.load_dataset(cfg, cfg.output_dir, training=False, transform=transform)
    test_data = test_data[0]

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
