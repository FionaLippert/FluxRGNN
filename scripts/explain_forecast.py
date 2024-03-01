import fluxrgnn
from fluxrgnn import dataloader, utils, explainer
from fluxrgnn.models import *
import torch
from torch.utils.data import random_split, Subset
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
from pytorch_lightning.loggers import WandbLogger

# trade precision for performance
torch.set_float32_matmul_precision('medium')


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """
    Explain the prediction of a forecast model

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which all outputs are written to
    :param log: log file
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f'output_dir = {cfg.output_dir}')

    utils.seed_all(cfg.seed + cfg.get('job_id', 0))

    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)

    if (cfg.model.load_states_from is not None) and isinstance(trainer.logger, WandbLogger):
        # load model checkpoint
        # model_checkpoint has form 'user/project/model-runID:version' where version is vX, latest or best
        artifact_dir = trainer.logger.download_artifact(cfg.model.load_states_from, artifact_type='model')
        model_path = osp.join(artifact_dir, 'model.ckpt')

        if torch.cuda.is_available():
            model = eval(cfg.model._target_).load_from_checkpoint(model_path, map_location=torch.device('cuda'))
        else:
            model = eval(cfg.model._target_).load_from_checkpoint(model_path, map_location=torch.device('cpu'))

    explain(trainer, model, cfg)

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


def get_transform(cfg):
    transform_list = [instantiate(t) for t in cfg.model.transforms]
    transform = T.Compose(transform_list)

    return transform


def load_background_data(cfg, feature_names, reduction='sampling', n_samples=100):
    transform = get_transform(cfg)
    data = dataloader.load_dataset(cfg, cfg.output_dir, training=True, transform=transform)[0]

    data = torch.utils.data.ConcatDataset(data)
    background = explainer.construct_background(data, feature_names, reduction=reduction, n_samples=n_samples)

    return background



def explain(trainer, model, cfg: DictConfig):
    """
    Explain forecast
    """

    # load test data
    transform = get_transform(cfg)
    test_data, context, seq_len = dataloader.load_dataset(cfg, cfg.output_dir, training=False, transform=transform)
    test_data = test_data[0]

    model.horizon = cfg.model.test_horizon
    # model.store_fluxes = cfg.model.store_fluxes

    feature_names = list(cfg.model.env_vars.keys()) #[:4]
    
    idx = 0
    input_graph = test_data[idx]
    background = load_background_data(cfg, feature_names, reduction='mean', n_samples=100)

    expl = explainer.ForecastExplainer(model, background, feature_names)
    explanation = expl.explain(input_graph, n_samples=100)

    shap_values = explanation['shap_values']
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    print(shap_values.shape)
    print(shap_values.sum(-1))

    expl_path = osp.join(cfg.output_dir, 'explanation')
    utils.dump_outputs(explanation, expl_path)

    if isinstance(trainer.logger, WandbLogger):
        artifact = wandb.Artifact(f'explanation-{trainer.logger.version}', type='explanation')
        artifact.add_dir(expl_path)
        wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    run()
