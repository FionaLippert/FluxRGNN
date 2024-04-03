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

    with torch.no_grad():
        explain(trainer, model, cfg)

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


def get_transform(cfg):
    transform_list = [instantiate(t) for t in cfg.model.transforms]
    transform = T.Compose(transform_list)

    return transform


def load_background_data(cfg, feature_names, reduction='sampling', n_samples=100):
    transform = get_transform(cfg)
    data = dataloader.load_dataset(cfg, cfg.output_dir, split='train', transform=transform)[0]
    data = torch.utils.data.ConcatDataset(data)

    background = explainer.construct_background(data, feature_names, reduction=reduction, n_samples=n_samples)

    return background



def explain(trainer, model, cfg: DictConfig):
    """
    Explain forecast
    """

    # load test data
    transform = get_transform(cfg)
    test_data, context, seq_len = dataloader.load_dataset(cfg, cfg.output_dir, split='test', transform=transform)
    # test_data = test_data[0]
    test_data = torch.utils.data.ConcatDataset(test_data)

    normalization = test_data.normalization

    model.horizon = cfg.model.test_horizon
    # model.store_fluxes = cfg.model.store_fluxes

    #feature_names = list(cfg.model.env_vars.keys()) #[:4]

    feature_names = list(cfg.task.get('feature_names', ['u10', 'v10', 't2m', 'cc', 'q', 'sp', 'tp']))
    print(feature_names, type(feature_names))

    feature_names = cfg.task.feature_names
    print(type(feature_names))

    feature_names = OmegaConf.to_object(cfg.task)['feature_names']

    n_bg_samples = cfg.task.get('n_bg_samples', 10)
    n_shap_samples = cfg.task.get('n_shap_samples', 1000)
    idx = cfg.task.seqID

    print(f'Explain sequence {idx}')
    print(f'Consider features {feature_names}')

    input_graph = test_data[idx]
    background = load_background_data(cfg, feature_names, reduction='sampling', n_samples=n_bg_samples)

    expl = explainer.ForecastExplainer(model, background, feature_names)
    explanation = expl.explain(input_graph, n_samples=n_shap_samples) #0)

    explanation['local_night'] = input_graph[expl.node_store]['local_night'][:, expl.t0 + expl.context: expl.t0 + expl.context + expl.horizon]

    for name in feature_names:
        values = input_graph[expl.node_store][name][..., expl.t0 + expl.context: expl.t0 + expl.context + expl.horizon]

        # reverse normalization
        if name in ['u', 'v']:
            uv_scale = max(normalization.absmax('u'), normalization.absmax('v'))
            explanation[name] = values * uv_scale

        elif name in ['u10', 'v10']:
            uv_scale = max(normalization.absmax('u10'), normalization.absmax('v10'))
            explanation[name] = values * uv_scale

        elif name in ['sshf']:
            explanation[name] = values * normalization.absmax('sshf')

        else:
            values = values + 1
            values = values / 2.0
            values = values * (normalization.max(name) - normalization.min(name))
            values = values + normalization.min(name)
            explanation[name] = values

    expl_path = osp.join(cfg.output_dir, f'explanation_{idx}')
    utils.dump_outputs(explanation, expl_path)

    if isinstance(trainer.logger, WandbLogger):
        artifact = wandb.Artifact(f'explanation-{trainer.logger.version}', type='explanation')
        artifact.add_dir(expl_path)
        wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    run()
