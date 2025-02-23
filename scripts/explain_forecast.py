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
from timeit import default_timer as timer

import logging
logging.getLogger('shap').setLevel(logging.WARNING) # turns off the "shap INFO" logs

# trade precision for performance
torch.set_float32_matmul_precision('medium')

def merge_lists(*lists):
    merged = []
    for l in lists:
        merged += l
    return merged

OmegaConf.register_new_resolver("sum", sum)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("merge", merge_lists)

OmegaConf.register_new_resolver("eval", eval)


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

    if (cfg.model.load_states_from is not None):
        # try to load from disk (path should point to .ckpt file containing model states)
        if osp.isfile(cfg.model.load_states_from):
            model_path = cfg.model.load_states_from
            print('Loading model state from disk...')
        elif isinstance(trainer.logger, WandbLogger):
            # load model checkpoint from W&B
            # path should have form 'user/project/model-runID:version' where version is vX, latest or best
            artifact_dir = trainer.logger.download_artifact(cfg.model.load_states_from, artifact_type='model')
            model_path = osp.join(artifact_dir, 'model.ckpt')
            print('Loading model state from W&B...')
        else:
            raise Exception('Model states could not be loaded from disk or W&B')

        if torch.cuda.is_available():
            model = eval(cfg.model._target_).load_from_checkpoint(model_path)
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


def load_background_data(cfg, feature_names, seqID_min, seqID_max, reduction='sampling', n_samples=100, normalization=None):
    transform = get_transform(cfg)

    data = dataloader.load_dataset(cfg, cfg.output_dir, split='train', transform=transform, seqID_min=seqID_min, seqID_max=seqID_max, normalization=normalization)[0]
    data = torch.utils.data.ConcatDataset(data)

    background = explainer.construct_background(data, feature_names, reduction=reduction, n_samples=n_samples)

    return background



def explain(trainer, model, cfg: DictConfig):
    """
    Explain forecast
    """

    # load test data
    transform = get_transform(cfg)

    # define time period to consider
    seqID_start = cfg.task.get('seqID_start', 0)
    seqID_end = cfg.task.get('seqID_end', -1)

    start = timer()

    print(cfg.datasource.get('load_normalization_from', ''))
    if osp.isfile(cfg.datasource.get('load_normalization_from', '')):
        print('load normalization from disk')
        with open(cfg.datasource.get('load_normalization_from'), 'rb') as f:
            normalization = pickle.load(f)
    else:
        normalization = None

    test_data, normalization = dataloader.load_dataset(cfg, cfg.output_dir, split='test', transform=transform, seqID_min=seqID_start, seqID_max=seqID_end, normalization=normalization)
    
    n_years = len(test_data)

    print(f'consider {n_years} years')
    test_data = torch.utils.data.ConcatDataset(test_data)

    if seqID_end < 0:
        seqID_end = len(test_data) - 1
    seqID_list = torch.concat([torch.arange(seqID_start, seqID_end + 1) for y in range(n_years)]) 
    
    end = timer()
    print(f'data loading took {end - start} seconds')

    feature_names = OmegaConf.to_object(cfg.task)['feature_names']

    n_bg_samples = cfg.task.get('n_bg_samples', 10)
    reduction = cfg.task.get('bg_reduction', 'all')
    n_shap_samples = cfg.task.get('n_shap_samples', 1000)
    
    
    n_nights = len(test_data)

    sample_idx_start = cfg.task.get('sample_idx', 0)
    n_seq_samples = cfg.task.get('n_seq_samples', 1)
    
    if n_seq_samples == 'all':
        sample_idx_start = 0
        n_seq_samples = n_nights

    sample_idx = range(sample_idx_start, sample_idx_start + n_seq_samples)
    
    if cfg.task.random_sample:
        # randomly sample a sequence
        rng = np.random.default_rng(cfg.seed)
        random_nights = rng.permutation(n_nights)
        idx_list = random_nights[sample_idx]
    else:
        # select sequences sequentially
        idx_list = np.arange(n_nights)[sample_idx]

    model.horizon = cfg.model.horizon

    for idx in idx_list:
        # ID in overall dataset
        seqID = seqID_list[idx]

        print(f'Explain idx {idx}, night {seqID}')
        print(f'Considered features: {feature_names}')

        input_graph = test_data[idx]

        print(f'tidx range = {input_graph["cell"].tidx.min()} - {input_graph["cell"].tidx.max()}')
        seqID_min = max(seqID_start, seqID - cfg.task.get('bg_window', 0))
        seqID_max = min(seqID + cfg.task.get('bg_window', 0), seqID_end)

        print(f'Load background sequences: {seqID_min} - {seqID_max}')

        start = timer()
        background = load_background_data(cfg, feature_names, seqID_min, seqID_max, reduction=reduction, n_samples=n_bg_samples, normalization=normalization)
        end = timer()
        print(f'Background loading took {end - start} seconds')


        start = timer()
        expl = explainer.ForecastExplainer(model, background, feature_names, explain_processes=cfg.task.explain_processes)
        explanation = expl.explain(input_graph, n_samples=n_shap_samples)
        end = timer()
        print(f'Explanation took {end - start} seconds')

        explanation['local_night'] = input_graph[expl.node_store]['local_night'][:, expl.t0 + expl.context: expl.t0 + expl.context + expl.horizon]

        explanation['background'] = background


        fidx = 0
        for name in feature_names:
            for sub_name in name.split('+'):
                values = input_graph[expl.node_store][sub_name][..., expl.t0: expl.t0 + expl.context + expl.horizon]

                explanation[sub_name] = transformed2raw(values, normalization, sub_name)

                explanation['background'][:, fidx] = transformed2raw(
                        explanation['background'][:, fidx],
                        normalization, sub_name
                )
                fidx += 1

        expl_path = osp.join(cfg.output_dir, f'explanation_{idx}')
        utils.dump_outputs(explanation, expl_path)

        if isinstance(trainer.logger, WandbLogger):
            artifact = wandb.Artifact(f'explanation-{idx}-{trainer.logger.version}', type='explanation')
            artifact.add_dir(expl_path)
            wandb.run.log_artifact(artifact)

def transformed2raw(values, normalization, var_name):
    # reverse normalization
    values = values + 1
    values = values / 2.0
    values = values * (normalization.max(var_name) - normalization.min(var_name))
    values = values + normalization.min(var_name)

    return values



if __name__ == "__main__":
    run()
