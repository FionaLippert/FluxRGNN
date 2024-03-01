import numpy as np
from matplotlib import pyplot as plt
import os
import os.path as osp
import torch
import warnings
import pickle
import pandas as pd
#import ruamel.yaml
from omegaconf import OmegaConf
import pytorch_lightning as pl

# from src import dataloader

def seed_all(seed):
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

def val_test_split(dataloader, val_ratio):
    N = len(dataloader)
    n_val = int(N * val_ratio)
    val_idx = np.arange(n_val)
    val_loader = [list(dataloader)[i] for i in val_idx]
    test_loader = [list(dataloader)[i] for i in range(N) if i not in val_idx]

    return val_loader, test_loader

def dump_outputs(output_dict, dir):

    for k, v in output_dict.items():
        if isinstance(v, torch.Tensor):
            fp = osp.join(dir, f'{k}.pt')
            os.makedirs(osp.dirname(fp), exist_ok=True)
            torch.save(v.cpu(), fp)
        elif isinstance(v, np.ndarray):
            fp = osp.join(dir, f'{k}.np')
            os.makedirs(osp.dirname(fp), exist_ok=True)
            np.save(fp, v)
        else:
            fp = osp.join(dir, f'{k}.pickle')
            os.makedirs(osp.dirname(fp), exist_ok=True)
            with open(fp, 'wb') as f:
                pickle.dump(v, f)

def SMAPE(output, gt, mask):
    # compute symmetric mean absolute percentage error
    abs_diff = torch.abs(output - gt)
    abs_sum = torch.abs(output) + torch.abs(gt)
    ratios = (abs_diff * mask) / (abs_sum * mask + 1e-8)
    smape = ratios.sum(0) / mask.sum(0)

    return smape

def MAPE(output, gt, mask):
    # compute mean absolute percentage error
    abs_diff = torch.abs(output - gt)
    abs_gt = torch.abs(gt)
    ratios = (abs_diff * mask) / (abs_gt * mask + 1e-8)
    mape = ratios.sum(0) / mask.sum(0)

    return mape

def MAE(output, gt, mask):
    # compute mean absolute error

    abs_diff = torch.abs(output - gt)
    # mae = torch.sum(abs_diff * mask, dim=0) / torch.sum(mask, dim=0)
    mae = (abs_diff * mask).sum(0) / mask.sum(0)

    return mae

def MSE_numpy(output, gt, mask):

    diff = output - gt
    diff2 = np.square(diff)
    mse = np.sum(diff2 * mask) / np.sum(mask)
    return mse

def MSE(output, gt, mask, weights=None):

    diff = torch.abs(output - gt)
    diff2 = torch.square(diff)
    if weights is not None:
        diff2 = weights * diff2
    mse = torch.sum(diff2 * mask, dim=0) / torch.sum(mask, dim=0)
    return mse

def R2(output, gt, mask):

    mse = MSE(output, gt, mask)
    var = MSE(gt, torch.mean(gt, dim=0).unsqueeze(0), mask)

    r2 = 1 - (mse/var)

    return r2

def MSE_weighted(output, gt, mask, p=0.75):

    diff = torch.abs(output - gt)
    diff2 = torch.square(diff)
    weight = 1 + torch.pow(torch.abs(gt), p)
    mse = torch.sum(diff2 * weight * mask) / torch.sum(mask)
    return mse

def MSE_root_transformed(output, gt, mask, root=3):
    errors = (torch.pow(output.relu(), 1/root) - torch.pow(gt, 1/root))**2
    errors = errors[mask]
    mse = errors.mean()
    return mse

def plot_training_curves(training_curves, val_curves, dir, log=True):
    epochs = training_curves.shape[1]
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice.')
        train_line = ax.plot(range(1, epochs + 1), np.nanmean(training_curves, 0), label='training')
        ax.fill_between(range(1, epochs + 1), np.nanmean(training_curves, 0) - np.nanstd(training_curves, 0),
                        np.nanmean(training_curves, 0) + np.nanstd(training_curves, 0), alpha=0.2,
                        color=train_line[0].get_color())
        val_line = ax.plot(range(1, epochs + 1), np.nanmean(val_curves, 0), label='validation')
        ax.fill_between(range(1, epochs + 1), np.nanmean(val_curves, 0) - np.nanstd(val_curves, 0),
                        np.nanmean(val_curves, 0) + np.nanstd(val_curves, 0), alpha=0.2,
                        color=val_line[0].get_color())
    ax.set(xlabel='epoch', ylabel='MSE')
    if log: ax.set(yscale='log', xscale='log')
    plt.legend()
    fig.savefig(osp.join(dir, f'training_validation_curves_log={log}.png'), bbox_inches='tight')
    plt.close(fig)

def load_model_cfg(model_dir):
    # yaml = ruamel.yaml.YAML()
    fp = osp.join(model_dir, 'config.yaml')
    # with open(fp, 'r') as f:
    #     model_cfg = yaml.load(f)
    model_cfg = OmegaConf.load(fp)
    return model_cfg

def finalize_results(results, output_dir, ext=''):
    # create dataframe containing all results
    for k, v in results.items():
        if torch.is_tensor(v[0]):
            results[k] = torch.cat(v).numpy()
        else:
            results[k] = np.concatenate(v)

    results['residual_km2'] = results['gt_km2'] - results['prediction_km2']
    df = pd.DataFrame(results)
    df.to_csv(osp.join(output_dir, f'results{ext}.csv'))

