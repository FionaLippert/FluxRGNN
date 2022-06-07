from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import scipy.stats as stats
import os
import os.path as osp
import pandas as pd
import itertools as it
from yaml import Loader, load


@hydra.main(config_path="conf", config_name="config")
def evaluate(cfg: DictConfig):
    """
    Evaluation of predictive performance.

    Predictive performance is evaluated as a function of the forecasting horizon, and it is measured in terms of
    root mean square error, pearson correlation, and binary classification metrics.
    """

    experiments = cfg.get('experiment_type', 'final')
    fixed_t0 = cfg.get('fixed_t0', False)
    ext = '_fixedT0' if fixed_t0 else ''

    if experiments == 'ablations':
        models = {
            'FluxRGNN': ['final',
                         'final_without_encoder',
                         'final_without_boundary'],
            'LocalLSTM': ['final']
        }
    elif experiments == 'final':
        models = {
            'FluxRGNN': ['final'],
            'GAM': ['final'],
            'HA': ['final'],
            'GBT': ['final']
        }

    base_dir = cfg.device.root

    if cfg.datasource.name == 'abm':
        thresholds = [0.0019, 0.0207]
    else:
        thresholds = [0, 10, 20]

    rmse_per_hour = []
    mae_per_hour = []
    pcc_per_hour = []
    bin_per_hour = []

    rmse_per_night = []
    mae_per_night = []

    output_dir = osp.join(base_dir, 'results', cfg.datasource.name, f'performance_evaluation{ext}', experiments)
    os.makedirs(output_dir, exist_ok=True)


    for m, dirs in models.items():
        print(f'evaluate {m}')

        for d in dirs:
            result_dir = osp.join(base_dir, 'results', cfg.datasource.name, m, f'test_{cfg.datasource.test_year}', d)

            results, model_cfg = load_cv_results(result_dir, trials=cfg.task.repeats, ext=ext)

            df_prep = pd.read_csv(osp.join(base_dir, 'data', 'preprocessed',
                    f'{model_cfg["t_unit"]}_{model_cfg["model"]["edge_type"]}_ndummy={model_cfg["model"]["n_dummy_radars"]}',
                        cfg.datasource.name, cfg.season, str(cfg.datasource.test_year), 'dynamic_features.csv'))
            tidx2night = dict(zip(df_prep.tidx, df_prep.nightID))

            rmse_per_hour.append(compute_rmse(m, d, results, groupby=['horizon', 'trial'], threshold=0, km2=True))
            mae_per_hour.append(compute_mae(m, d, results, groupby=['horizon', 'trial'], threshold=0, km2=True))
            pcc_per_hour.append(compute_pcc(m, d, results, groupby=['horizon', 'trial'], threshold=0, km2=True))

            rmse_per_night.append(compute_rmse_per_night(m, d, results, tidx2night, groupby=['night_horizon', 'trial']))
            mae_per_night.append(compute_mae_per_night(m, d, results, tidx2night, groupby=['night_horizon', 'trial']))

            # compute binary classification measures
            for thr in thresholds:
                bin_per_hour.append(compute_bin(m, d, results, groupby=['horizon', 'trial'], threshold=thr, km2=True))

            # compute spatial correlation of residuals
            corr = compute_residual_corr(results, km2=True)

            corr.to_csv(osp.join(output_dir, f'spatial_corr_{m}_{d}.csv'))

    # output_dir = osp.join(base_dir, 'results', cfg.datasource.name, 'performance_evaluation', experiments)
    # os.makedirs(output_dir, exist_ok=True)

    rmse_per_hour = pd.concat(rmse_per_hour)
    rmse_per_hour.to_csv(osp.join(output_dir, f'rmse_per_hour.csv'))

    mae_per_hour = pd.concat(mae_per_hour)
    mae_per_hour.to_csv(osp.join(output_dir, f'mae_per_hour.csv'))

    pcc_per_hour = pd.concat(pcc_per_hour)
    pcc_per_hour.to_csv(osp.join(output_dir, f'pcc_per_hour.csv'))

    bin_per_hour = pd.concat(bin_per_hour)
    bin_per_hour.to_csv(osp.join(output_dir, f'bin_per_hour.csv'))

    rmse_per_night = pd.concat(rmse_per_night)
    rmse_per_night.to_csv(osp.join(output_dir, f'rmse_per_night.csv'))

    mae_per_night = pd.concat(mae_per_night)
    mae_per_night.to_csv(osp.join(output_dir, f'mae_per_night.csv'))


def load_cv_results(result_dir, ext='', trials=1):

    result_list = []
    for t in range(1, trials+1):
        print(f'trial = {t}')
        file = osp.join(result_dir, f'trial_{t}', f'results{ext}.csv')
        if osp.isfile(file):
            df = pd.read_csv(file)
            df['trial'] = t
            result_list.append(df)

            cfg_file = osp.join(result_dir, f'trial_{t}', 'config.yaml')
            with open(cfg_file) as f:
                cfg = load(f, Loader=Loader)

    results = pd.concat(result_list)

    return results, cfg


def compute_rmse(model, experiment, results, groupby='trial', threshold=0, km2=True, bs=1):
    ext = '_km2' if km2 else ''

    results[f'squared_error{ext}'] = (results[f'residual{ext}'] / bs).pow(2)
    df = results.query(f'missing == 0 & gt{ext} >= {threshold}') # & night == 1')
    rmse = df.groupby(groupby)[f'squared_error{ext}'].aggregate(np.mean).apply(np.sqrt)
    rmse = rmse.reset_index(name='rmse')
    rmse['model'] = model
    rmse['experiment'] = experiment

    return rmse

def aggregate_nights(results):
    groups = [list(g) for k, g in it.groupby(enumerate(results.night), key=lambda x: x[-1])]
    nights = [[item[0] for item in g] for g in groups if g[0][1]]

    results['nightID'] = -1
    for idx, night in enumerate(nights):
        results.loc[night, 'nightID'] = idx

    results = results.groupby('nightID').aggregate(np.sum)
    results = results.reset_index()

    return results


def compute_rmse_per_night(model, experiment, results, tidx2night, groupby='trial', threshold=0, km2=True, bs=1):
    ext = '_km2' if km2 else ''

    results['nightID'] = results.tidx.apply(lambda x: tidx2night[x])
    results = results.query(f'missing == 0 & gt{ext} >= {threshold} & nightID > 0')
    results = results.groupby(['nightID', 'trial', 'seqID', 'radar']).aggregate(np.mean)
    results = results.reset_index()

    results[f'squared_error_night{ext}'] = (results[f'residual{ext}'] / bs).pow(2)

    # make sure night_horizon starts at 0 for all sequences
    results['night_horizon'] = results['nightID'] - results.groupby('seqID')['nightID'].transform('min')

    # make sure that at each horizon the same nights are used (to be comparable)
    nights = set(results.nightID.unique())
    for group, df in results.groupby('night_horizon'):
        reduced_nights = nights.intersection(set(df.nightID.unique()))
        if len(reduced_nights) / len(nights) > 0.05:
            # reduce only if not too many nights are thrown away
            nights = reduced_nights
    results = results[results.nightID.isin(nights)]

    rmse = results.groupby(groupby)[f'squared_error_night{ext}'].aggregate(np.mean).apply(np.sqrt)
    rmse = rmse.reset_index(name='rmse')
    rmse['model'] = model
    rmse['experiment'] = experiment

    return rmse

def compute_mae(model, experiment, results, groupby='trial', threshold=0, km2=True, bs=1):
    ext = '_km2' if km2 else ''

    results[f'absolute_error{ext}'] = (results[f'residual{ext}'] / bs).abs()
    df = results.query(f'missing == 0 & gt{ext} >= {threshold}') # & night == 1')
    mae = df.groupby(groupby)[f'absolute_error{ext}'].aggregate(np.mean)
    mae = mae.reset_index(name='mae')
    mae['model'] = model
    mae['experiment'] = experiment

    return mae


def compute_mae_per_night(model, experiment, results, tidx2night, groupby='trial', threshold=0, km2=True, bs=1):
    ext = '_km2' if km2 else ''

    results['nightID'] = results.tidx.apply(lambda x: tidx2night[x])
    results = results.query(f'missing == 0 & gt{ext} >= {threshold} & nightID > 0')
    results = results.groupby(['nightID', 'trial', 'seqID', 'radar']).aggregate(np.mean)
    results = results.reset_index()

    results[f'absolute_error_night{ext}'] = (results[f'residual{ext}'] / bs).abs()

    # make sure night_horizon starts at 0 for all sequences
    results['night_horizon'] = results['nightID'] - results.groupby('seqID')['nightID'].transform('min')

    # make sure that at each horizon the same nights are used (to be comparable)
    nights = set(results.nightID.unique())
    for group, df in results.groupby('night_horizon'):
        reduced_nights = nights.intersection(set(df.nightID.unique()))
        if len(reduced_nights) / len(nights) > 0.05:
            # reduce only if not too many nights are thrown away
            nights = reduced_nights
    results = results[results.nightID.isin(nights)]


    mae = results.groupby(groupby)[f'absolute_error_night{ext}'].aggregate(np.mean)
    mae = mae.reset_index(name='mae')
    mae['model'] = model
    mae['experiment'] = experiment

    return mae


def compute_pcc(model, experiment, results, groupby='trial', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    df = results.query(f'missing == 0 & gt{ext} >= {threshold}').dropna()
    pcc = df.groupby(groupby)[[f'gt{ext}', f'prediction{ext}']].corr().iloc[0::2, -1]
    pcc = pcc.reset_index()
    pcc['pcc'] = pcc[f'prediction{ext}']
    pcc['model'] = model
    pcc['experiment'] = experiment

    return pcc

def compute_bin(model, experiment, results, groupby='trial', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    df = results.query(f'missing == 0').dropna()
    df['tp'] = (df[f'prediction{ext}'] > threshold) & (df[f'gt{ext}'] > threshold)
    df['fp'] = (df[f'prediction{ext}'] > threshold) & (df[f'gt{ext}'] < threshold)
    df['fn'] = (df[f'prediction{ext}'] < threshold) & (df[f'gt{ext}'] > threshold)
    df['tn'] = (df[f'prediction{ext}'] < threshold) & (df[f'gt{ext}'] < threshold)

    bin = df.groupby(groupby).aggregate(sum).reset_index()
    bin['accuracy'] = (bin.tp + bin.tn) / (bin.tp + bin.fp + bin.tn + bin.fn)
    bin['precision'] = bin.tp / (bin.tp + bin.fp)
    bin['sensitivity'] = bin.tp / (bin.tp + bin.fn)
    bin['specificity'] = bin.tn / (bin.tn + bin.fp)
    bin['fscore'] = 2 / ((1/bin.precision) + (1/bin.sensitivity))

    bin = bin.reset_index()
    bin['model'] = model
    bin['experiment'] = experiment
    bin['threshold'] = threshold

    return bin

def compute_residual_corr(results, km2=True):
    ext = '_km2' if km2 else ''

    radars = [r for r in results.radar.unique() if not 'boundary' in r]
    
    corr = []
    radars1 = []
    radars2 = []
    for r1, r2 in it.product(radars, repeat=2):
        data1 = results.query(f'radar == "{r1}"')[f'residual{ext}'].to_numpy()
        data2 = results.query(f'radar == "{r2}"')[f'residual{ext}'].to_numpy()

        mask = np.logical_and(np.isfinite(data1), np.isfinite(data2))
        r, p = stats.pearsonr(data1[mask], data2[mask])
        radars1.append(r1)
        radars2.append(r2)
        corr.append(r)

    corr = pd.DataFrame(list(zip(radars1, radars2, corr)), columns=['radar1', 'radar2', 'corr'])

    return corr


if __name__ == "__main__":

    evaluate()