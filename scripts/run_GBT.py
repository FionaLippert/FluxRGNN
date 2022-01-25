from birds import dataloader, utils
import torch
from torch.utils.data import random_split, Subset
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle5 as pickle
import os.path as osp
import os
import numpy as np
import ruamel.yaml
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def fit_GBT(X, y, **kwargs):
    seed = kwargs.get('seed', 1234)
    n_estimators = kwargs.get('n_estimators', 100)
    lr = kwargs.get('lr', 0.05)
    max_depth = kwargs.get('max_depth', 5)
    tolerance = kwargs.get('tolerance', 1e-6)

    gbt = GradientBoostingRegressor(random_state=seed, n_estimators=n_estimators, learning_rate=lr,
                                    max_depth=max_depth, tol=tolerance, n_iter_no_change=10)
    gbt.fit(X, y)
    return gbt


def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GBT'

    seq_len = cfg.model.horizon
    seed = cfg.seed + cfg.get('job_id', 0)

    data_root = osp.join(cfg.device.root, 'data')
    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_root={cfg.root_transform}_' \
                        f'fixedT0={cfg.fixed_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'

    print('normalize features')
    training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
    normalization = dataloader.Normalization(training_years, cfg.datasource.name,
                                             data_root, preprocessed_dirname, **cfg)

    print('load data')
    data = [dataloader.RadarData(year, seq_len, preprocessed_dirname, processed_dirname,
                                 **cfg, **cfg.model,
                                 data_root=data_root,
                                 data_source=cfg.datasource.name,
                                 normalization=normalization,
                                 env_vars=cfg.datasource.env_vars,
                                 )
            for year in training_years]

    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    # split data into training and validation set
    n_val = max(1, int(cfg.datasource.val_train_split * n_data))
    n_train = n_data - n_val

    print('------------------------------------------------------')
    print('-------------------- data sets -----------------------')
    print(f'total number of sequences = {n_data}')
    print(f'number of training sequences = {n_train}')
    print(f'number of validation sequences = {n_val}')

    train_data, val_data = random_split(data, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed))
    X_train, y_train, mask_train = dataloader.get_training_data_gbt(train_data, timesteps=seq_len, mask_daytime=False,
                                                    use_acc_vars=cfg.model.use_acc_vars)

    X_val, y_val, mask_val = dataloader.get_training_data_gbt(val_data, timesteps=seq_len, mask_daytime=False,
                                              use_acc_vars=cfg.model.use_acc_vars)

    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max('birds_km2'))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max('birds_km2', cfg.root_transform))
    cfg.model_seed = seed

    print('------------------ model settings --------------------')
    print(cfg.model)
    print('------------------------------------------------------')


    print(f'train model')
    model = fit_GBT(X_train[mask_train], y_train[mask_train], **cfg.model, seed=seed)

    with open(osp.join(output_dir, f'model.pkl'), 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    y_hat = model.predict(X_val)
    val_loss = utils.MSE_numpy(y_hat, y_val, mask_val)

    print(f'validation loss = {val_loss}', file=log)

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def cross_validation(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GBT'

    seq_len = cfg.model.horizon

    n_folds = cfg.task.n_folds
    seed = cfg.seed + cfg.get('job_id', 0)

    data_root = osp.join(cfg.device.root, 'data')
    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_root={cfg.root_transform}_' \
                        f'fixedT0={cfg.fixed_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'

    print('normalize features')
    training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
    normalization = dataloader.Normalization(training_years, cfg.datasource.name,
                                             data_root, preprocessed_dirname, **cfg)

    print('load data')
    data = [dataloader.RadarData(year, seq_len, preprocessed_dirname, processed_dirname,
                                 **cfg, **cfg.model,
                                 data_root=data_root,
                                 data_source=cfg.datasource.name,
                                 normalization=normalization,
                                 env_vars=cfg.datasource.env_vars,
                                 )
            for year in training_years]

    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print(f'environmental variables: {cfg.datasource.env_vars}')


    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max('birds_km2'))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max('birds_km2', cfg.root_transform))
    cfg.model_seed = seed

    cv_folds = np.array_split(np.arange(n_data), n_folds)

    if cfg.verbose: print(f'--- run cross-validation with {n_folds} folds ---')


    best_val_losses = np.ones(n_folds) * np.nan

    for f in range(n_folds):
        if cfg.verbose: print(f'------------------- fold = {f} ----------------------')

        subdir = osp.join(output_dir, f'cv_fold_{f}')
        os.makedirs(subdir, exist_ok=True)

        # split into training and validation set
        val_data = Subset(data, cv_folds[f].tolist())
        train_idx = np.concatenate([cv_folds[i] for i in range(n_folds) if i!=f]).tolist()
        train_data = Subset(data, train_idx) # everything else
        X_train, y_train, mask_train = dataloader.get_training_data_gbt(train_data, timesteps=seq_len, mask_daytime=False,
                                                        use_acc_vars=cfg.model.use_acc_vars)

        X_val, y_val, mask_val = dataloader.get_training_data_gbt(val_data, timesteps=seq_len, mask_daytime=False,
                                                  use_acc_vars=cfg.model.use_acc_vars)

        print(f'train model')
        model = fit_GBT(X_train[mask_train], y_train[mask_train], **cfg.model, seed=seed)

        with open(osp.join(subdir, f'model.pkl'), 'wb') as file:
            pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)

        y_hat = model.predict(X_val)
        val_loss = utils.MSE_numpy(y_hat, y_val, mask_val)

        print(f'fold {f}: validation loss = {val_loss}', file=log)
        best_val_losses[f] = val_loss

        log.flush()

    print(f'average validation loss = {np.nanmean(best_val_losses)}', file=log)

    summary = pd.DataFrame({'fold': range(n_folds),
                            'final_val_loss': best_val_losses,
                            'best_val_loss': best_val_losses})
    summary.to_csv(osp.join(output_dir, 'summary.csv'))

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()


def test(cfg: DictConfig, output_dir: str, log, model_dir=None, ext=''):
    assert cfg.model.name == 'GBT'

    data_root = osp.join(cfg.device.root, 'data')
    seq_len = cfg.model.test_context + cfg.model.test_horizon
    if model_dir is None: model_dir = output_dir

    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_root={cfg.root_transform}_' \
                        f'fixedT0={cfg.fixed_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'

    # load normalizer
    with open(osp.join(model_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max('birds_km2'))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max('birds_km2', cfg.root_transform))

    # load test data
    test_data = dataloader.RadarData(str(cfg.datasource.test_year), seq_len,
                                     preprocessed_dirname, processed_dirname,
                                     **cfg, **cfg.model,
                                     data_root=data_root,
                                     data_source=cfg.datasource.name,
                                     normalization=normalization,
                                     env_vars=cfg.datasource.env_vars,
                                     )
    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    areas = test_data.info['areas']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    X_test, y_test, mask_test = dataloader.get_test_data_gbt(test_data,
                                                           context=cfg.model.test_context,
                                                           horizon=cfg.model.test_horizon,
                                                           mask_daytime=False,
                                                           use_acc_vars=cfg.model.use_acc_vars)


    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], gt=[], prediction=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[])

    with open(osp.join(model_dir, f'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    for nidx, data in enumerate(test_data):
        y = data.y * cfg.datasource.bird_scale
        _tidx = data.tidx
        local_night = data.local_night
        missing = data.missing

        if cfg.root_transform > 0:
            y = np.power(y, cfg.root_transform)

        fill_context = np.ones(cfg.model.test_context) * np.nan

        for ridx, name in radar_index.items():
            y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale
            if cfg.root_transform > 0:
                y_hat = np.power(y_hat, cfg.root_transform)
            y_hat = np.concatenate([fill_context, y_hat])
            
            results['gt_km2'].append(y[ridx, :] if cfg.model.birds_per_km2 else y[ridx, :] / areas[ridx])
            results['prediction_km2'].append(y_hat if cfg.model.birds_per_km2 else y_hat / areas[ridx])
            results['gt'].append(y[ridx, :] * areas[ridx] if cfg.model.birds_per_km2 else y[ridx, :])
            results['prediction'].append(y_hat * areas[ridx] if cfg.model.birds_per_km2 else y_hat)
            results['night'].append(local_night[ridx, :])
            results['radar'].append([name] * y.shape[1])
            results['seqID'].append([nidx] * y.shape[1])
            results['tidx'].append(_tidx)
            results['datetime'].append(time[_tidx])
            results['trial'].append([cfg.get('job_id', 0)] * y.shape[1])
            results['horizon'].append(np.arange(-(cfg.model.test_context-1), cfg.model.test_horizon+1))
            results['missing'].append(missing[ridx, :])

    with open(osp.join(output_dir, f'radar_index.pickle'), 'wb') as f:
        pickle.dump(radar_index, f, pickle.HIGHEST_PROTOCOL)

    # create dataframe containing all results
    for k, v in results.items():
        results[k] = np.concatenate(v)
    results['residual_km2'] = results['gt_km2'] - results['prediction_km2']
    results['residual'] = results['gt'] - results['prediction']
    df = pd.DataFrame(results)
    df.to_csv(osp.join(output_dir, f'results{ext}.csv'))

    print(f'successfully saved results to {osp.join(output_dir, f"results{ext}.csv")}', file=log)
    log.flush()



def run(cfg: DictConfig, output_dir: str, log):
    if 'search' in cfg.task.name:
        cross_validation(cfg, output_dir, log)
    if 'train' in cfg.task.name:
        train(cfg, output_dir, log)
    if 'eval' in cfg.task.name:

        if hasattr(cfg, 'importance_sampling'):
            cfg.importance_sampling = False

        cfg['fixed_t0'] = True
        test(cfg, output_dir, log, ext='_fixedT0')
        cfg['fixed_t0'] = False
        test(cfg, output_dir, log)

        if cfg.get('test_train_data', False):
            training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
            cfg.model.test_horizon = cfg.model.horizon
            for y in training_years:
                cfg.datasource.test_year = y
                test(cfg, output_dir, log, ext=f'_training_year_{y}')


if __name__ == "__main__":
    train()
