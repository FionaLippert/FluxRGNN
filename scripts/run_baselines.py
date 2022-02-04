from fluxrgnn import dataloader, utils
import torch
from torch.utils.data import random_split, Subset
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
import os.path as osp
import os
import numpy as np
import ruamel.yaml
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from pygam import PoissonGAM, te


def run(cfg: DictConfig, output_dir: str, log):
    """
    Run training and/or testing for baseline model.

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which all outputs are written to
    :param log: log file
    """

    if 'search' in cfg.task.name and cfg.model.name == 'GBT':
        cross_validation_GBT(cfg, output_dir, log)
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


def train(cfg, output_dir, log):
    if cfg.model.name == 'GBT':
        train_GBT(cfg, output_dir, log)
    elif cfg.model.name == 'GAM':
        train_GAM(cfg, output_dir, log)
    elif cfg.model.name == 'HA':
        train_HA(cfg, output_dir, log)



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


def train_GBT(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GBT'

    data, _, _, seq_len = dataloader.load_dataset(cfg, output_dir, training=True)
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


    print('------------------ model settings --------------------')
    print(cfg.model)
    print('------------------------------------------------------')


    print(f'train model')
    model = fit_GBT(X_train[mask_train], y_train[mask_train], **cfg.model, seed=cfg.seed + cfg.get('job_id', 0))

    with open(osp.join(output_dir, f'model.pkl'), 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    y_hat = model.predict(X_val)
    val_loss = utils.MSE_numpy(y_hat, y_val, mask_val)

    print(f'validation loss = {val_loss}', file=log)

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def train_HA(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'HA'

    data_list, _, _, seq_len = dataloader.load_dataset(cfg, output_dir, training=True)

    all_y = []
    all_masks = []
    all_mappings = []
    for idx, data in enumerate(data_list):
        _, y_train, mask_train = dataloader.get_training_data(cfg.model.name, data, timesteps=seq_len, mask_daytime=True)
        all_y.append(y_train)
        all_masks.append(mask_train)
        radars = ['nldbl-nlhrw' if r in ['nldbl', 'nlhrw'] else r for r in data.info['radars']]
        m = {name: idx for idx, name in enumerate(radars)}
        all_mappings.append(m)

    ha = dict()
    for r in all_mappings[0].keys():
        y_r = []
        for i, mapping in enumerate(all_mappings):
            ridx = mapping[r]
            mask = all_masks[i][:, ridx]
            y_r.append(all_y[i][mask, ridx])
        y_r = np.concatenate(y_r, axis=0)

        ha[r] = y_r.mean()

    with open(osp.join(output_dir, f'HAs.pkl'), 'wb') as f:
        pickle.dump(ha, f)

    log.flush()


def train_GAM(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GAM'

    data_list, _, _, seq_len = dataloader.load_dataset(cfg, output_dir, training=True)

    all_X = []
    all_y = []
    all_masks = []
    all_mappings = []
    for idx, data in enumerate(data_list):
        X_train, y_train, mask_train = dataloader.get_training_data(cfg.model.name, data, timesteps=seq_len, mask_daytime=False)
        all_X.append(X_train)
        all_y.append(y_train)
        all_masks.append(mask_train)
        radars = ['nldbl-nlhrw' if r in ['nldbl', 'nlhrw'] else r for r in data.info['radars']]
        m = {name: jdx for jdx, name in enumerate(radars)}
        all_mappings.append(m)

    for r in all_mappings[0].keys():
        X_r = []
        y_r = []
        for i, mapping in enumerate(all_mappings):
            ridx = mapping[r]
            X_r.append(all_X[i][all_masks[i][:, ridx], ridx]) # shape (time, features)
            y_r.append(all_y[i][all_masks[i][:, ridx], ridx]) # shape (time)
        X_r = np.concatenate(X_r, axis=0)
        y_r = np.concatenate(y_r, axis=0)


        # fit GAM with poisson distribution and log link
        print(f'fitting GAM for radar {r}')
        gam = PoissonGAM(te(0, 1, 2))
        gam.fit(X_r, y_r)

        with open(osp.join(output_dir, f'model_{r}.pkl'), 'wb') as f:
            pickle.dump(gam, f)

    log.flush()


def cross_validation_GBT(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GBT'

    n_folds = cfg.task.n_folds
    data, _, _, seq_len = dataloader.load_dataset(cfg, output_dir, training=True)
    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print(f'environmental variables: {cfg.datasource.env_vars}')

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
        model = fit_GBT(X_train[mask_train], y_train[mask_train], **cfg.model, seed=cfg.seed + cfg.get('job_id', 0))

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


def test(cfg: DictConfig, output_dir: str, log, ext=''):
    assert cfg.model.name in ['HA', 'GAM', 'GBT']

    model_dir = cfg.get('model_dir', output_dir)
    model_cfg = utils.load_model_cfg(model_dir)
    cfg.datasource.bird_scale = model_cfg['datasource']['bird_scale']

    test_data, input_col, context, seed = dataloader.load_dataset(cfg, output_dir, training=False)
    test_data = test_data[0]

    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    areas = test_data.info['areas']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    X_test, y_test, mask_test = dataloader.get_test_data(cfg.model.name, test_data,
                                                           context=cfg.model.test_context,
                                                           horizon=cfg.model.test_horizon,
                                                           mask_daytime=False,
                                                           use_acc_vars=cfg.model.get('use_acc_vars', False))

    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], gt=[], prediction=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[])


    for nidx, data in enumerate(test_data):
        y = data.y * cfg.datasource.bird_scale
        _tidx = data.tidx
        local_night = data.local_night
        missing = data.missing

        if cfg.root_transform > 0:
            y = np.power(y, cfg.root_transform)

        fill_context = np.ones(cfg.model.test_context) * np.nan

        for ridx, name in radar_index.items():
            # make predictions
            if cfg.model.name == 'GBT':
                with open(osp.join(model_dir, f'model.pkl'), 'rb') as f:
                    model = pickle.load(f)
                y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale
            else:
                if name in ['nlhrw', 'nldbl']:
                    name = 'nldbl-nlhrw'
                if cfg.model.name == 'HA':
                    with open(osp.join(model_dir, f'HAs.pkl'), 'rb') as f:
                        model = pickle.load(f)
                    y_hat = model[name] * cfg.datasource.bird_scale
                elif cfg.model.name == 'GAM':
                    with open(osp.join(model_dir, f'model_{name}.pkl'), 'rb') as f:
                        model = pickle.load(f)
                    y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale

            if cfg.root_transform > 0:
                y_hat = np.power(y_hat, cfg.root_transform)
            y_hat = np.concatenate([fill_context, y_hat])

            # store results
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

    utils.finalize_results(results, output_dir, ext)

    with open(osp.join(output_dir, f'radar_index.pickle'), 'wb') as f:
        pickle.dump(radar_index, f, pickle.HIGHEST_PROTOCOL)

    print(f'successfully saved results to {osp.join(output_dir, f"results{ext}.csv")}', file=log)
    log.flush()

