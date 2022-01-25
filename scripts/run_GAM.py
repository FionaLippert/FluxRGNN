from birds import dataloader
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle5 as pickle
import os.path as osp
import numpy as np
import pandas as pd
import ruamel.yaml
from pygam import PoissonGAM, te


#@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GAM'

    data_root = osp.join(cfg.device.root, 'data')
    seq_len = cfg.model.horizon

    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_root={cfg.root_transform}_' \
                        f'fixedT0={cfg.fixed_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}'

    print('normalize features')
    training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
    normalization = dataloader.Normalization(training_years, cfg.datasource.name,
                                             data_root, preprocessed_dirname, **cfg)

    print('load data')
    data_list = [dataloader.RadarData(year, seq_len, preprocessed_dirname, processed_dirname,
                                 **cfg, **cfg.model,
                                 data_root=data_root,
                                 data_source=cfg.datasource.name,
                                 normalization=normalization,
                                 env_vars=cfg.datasource.env_vars,
                                 )
            for year in training_years]


    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    cfg.datasource.bird_scale = float(normalization.max('birds_km2'))

    all_X = []
    all_y = []
    all_masks = []
    all_mappings = []
    for idx, data in enumerate(data_list):
        X_train, y_train, mask_train = dataloader.get_training_data_gam(data, timesteps=seq_len, mask_daytime=False)
        all_X.append(X_train)
        all_y.append(y_train)
        all_masks.append(mask_train)
        radars = ['nldbl-nlhrw' if r in ['nldbl', 'nlhrw'] else r for r in data.info['radars']]
        m = {name: jdx for jdx, name in enumerate(radars)}
        all_mappings.append(m)

    print(all_mappings)

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


    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def test(cfg: DictConfig, output_dir: str, log, model_dir=None, ext=''):
    assert cfg.model.name == 'GAM'

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

    cfg.datasource.bird_scale = float(normalization.max('birds_km2'))

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

    X_test, y_test, mask_test = dataloader.get_test_data_gam(test_data,
                                                           context=cfg.model.test_context,
                                                           horizon=cfg.model.test_horizon,
                                                           mask_daytime=False)
    

    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], gt=[], prediction=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[], constant_prediction=[])

    for nidx, data in enumerate(test_data):
        y = data.y * cfg.datasource.bird_scale
        _tidx = data.tidx
        local_night = data.local_night
        missing = data.missing

        if cfg.root_transform > 0:
            y = np.power(y, cfg.root_transform)

        fill_context = np.ones(cfg.model.test_context) * np.nan

        for ridx, name in radar_index.items():
            if name in ['nlhrw', 'nldbl']: name = 'nldbl-nlhrw'
            with open(osp.join(model_dir, f'model_{name}.pkl'), 'rb') as f:
                model = pickle.load(f)
            y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale
            if cfg.root_transform > 0:
                y_hat = np.power(y_hat, cfg.root_transform)
            y_hat = np.concatenate([fill_context, y_hat])

            results['gt_km2'].append(y[ridx, :] if cfg.model.birds_per_km2 else y[ridx, :] / areas[ridx])
            results['prediction_km2'].append(y_hat if cfg.model.birds_per_km2 else y_hat / areas[ridx])
            results['gt'].append(y[ridx, :] * areas[ridx] if cfg.model.birds_per_km2 else y[ridx, :])
            results['prediction'].append(y_hat * areas[ridx] if cfg.model.birds_per_km2 else y_hat)
            results['constant_prediction'].append([y[ridx, 0]] * y.shape[1])
            results['night'].append(local_night[ridx, :])
            results['radar'].append([name] * y.shape[1])
            results['seqID'].append([nidx] * y.shape[1])
            results['tidx'].append(_tidx)
            results['datetime'].append(time[_tidx])
            results['trial'].append([cfg.get('job_id', 0)] * y.shape[1])
            results['horizon'].append(np.arange(-(cfg.model.test_context-1), cfg.model.test_horizon+1))
            results['missing'].append(missing[ridx, :])

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
