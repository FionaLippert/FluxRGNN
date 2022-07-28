from fluxrgnn import dataloader, utils
from fluxrgnn.models import *
import torch
from torch.utils.data import random_split, Subset
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
from omegaconf import DictConfig, OmegaConf
import pickle
import os.path as osp
import os
import numpy as np
import pandas as pd

# map model name to implementation
MODEL_MAPPING = {'LocalMLP': LocalMLP,
                 'LocalLSTM': LocalLSTM,
                 'FluxRGNN': FluxRGNN}

def run(cfg: DictConfig, output_dir: str, log):
    """
    Run training and/or testing for neural network model.

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which all outputs are written to
    :param log: log file
    """

    if 'search' in cfg.task.name:
        cross_validation(cfg, output_dir, log)
    if 'train' in cfg.task.name:
        training(cfg, output_dir, log)
    if 'eval' in cfg.task.name:
        if hasattr(cfg, 'importance_sampling'):
            cfg.importance_sampling = False

        cfg['fixed_t0'] = True
        testing(cfg, output_dir, log, ext='_fixedT0')
        cfg['fixed_t0'] = False
        testing(cfg, output_dir, log)

        if cfg.get('test_train_data', False):
            # evaluate performance on training data
            training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
            cfg.model.test_horizon = cfg.model.horizon
            for y in training_years:
                cfg.datasource.test_year = y
                testing(cfg, output_dir, log, ext=f'_training_year_{y}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training(cfg: DictConfig, output_dir: str, log):
    """
    Run training of a neural network model.

    :param cfg: DictConfig specifying model, data and training details
    :param output_dir: directory to which the final model and logs are written to
    :param log: log file
    """

    assert cfg.model.name in MODEL_MAPPING

    if cfg.debugging: torch.autograd.set_detect_anomaly(True)

    Model = MODEL_MAPPING[cfg.model.name]

    device = 'cuda' if (cfg.device.cuda and torch.cuda.is_available()) else 'cpu'
    seed = cfg.seed + cfg.get('job_id', 0)

    data = dataloader.load_dataset(cfg, output_dir, training=True)[0]
    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    print('done with setup', file=log)
    log.flush()

    # split data into training and validation set
    n_val = max(1, int(cfg.datasource.val_train_split * n_data))
    n_train = n_data - n_val

    if cfg.verbose:
        print('------------------------------------------------------', file=log)
        print('-------------------- data sets -----------------------', file=log)
        print(f'total number of sequences = {n_data}', file=log)
        print(f'number of training sequences = {n_train}', file=log)
        print(f'number of validation sequences = {n_val}', file=log)

    train_data, val_data = random_split(data, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_data, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    if cfg.model.edge_type == 'voronoi':
        n_edge_attr = 5
    else:
        n_edge_attr = 4

    if cfg.model.get('root_transformed_loss', False):
        loss_func = utils.MSE_root_transformed
    elif cfg.model.get('weighted_loss', False):
        loss_func = utils.MSE_weighted
    else:
        loss_func = utils.MSE

    if cfg.verbose:
        print('------------------ model settings --------------------', file=log)
        print(cfg.model, file=log)
        print('------------------------------------------------------', file=log)

    log.flush()

    best_val_loss = np.inf
    training_curve = np.ones((1, cfg.model.epochs)) * np.nan
    val_curve = np.ones((1, cfg.model.epochs)) * np.nan

    model = Model(n_env=len(cfg.datasource.env_vars), coord_dim=2, n_edge_attr=n_edge_attr,
                  seed=seed, **cfg.model)

    n_params = count_parameters(model)

    if cfg.verbose:
        print('initialized model', file=log)
        print(f'number of model parameters: {n_params}', file=log)
        print(f'environmental variables: {cfg.datasource.env_vars}')

    log.flush()

    ext = ''
    if cfg.get('use_pretrained_model', False):
        states_path = osp.join(output_dir, 'model.pkl')
        if osp.isfile(states_path):
            model.load_state_dict(torch.load(states_path))
            if cfg.verbose: print('successfully loaded pretrained model')
            ext = '_pretrained'

    model = model.to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=cfg.model.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model.lr_decay, gamma=cfg.model.get('lr_gamma', 0.1))

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))


    tf = cfg.model.get('teacher_forcing_init', 1.0)
    all_tf = np.zeros(cfg.model.epochs)
    all_lr = np.zeros(cfg.model.epochs)
    avg_loss = np.inf
    saved = False

    for epoch in range(cfg.model.epochs):
        all_tf[epoch] = tf
        all_lr[epoch] = optimizer.param_groups[0]["lr"]

        loss = train(model, train_loader, optimizer, loss_func, device, teacher_forcing=tf, **cfg.model)
        training_curve[0, epoch] = loss / n_train

        val_loss = test(model, val_loader, loss_func, device, **cfg.model).cpu()
        val_loss = val_loss[torch.isfinite(val_loss)].mean()
        val_curve[0, epoch] = val_loss

        if cfg.verbose:
            print(f'epoch {epoch + 1}: loss = {training_curve[0, epoch]}', file=log)
            print(f'epoch {epoch + 1}: val loss = {val_loss}', file=log)
            log.flush()

        if val_loss <= best_val_loss:
            if cfg.verbose: print('best model so far; save to disk ...')
            torch.save(model.state_dict(), osp.join(output_dir, f'best_model{ext}.pkl'))
            best_val_loss = val_loss

        if cfg.model.early_stopping and (epoch + 1) % cfg.model.avg_window == 0:
            # every X epochs, check for convergence of validation loss
            if epoch == 0:
                l = val_curve[0, 0]
            else:
                l = val_curve[0, (epoch - (cfg.model.avg_window - 1)) : (epoch + 1)].mean()
            if (avg_loss - l) > cfg.model.stopping_criterion:
                # loss decayed significantly, continue training
                avg_loss = l
                torch.save(model.state_dict(), osp.join(output_dir, f'model{ext}.pkl'))
                saved = True
            else:
                # loss converged sufficiently, stop training
                break

        tf = tf * cfg.model.get('teacher_forcing_gamma', 0)
        scheduler.step()

    if not cfg.model.early_stopping or not saved:
        torch.save(model.state_dict(), osp.join(output_dir, f'model{ext}.pkl'))

    if cfg.verbose:
        print(f'validation loss = {best_val_loss}', file=log)
        log.flush()

    # save training and validation curves
    np.save(osp.join(output_dir, f'training_curves{ext}.npy'), training_curve)
    np.save(osp.join(output_dir, f'validation_curves{ext}.npy'), val_curve)
    np.save(osp.join(output_dir, f'learning_rates{ext}.npy'), all_lr)
    np.save(osp.join(output_dir, f'teacher_forcing{ext}.npy'), all_tf)

    # plotting
    utils.plot_training_curves(training_curve, val_curve, output_dir, log=True)
    utils.plot_training_curves(training_curve, val_curve, output_dir, log=False)

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def cross_validation(cfg: DictConfig, output_dir: str, log):
    """
    Run cross-validation for neural network model.

    The training data is split into N subsets, and N models are trained where for each model a different subset
    is left for validation.

    :param cfg: DictConfig specifying the model, data and training details, including the number of folds to use
    :param output_dir: directory to which all N models and logs are written to
    :param log: log file
    """
    assert cfg.model.name in MODEL_MAPPING

    if cfg.debugging: torch.autograd.set_detect_anomaly(True)

    Model = MODEL_MAPPING[cfg.model.name]

    device = 'cuda' if (cfg.device.cuda and torch.cuda.is_available()) else 'cpu'
    epochs = cfg.model.epochs
    n_folds = cfg.task.n_folds
    seed = cfg.seed + cfg.get('job_id', 0)

    data = dataloader.load_dataset(cfg, output_dir, training=True)[0]
    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    if cfg.model.edge_type == 'voronoi':
        n_edge_attr = 5
    else:
        n_edge_attr = 4

    if cfg.model.get('root_transformed_loss', False):
        loss_func = utils.MSE_root_transformed
    elif cfg.model.get('weighted_loss', False):
        loss_func = utils.MSE_weighted
    else:
        loss_func = utils.MSE

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print(f'environmental variables: {cfg.datasource.env_vars}')

    cv_folds = np.array_split(np.arange(n_data), n_folds)

    if cfg.verbose: print(f'--- run cross-validation with {n_folds} folds ---')

    training_curves = np.ones((n_folds, epochs)) * np.nan
    val_curves = np.ones((n_folds, epochs)) * np.nan
    best_val_losses = np.ones(n_folds) * np.nan
    best_epochs = np.zeros(n_folds)

    for f in range(n_folds):
        if cfg.verbose: print(f'------------------- fold = {f} ----------------------')

        subdir = osp.join(output_dir, f'cv_fold_{f}')
        os.makedirs(subdir, exist_ok=True)

        # split into training and validation set
        val_data = Subset(data, cv_folds[f].tolist())
        train_idx = np.concatenate([cv_folds[i] for i in range(n_folds) if i!=f]).tolist()
        n_train = len(train_idx)
        train_data = Subset(data, train_idx) # everything else
        train_loader = DataLoader(train_data, batch_size=cfg.model.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

        model = Model(n_env=len(cfg.datasource.env_vars), coord_dim=2, n_edge_attr=n_edge_attr,
                      seed=seed, **cfg.model)

        states_path = cfg.model.get('load_states_from', '')
        if osp.isfile(states_path):
            model.load_state_dict(torch.load(states_path))

        model = model.to(device)
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=cfg.model.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model.lr_decay, gamma=cfg.model.get('lr_gamma', 0.1))
        best_val_loss = np.inf
        avg_loss = np.inf

        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        tf = cfg.model.get('teacher_forcing_init', 1.0)
        all_tf = np.zeros(epochs)
        all_lr = np.zeros(epochs)
        for epoch in range(epochs):
            all_tf[epoch] = tf
            all_lr[epoch] = optimizer.param_groups[0]["lr"]

            loss = train(model, train_loader, optimizer, loss_func, device, teacher_forcing=tf, **cfg.model)
            training_curves[f, epoch] = loss / n_train

            val_loss = test(model, val_loader, loss_func, device, **cfg.model).cpu()
            val_loss = val_loss[torch.isfinite(val_loss)].mean()
            val_curves[f, epoch] = val_loss

            if cfg.verbose:
                print(f'epoch {epoch + 1}: loss = {training_curves[f, epoch]}')
                print(f'epoch {epoch + 1}: val loss = {val_loss}')

            if val_loss <= best_val_loss:
                if cfg.verbose: print('best model so far; save to disk ...')
                torch.save(model.state_dict(), osp.join(subdir, f'best_model.pkl'))
                best_val_loss = val_loss
                best_epochs[f] = epoch

            if cfg.model.early_stopping and (epoch % cfg.model.avg_window) == 0:
                # every X epochs, check for convergence of validation loss
                if epoch == 0:
                    l = val_curves[f, 0]
                else:
                    l = val_curves[f, (epoch - cfg.model.avg_window): epoch].mean()
                if (avg_loss - l) > cfg.model.stopping_criterion:
                    # loss decayed significantly, continue training
                    avg_loss = l
                    torch.save(model.state_dict(), osp.join(subdir, 'model.pkl'))
                else:
                    # loss converged sufficiently, stop training
                    val_curves[f, epoch:] = l
                    break

            tf = tf * cfg.model.get('teacher_forcing_gamma', 0)
            scheduler.step()

        if not cfg.model.early_stopping:
            torch.save(model.state_dict(), osp.join(subdir, 'model.pkl'))

        if cfg.verbose:
            print(f'fold {f}: final validation loss = {val_curves[f, -1]}', file=log)
        best_val_losses[f] = best_val_loss

        log.flush()

        # update training and validation curves
        np.save(osp.join(subdir, 'training_curves.npy'), training_curves)
        np.save(osp.join(subdir, 'validation_curves.npy'), val_curves)
        np.save(osp.join(subdir, 'learning_rates.npy'), all_lr)
        np.save(osp.join(subdir, 'teacher_forcing.npy'), all_tf)

        # plotting
        utils.plot_training_curves(training_curves, val_curves, subdir, log=True)
        utils.plot_training_curves(training_curves, val_curves, subdir, log=False)

    if cfg.verbose:
        print(f'average validation loss = {val_curves[:, -1].mean()}', file=log)

    summary = pd.DataFrame({'fold': range(n_folds),
                            'final_val_loss': val_curves[:, -cfg.model.avg_window:].mean(1),
                            'best_val_loss': best_val_losses,
                            'best_epoch': best_epochs})
    summary.to_csv(osp.join(output_dir, 'summary.csv'))


    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()


def testing(cfg: DictConfig, output_dir: str, log, ext=''):
    """
    Test neural network model on unseen test data.

    :param cfg: DictConfig specifying model, data and testing details
    :param output_dir: directory to which test results are written to
    """

    assert cfg.model.name in MODEL_MAPPING

    Model = MODEL_MAPPING[cfg.model.name]

    device = 'cuda' if (cfg.device.cuda and torch.cuda.is_available()) else 'cpu'

    if cfg.model.edge_type == 'voronoi':
        n_edge_attr = 5
    else:
        n_edge_attr = 4

    if cfg.get('use_pretrained_model', False):
        model_ext = '_pretrained'
    else:
        model_ext = ''
    ext = f'{ext}{model_ext}'

    model_dir = cfg.get('model_dir', output_dir)
    model_cfg = utils.load_model_cfg(model_dir)
    cfg.datasource.bird_scale = float(model_cfg['datasource']['bird_scale'])

    # load test data
    test_data, input_col, context, seq_len = dataloader.load_dataset(cfg, output_dir, training=False)
    test_data = test_data[0]
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    areas = test_data.info['areas']
    to_km2 = np.ones(len(radars)) if input_col == 'birds_km2' else test_data.info['areas']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], night=[], radar=[], area=[], seqID=[],
                   tidx=[], datetime=[], horizon=[], missing=[], trial=[])
    if cfg.model.name in ['LocalLSTM', 'FluxRGNN']:
        results['source_km2'] = []
        results['sink_km2'] = []
    if cfg.model.name == 'FluxRGNN':
        results['influx_km2'] = []
        results['outflux_km2'] = []


    model = Model(n_env=len(cfg.datasource.env_vars), coord_dim=2, n_edge_attr=n_edge_attr,
                  seed=model_cfg['seed'], **model_cfg['model'])
    model.load_state_dict(torch.load(osp.join(model_dir, f'model{model_ext}.pkl')))

    # adjust model settings for testing
    model.horizon = cfg.model.test_horizon
    if cfg.model.get('fixed_boundary', 0):
        model.fixed_boundary = True

    model.to(device)
    model.eval()

    edge_fluxes = {}
    radar_fluxes = {}

    for nidx, data in enumerate(test_loader):
        # load ground truth and predicted densities
        data = data.to(device)
        y_hat = model(data).cpu().detach() * cfg.datasource.bird_scale
        y = data.y.cpu() * cfg.datasource.bird_scale

        if cfg.root_transform > 0:
            # transform back
            y = torch.pow(y, cfg.root_transform)
            y_hat = torch.pow(y_hat, cfg.root_transform)

        _tidx = data.tidx.cpu()
        local_night = data.local_night.cpu()
        missing = data.missing.cpu()

        if 'Flux' in cfg.model.name:
            # fluxes along edges
            adj = to_dense_adj(data.edge_index, edge_attr=model.edge_fluxes)
            edge_fluxes[nidx] = adj.view(
                                data.num_nodes, data.num_nodes, -1).detach().cpu() * cfg.datasource.bird_scale

            # net fluxes per node
            influxes = edge_fluxes[nidx].sum(1)
            outfluxes = edge_fluxes[nidx].permute(1, 0, 2).sum(1)

            # absolute fluxes across Voronoi faces
            if input_col == 'birds_km2':
               edge_fluxes[nidx] *= areas.max()

            radar_fluxes[nidx] = to_dense_adj(data.edge_index, edge_attr=data.fluxes).view(
                data.num_nodes, data.num_nodes, -1).detach().cpu()

        if 'LSTM' in cfg.model.name or 'RGNN' in cfg.model.name:
            node_source = model.node_source.detach().cpu() * cfg.datasource.bird_scale
            node_sink = model.node_sink.detach().cpu() * cfg.datasource.bird_scale


        # fill prediction columns with nans for context timesteps
        fill_context = torch.ones(context) * float('nan')

        for ridx, name in radar_index.items():
            results['gt_km2'].append(y[ridx, :] / to_km2[ridx])
            results['prediction_km2'].append(torch.cat([fill_context, y_hat[ridx, :] / to_km2[ridx]]))
            results['night'].append(local_night[ridx, :])
            results['radar'].append([name] * y.shape[1])
            results['area'].append([areas[ridx]] * y.shape[1])
            results['seqID'].append([nidx] * y.shape[1])
            results['tidx'].append(_tidx)
            results['datetime'].append(time[_tidx])
            results['trial'].append([cfg.get('job_id', 0)] * y.shape[1])
            results['horizon'].append(np.arange(-(cfg.model.context-1), cfg.model.test_horizon+1))
            results['missing'].append(missing[ridx, :])

            if 'LSTM' in cfg.model.name or 'RGNN' in cfg.model.name:
                results['source_km2'].append(torch.cat([fill_context, node_source[ridx].view(-1) / to_km2[ridx]]))
                results['sink_km2'].append(torch.cat([fill_context, node_sink[ridx].view(-1) / to_km2[ridx]]))
            if 'Flux' in cfg.model.name:
                results['influx_km2'].append(torch.cat([fill_context, influxes[ridx].view(-1)]) / to_km2[ridx])
                results['outflux_km2'].append(torch.cat([fill_context, outfluxes[ridx].view(-1)]) / to_km2[ridx])

    utils.finalize_results(results, output_dir, ext)

    with open(osp.join(output_dir, f'radar_index.pickle'), 'wb') as f:
        pickle.dump(radar_index, f, pickle.HIGHEST_PROTOCOL)

    if 'Flux' in cfg.model.name:
        with open(osp.join(output_dir, f'model_fluxes{ext}.pickle'), 'wb') as f:
            pickle.dump(edge_fluxes, f, pickle.HIGHEST_PROTOCOL)

    if cfg.verbose:
        print(f'successfully saved results to {osp.join(output_dir, f"results{ext}.csv")}', file=log)
        log.flush()
