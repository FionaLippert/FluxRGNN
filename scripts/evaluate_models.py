import pandas as pd
import torch
import numpy as np
import os.path as osp
import os

def res_to_rmse(res):
    return np.sqrt(np.mean(np.square(res)))
def res_to_mae(res):
    return np.mean(np.abs(res))
def df_to_smape(df):
    abs_res = np.abs(df.residual)
    abs_sum = np.abs(df.predictions) + np.abs(df.measurements)
    ratios = (abs_res) / (abs_sum + 1e-8)
    smape = ratios.mean()
    return smape

def summarize_performance(test_results, cfg, var='x', groupby=[], path=None):

    all_results = {}
    
    predictions = test_results[f'test/predictions/{var}']
    measurements = test_results[f'test/measurements/{var}']
    
    S, R, V, T = predictions.size()
    
    all_results['measurements'] = measurements.flatten().cpu().numpy()
    all_results['predictions'] = predictions.flatten().cpu().numpy()

    densities = test_results['test/predictions/x']
    all_results['densities'] = densities.view(S, R, 1, T).repeat(1, 1, V, 1).flatten().cpu().numpy()

    masks = test_results[f'test/masks/{var}']
    all_results['valid_data'] = masks.view(S, R, 1, T).repeat(1, 1, V, 1).flatten().cpu().numpy()

    train_masks = test_results['test/train_mask']
    all_results['observed'] = train_masks.view(S, R, 1, 1).repeat(1, 1, V, T).flatten().cpu().numpy()

    horizons = torch.arange(T).view(1, 1, 1, T).repeat(S, R, V, 1)
    all_results['horizon'] = horizons.flatten().cpu().numpy()

    all_results['night'] = all_results['horizon'] // 24

    radars = torch.arange(R).view(1, R, 1, 1).repeat(S, 1, V, T)
    all_results['radar'] = radars.flatten().cpu().numpy()

    all_results['model'] = [cfg.model.name] * S * R * V * T
    all_results['fold'] = np.ones(S * R * V * T) * cfg.task.cv_fold

    df = pd.DataFrame(all_results)
    df = df.query('valid_data == True')

    if cfg.get('ignore_day', False):
        df = df.query('densities > 0.0')

    df['residual'] = df.predictions - df.measurements
    #df['measurements_pow'] = df.measurements.pow(1/3)
    #df['predictions_pow'] = df.predictions.pow(1/3)
    #df['residual_pow'] = df.predictions_pow - df.measurements_pow
    #df['night'] = df['horizon'].apply(lambda h: h // 24)

    #q = [0.0, .9, .95, .99, 1.]
    #zero_index = (df.densities < 1.0)
    #df.loc[zero_index, 'densities'] = 1.0 # to avoid negative lowest bin using qcut
    #df['quantile'], bins = pd.qcut(df['densities'], q, retbins=True, precision=0) #labels=q[1:])
    #bins = [0, 1, 10, 25, 50, 100, 150, 200, 250, 1500] # roughly [0%, 80%, 95%, 100%]
    bins = [0, 150, 1500]
    df['bird_bin'] = pd.cut(df['densities'], bins)

    grouped = df.groupby(groupby + ['model', 'fold'])

    df_eval = grouped['residual'].aggregate(res_to_rmse).reset_index(name='RMSE')
    df_eval['MAE'] = grouped['residual'].aggregate(res_to_mae).reset_index(name='MAE')['MAE']
    df_eval['SMAPE'] = [df_to_smape(df_sub) for group, df_sub in grouped]
    #df_rmse_pow = df.groupby(['model', 'fold', 'quantile', 'observed'])['residual_pow'].aggregate(res_to_rmse).reset_index(name='RMSE')
    #df_mae_pow = df.groupby(['model', 'fold', 'quantile', 'observed'])['residual_pow'].aggregate(res_to_mae).reset_index(name='MAE')

    if path is None:
        path = osp.join(cfg.output_dir, 'evaluation')
    os.makedirs(path, exist_ok=True)

    if len(groupby) == 0:
        groupby = ['none']
    df_eval.to_csv(osp.join(path, f'{cfg.logger.project}_{cfg.task.cv_fold}_{var}_{"+".join(groupby)}_model_evaluation.csv'))
    #df_mae.to_csv(f'{cfg.logger.project}_{cfg.task.cv_fold}_{var}_model_evaluation_MAE.csv')
    #df_rmse_pow.to_csv(f'{cfg.logger.project}_model_evaluation_RMSE_x_pow.csv')
    #df_mae_pow.to_csv(f'{cfg.logger.project}_model_evaluation_MAE_x_pow.csv')
