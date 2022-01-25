import numpy as np
import os
import os.path as osp
import pandas as pd
import geopandas as gpd
import pickle5 as pickle
from yaml import Loader, load
import itertools as it
import networkx as nx
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib import cm
from shapely import geometry
import geoplot as gplt
import torch
from cartopy.feature import ShapelyFeature
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs


def load_cv_results(result_dir, ext='', trials=1):

    result_list = []
    for t in range(1, trials+1):
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

def load_model_fluxes(result_dir, ext='', trials=1):

    fluxes = {}
    for t in range(1, trials + 1):

        file = osp.join(result_dir, f'trial_{t}', f'model_fluxes{ext}.pickle')
        if osp.isfile(file):
            with open(file, 'rb') as f:
                fluxes[t] = pickle.load(f)

    return fluxes



def flux_corr_per_dist2boundary(voronoi, model_fluxes, gt_fluxes):

    # shortest paths to any boundary cell
    sp = nx.shortest_path(G)
    d_to_b = np.zeros(len(G))
    for ni, datai in G.nodes(data=True):
        min_d = np.inf
        for nj, dataj in G.nodes(data=True):
            if dataj['boundary']:
                d = len(sp[ni][nj])
                if d < min_d:
                    min_d = d
                    d_to_b[ni] = d
    voronoi['dist2boundary'] = d_to_b

    print(voronoi)

    df = dict(radar1=[], radar2=[], corr=[], dist2boundary=[])
    for i, rowi in voronoi.iterrows():
        for j, rowj in voronoi.iterrows():
            if not i == j:
                df['radar1'].append(rowi['radar'])
                df['radar2'].append(rowj['radar'])
                df['dist2boundary'].append(int(min(rowi['dist2boundary'], rowj['dist2boundary'])))
                if not np.all(model_fluxes[i, j] == 0) and not np.all(gt_fluxes[i, j] == 0) and np.all(
                        np.isfinite(gt_fluxes[i, j])):
                    df['corr'].append(stats.pearsonr(gt_fluxes[i, j].flatten(), model_fluxes[i, j].flatten())[0])
                else:
                    df['corr'].append(np.nan)
    df = pd.DataFrame(df)

    return df

def fluxes_per_dist2boundary(G, voronoi):
    
    # shortest paths to any boundary cell
    sp = nx.shortest_path(G)
    d2b = np.zeros(len(G))
    for ni, datai in G.nodes(data=True):
        min_d = np.inf
        for nj, dataj in G.nodes(data=True):
            if dataj['boundary']:
                d = len(sp[ni][nj])
                if d < min_d:
                    min_d = d
                    d2b[ni] = d
    voronoi['dist2boundary'] = d2b

    d2b_index = {}
    for i, j in G.edges():
        d2b = int(voronoi.iloc[i]['dist2boundary'])
        if not d2b in d2b_index.keys():
            d2b_index[d2b] = dict(idx=[], jdx=[])
        
        d2b_index[d2b]['idx'].append(i)
        d2b_index[d2b]['jdx'].append(j)

    return d2b_index

def fluxes_per_angle(G, bins=12):
    angle_index = {}
    bins = np.linspace(0, 360, bins+1)
    binc = (bins[1:] + bins[:-1]) / 2
    for i, j, data in G.edges(data=True):
        angle = (data['angle'] + 360) % 360
        angle_bin = np.where(bins < angle)[0][-1]
        angle_bin = binc[angle_bin]

        if not angle_bin in angle_index.keys():
            angle_index[angle_bin] = dict(idx=[], jdx=[])

        angle_index[angle_bin]['idx'].append(i)
        angle_index[angle_bin]['jdx'].append(j)

    return angle_index


def flux_corr_per_angle(G, model_fluxes, gt_fluxes):
    df = dict(radar1=[], radar2=[], corr=[], angle=[])
    for i, j, data in G.edges(data=True):
        df['radar1'].append(G.nodes(data=True)[i]['radar'])
        df['radar2'].append(G.nodes(data=True)[j]['radar'])
        df['angle'].append((G.get_edge_data(i, j)['angle'] % 360))
        if not np.all(model_fluxes[i, j] == 0) and not np.all(gt_fluxes[i, j] == 0) and np.all(
                np.isfinite(gt_fluxes[i, j])):
            df['corr'].append(stats.pearsonr(gt_fluxes[i, j].flatten(), model_fluxes[i, j].flatten())[0])
        else:
            df['corr'].append(np.nan)
    df = pd.DataFrame(df)

    return df

def bin_metrics_fluxes(model_fluxes, gt_fluxes):
    mask = np.logical_and(np.isfinite(gt_fluxes), gt_fluxes != 0)

    model_bin = model_fluxes[mask].flatten() > 0
    gt_bin = gt_fluxes[mask].flatten() > 0

    tp = np.logical_and(model_bin, gt_bin).sum()
    fp = np.logical_and(model_bin, ~gt_bin).sum()
    fn = np.logical_and(~model_bin, gt_bin).sum()
    tn = np.logical_and(~model_bin, ~gt_bin).sum()

    summary = dict(precision = [], sensitivity = [], accuracy = [])
    summary['precision'].append(tp / (tp + fp))
    summary['sensitivity'].append(tp / (tp + fn))
    summary['accuracy'].append((tp + tn) / (tp + fp + tn + fn))

    return summary
    
def corr_on_graph(voronoi, G, gt_fluxes, model_fluxes):

    G_new = nx.DiGraph()
    G_new.add_nodes_from(list(G.nodes(data=True)))

    radars = voronoi.radar.values

    for i, ri in enumerate(radars):
        for j, rj in enumerate(radars):

            val = np.nanmean(gt_fluxes[i, j])

            if val > 0 and i != j:
                boundary1 = ('boundary' in ri) and ('boundary' in rj)
                boundary2 = voronoi.query(f'radar == "{ri}" or radar == "{rj}"')['boundary'].all()

                if not boundary1 and not boundary2:
                    model_fluxes_ij= model_fluxes[i, j]
                    gt_fluxes_ij = gt_fluxes[i, j]
                    mask = np.isfinite(gt_fluxes_ij)
                    corr = stats.pearsonr(model_fluxes_ij[mask].flatten(), gt_fluxes_ij[mask].flatten())[0]
                    G_new.add_edge(i, j, corr=corr)

    return G_new


def fluxes_on_graph(voronoi, G, fluxes, agg_func=np.nanmean, radars=None):

    G_new = nx.DiGraph()
    G_new.add_nodes_from(list(G.nodes(data=True)))

    all_radars = voronoi.radar.values

    for i, ri in enumerate(all_radars):
        for j, rj in enumerate(all_radars):
            if radars is None or ri in radars or rj in radars:

                val = agg_func(fluxes[i,j])

                if val > 0 and i != j:
                    boundary = ('boundary' in ri) and ('boundary' in rj)
                    #boundary2 = not voronoi.query(f'radar == "{ri}" or radar == "{rj}"')['observed'].any()

                    if not boundary: # and not boundary2:
                        G_new.add_edge(i, j, flux=val)

    return G_new

def total_net_flux(G, node):
    net_flux = 0
    for u, v, data in G.edges(data=True):
        if u == node:
            net_flux += data['flux']
        elif v == node:
            net_flux -= data['flux']
    return net_flux


if __name__ == "__main__":

    models = { 'FluxRGNN': ['final'] }

    source_sink = True
    fluxes = True

    trials = 5
    year = 2017
    season = 'fall'
    H_min = 24
    H_max = 24

    ext = ''
    #datasource = 'abm'
    #n_dummy = 25
    datasource = 'radar'
    n_dummy = 15

    base_dir = '/home/flipper/birdMigration'
    result_dir = osp.join(base_dir, 'results', datasource)
    # data_dir = osp.join(base_dir, 'data', 'raw', 'abm', season, str(year))
    data_dir = osp.join(base_dir, 'data', 'preprocessed', f'1H_voronoi_ndummy={n_dummy}',
                        datasource, season, str(year))

    if datasource == 'abm':
        dep = np.load(osp.join(data_dir, 'departing_birds.npy'))
        land = np.load(osp.join(data_dir, 'landing_birds.npy'))
        delta = dep - land

        with open(osp.join(data_dir, 'time.pkl'), 'rb') as f:
            abm_time = pickle.load(f)
        time_dict = {t: idx for idx, t in enumerate(abm_time)}


    voronoi = gpd.read_file(osp.join(base_dir, 'data', 'preprocessed',
                                     f'1H_voronoi_ndummy={n_dummy}',
                                     datasource, season, str(year), 'voronoi.shp'))

    radar_dict = voronoi.radar.to_dict()
    radar_dict = {v: k for k, v in radar_dict.items()}

    #inner_radars = voronoi.query('boundary == 0').radar.values
    #boundary_idx = voronoi.query('boundary == 1').index.values

    G = nx.read_gpickle(osp.join(base_dir, 'data', 'preprocessed', f'1H_voronoi_ndummy={n_dummy}',
                                 datasource, season, str(year), 'delaunay.gpickle'))


    def get_abm_data(data, datetime, radar, bird_scale=1):
        #print(radar, datetime)
        tidx = time_dict[pd.Timestamp(datetime)]
        ridx = radar_dict[radar]
        return data[tidx, ridx] / bird_scale

    inner_radars = voronoi.query('observed == 1').radar.values
    boundary_idx = voronoi.query('observed == 0').index.values
    
    if datasource == 'abm':
        gt_fluxes = np.load(osp.join(data_dir, 'outfluxes.npy'))


    for model, dirs in models.items():
        print(f'evaluate model components for {model}')
        for d in dirs:
            result_dir = osp.join(base_dir, 'results', datasource, model, f'test_{year}', d)
            results, cfg = load_cv_results(result_dir, ext=ext, trials=trials)
            model_fluxes = load_model_fluxes(result_dir, ext=ext, trials=trials)
            bird_scale = cfg['datasource']['bird_scale']
            output_dir = osp.join(result_dir, 'performance_evaluation', f'{H_min}-{H_max}')
            os.makedirs(output_dir, exist_ok=True)


            if source_sink:
                area_scale = results.area.max()

                #df = results.query(f'horizon == {H}')
                df = results.query(f'horizon <= {H_max} & horizon >= {H_min}')
                df = df[df.radar.isin(inner_radars)]
                df['month'] = pd.DatetimeIndex(df.datetime).month

                grouped = df.groupby(['radar', 'trial'])
                #grouped = df.groupby(['radar', 'trial', 'month'])
                #grouped = grouped[['source_km2', 'sink_km2']].aggregate(np.nansum)

                def get_net_source_sink(radar, trial):
                    if radar in inner_radars:
                        df = grouped.get_group((radar, trial)).aggregate(np.nansum)
                        return df['source_km2'] - df['sink_km2']
                    else:
                        return np.nan

                for t in df.trial.unique():
                    voronoi[f'net_source_sink_{t}'] = voronoi.apply(lambda row: 
                            get_net_source_sink(row.radar, t, month), axis=1)

                if datasource == 'abm':

                    print('evaluate source/sink')

                    corr_source = dict(month=[], trial=[], corr=[])
                    corr_sink = dict(month=[], trial=[], corr=[])
                
                    source_agg = dict(trial=[], gt=[], model=[])
                    sink_agg = dict(trial=[], gt=[], model=[])
                    for m in df.month.unique():
                        print(f'evaluate month {m}')
                        for t in df.trial.unique():
                            data = df.query(f'month == {m} & trial == {t}')

                            print(f'compute abm source/sink for trial {t}')

                            gt_source_km2 = []
                            gt_sink_km2 = []
                            for i, row in data.iterrows():
                                gt_source_km2.append(get_abm_data(dep, row['datetime'], row['radar']) / (row['area']/area_scale))
                                gt_sink_km2.append(get_abm_data(land, row['datetime'], row['radar']) / (row['area']/area_scale))
                        
                            #data.assign(gt_source_km2=np.stack(gt_source_km2))
                            #data.assign(gt_sink_km2=np.stack(gt_sink_km2))

                            data['gt_source_km2'] = gt_source_km2
                            data['gt_sink_km2'] = gt_sink_km2
                            #data.assign(gt_source_km2 = lambda row: get_abm_data(dep, row['datetime'], row['radar']) /
                            #                                        (row['area'] / area_scale))
                            #data.assign(gt_sink_km2 = lambda row: get_abm_data(land, row['datetime'], row['radar']) /
                            #                                      (row['area'] / area_scale))
                            # data['gt_source_km2'] = data.apply(
                            #    lambda row: get_abm_data(dep, row.datetime, row.radar) / (row.area / area_scale), axis=1)
                            # data['gt_sink_km2'] = data.apply(
                            #        lambda row: get_abm_data(land, row.datetime, row.radar) / (row.area / area_scale), axis=1)

                            print('aggregate source/sink over 24 H')
                            grouped = data.groupby(['seqID', 'radar'])
                            grouped = grouped[['gt_source_km2', 'source_km2', 'gt_sink_km2', 'sink_km2']].aggregate(
                                np.nansum).reset_index()

                            source_agg['gt'].extend(grouped.gt_source_km2.values)
                            source_agg['model'].extend(grouped.source_km2.values)
                            source_agg['trial'].extend([t]*len(grouped.gt_source_km2))
                            sink_agg['gt'].extend(grouped.gt_sink_km2.values)
                            sink_agg['model'].extend(grouped.sink_km2.values)
                            sink_agg['trial'].extend([t]*len(grouped.gt_sink_km2))

                            print('compute correlation')
                            corr = np.corrcoef(grouped.gt_source_km2.to_numpy(),
                                    grouped.source_km2.to_numpy())[0, 1]
                            corr_source['month'].append(m)
                            corr_source['trial'].append(t)
                            corr_source['corr'].append(corr)

                            corr = np.corrcoef(grouped.gt_sink_km2.to_numpy(),
                                           grouped.sink_km2.to_numpy())[0, 1]
                            corr_sink['month'].append(m)
                            corr_sink['trial'].append(t)
                            corr_sink['corr'].append(corr)

                    corr_source = pd.DataFrame(corr_source)
                    corr_sink = pd.DataFrame(corr_sink)

                    corr_source.to_csv(osp.join(output_dir, 'agg_source_corr_per_month_and_trial.csv'))
                    corr_sink.to_csv(osp.join(output_dir, 'agg_sink_corr_per_month_and_trial.csv'))


                    source_agg = pd.DataFrame(source_agg)
                    sink_agg = pd.DataFrame(sink_agg)

                    corr_source_all = dict(trial=[], corr=[])
                    corr_sink_all = dict(trial=[], corr=[])

                    for t in df.trial.unique():
                        source_agg_t = source_agg.query(f'trial == {t}')
                        corr_source_all['corr'].append(np.corrcoef(source_agg_t['gt'].to_numpy(),
                            source_agg_t.model.to_numpy())[0, 1])
                        corr_source_all['trial'].append(t)
                        sink_agg_t = sink_agg.query(f'trial == {t}')
                        corr_sink_all['corr'].append(np.corrcoef(sink_agg_t['gt'].to_numpy(),
                            sink_agg_t.model.to_numpy())[0, 1])
                        corr_sink_all['trial'].append(t)

                    corr_source_all = pd.DataFrame(corr_source_all)
                    corr_source_all.to_csv(osp.join(output_dir, 'agg_source_corr_per_trial.csv'))
                    corr_sink_all = pd.DataFrame(corr_sink_all)
                    corr_sink_all.to_csv(osp.join(output_dir, 'agg_sink_corr_per_trial.csv'))



            #df['gt_source_km2'] = df.apply(
            #    lambda row: get_abm_data(dep, row.datetime, row.radar) / (row.area / area_scale), axis=1)
            #df['gt_sink_km2'] = df.apply(
            #        lambda row: get_abm_data(land, row.datetime, row.radar) / (row.area / area_scale), axis=1)

            # corr per radar
            #gr = results[results.radar.isin(inner_radars)].dropna().groupby(['radar', 'trial'])
            #corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            #corr.to_csv(osp.join(output_dir, f'delta_corr_per_radar{ext}.csv'))

            # corr per gt bin
            
            #gr = results[results.radar.isin(inner_radars)].dropna().groupby(['seqID', 'radar', 'trial'])
            
            #activity = gr['gt'].aggregate(np.nanmean).reset_index()
            #results['activity_bin'] = pd.cut(results['gt_km2'] / birdscale, bins=np.linspace(0, 1, 4))
            
            """
            df['month'] = pd.DatetimeIndex(df.datetime).month

            grouped = df.groupby(['seqID', 'radar', 'trial', 'month'])
            grouped = grouped[['gt_source_km2', 'source_km2', 'gt_sink_km2', 'sink_km2']].aggregate(np.nansum).reset_index()
            #grouped = df.groupby(['trial', 'month'])
            corr_source = grouped[['gt_source_km2', 'source_km2']].corr().iloc[0::2, -1].reset_index()
            corr_source.to_csv(osp.join(output_dir, 'agg_source_corr_per_trial.csv'))

            corr_sink = grouped[['gt_sink_km2', 'sink_km2']].corr().iloc[0::2, -1].reset_index()
            corr_sink.to_csv(osp.join(output_dir, 'agg_sink_corr_per_trial.csv'))
            """

            #corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            #joint = corr.join(activity, how='outer', rsuffix='_r')
            #joint['activity_bin'] = pd.cut(joint['gt'].values, bins=np.arange(0, joint['gt'].max()+200, 200))
            #joint.to_csv(osp.join(output_dir, f'delta_corr_per_activity_bin{ext}.csv'))

            # corr per hour
            #gr = results[results.radar.isin(inner_radars)].dropna().groupby(['horizon', 'trial'])
            #corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            #corr.to_csv(osp.join(output_dir, f'delta_corr_per_hour{ext}.csv'))


            if fluxes:
                context = cfg['model']['context']
                horizon = cfg['model']['test_horizon']
                
                if datasource == 'abm':
                    # rearange abm fluxes to match model fluxes
                    gt_fluxes_H = []
                    gt_months = []
                    for s in sorted(results.groupby('seqID').groups.keys()):
                        df = results.query(f'seqID == {s}')
                        time = sorted(df.datetime.unique())
                        #t = time[context + H]
                        gt_months.append(pd.DatetimeIndex(time).month[context + 1])
                        agg_gt_fluxes = np.stack([gt_fluxes[time_dict[pd.Timestamp(time[context + h])]]
                                              for h in range(H_min, H_max+1)], axis=0).sum(0)
                        #gt_fluxes_H.append(gt_fluxes[time_dict[pd.Timestamp(t)]])
                        gt_fluxes_H.append(agg_gt_fluxes)


                    #context = cfg['model']['context']
                    #horizon = cfg['model']['test_horizon']
                    # gt_flux = np.stack([f[..., context: context + horizon] for
                    #             f in gt_flux_dict.values()], axis=-1)
                    gt_fluxes = np.stack(gt_fluxes_H, axis=-1)
                    gt_months = np.stack(gt_months, axis=-1)

                    # exclude "self-fluxes"
                    for i in range(gt_fluxes.shape[0]):
                        gt_fluxes[i, i] = np.nan

                    # exclude boundary to boundary fluxes
                    for i, j in it.product(boundary_idx, repeat=2):
                        gt_fluxes[i, j] = np.nan

                    # aggregate fluxes per sequence
                    # gt_flux_per_seq = gt_flux.sum(2)

                    # net fluxes
                    gt_net_fluxes = gt_fluxes - np.moveaxis(gt_fluxes, 0, 1)
                    # gt_net_flux_per_seq = gt_flux_per_seq - np.moveaxis(gt_flux_per_seq, 0, 1)

                    #gt_net_fluxes = gt_net_fluxes[..., :5]

                overall_corr = {}
                corr_per_radar = {}
                corr_per_hour = {}
                corr_influx = []
                corr_outflux = []
                corr_d2b = []
                corr_angles = []
                bin_fluxes = []
                # loop over all trials
                for t, model_fluxes_t in model_fluxes.items():

                    print(f'evaluate fluxes for trial {t}')
                    seqIDs = sorted(model_fluxes_t.keys())
                    #model_fluxes_t = np.stack([model_fluxes_t[s].detach().numpy()[..., H] for s in seqIDs], axis=-1)
                    model_fluxes_t = np.stack([model_fluxes_t[s].detach().numpy()[..., H_min:H_max+1].sum(-1) for s in seqIDs], axis=-1)
                    model_net_fluxes_t = model_fluxes_t - np.moveaxis(model_fluxes_t, 0, 1)
                    # model_flux_per_seq_t = model_flux_per_seq_t = model_flux_t.sum(2)
                    # model_net_flux_per_seq_t = model_flux_per_seq_t - np.moveaxis(model_flux_per_seq_t, 0, 1)
                    #model_net_fluxes_t = model_net_fluxes_t[..., :5]

                    if datasource == 'abm':
                        mask = np.isfinite(gt_net_fluxes)
                        overall_corr[t] = np.corrcoef(gt_net_fluxes[mask].flatten(),
                                                  model_net_fluxes_t[mask].flatten())[0, 1]


                        bin_results = bin_metrics_fluxes(model_net_fluxes_t, gt_net_fluxes)
                        bin_results = pd.DataFrame(bin_results)
                        bin_results['trial'] = t
                        bin_fluxes.append(bin_results)

                        corr_influx_per_month = dict(month=[], corr=[], trial=[])
                        corr_outflux_per_month = dict(month=[], corr=[], trial=[])
                        for m in np.unique(gt_months):
                            idx = np.where(gt_months == m)
                            gt_influx_m = np.nansum(gt_fluxes[..., idx], axis=1)
                            gt_outflux_m = np.nansum(gt_fluxes[..., idx], axis=0)

                            model_influx_m = np.nansum(model_fluxes_t[..., idx], axis=1)
                            model_outflux_m = np.nansum(model_fluxes_t[..., idx], axis=0)

                            mask = np.isfinite(gt_influx_m)
                            corr = np.corrcoef(gt_influx_m[mask].flatten(),
                                                  model_influx_m[mask].flatten())[0, 1]
                            corr_influx_per_month['corr'].append(corr)
                            corr_influx_per_month['month'].append(m)
                            corr_influx_per_month['trial'].append(t)

                            mask = np.isfinite(gt_outflux_m)
                            corr = np.corrcoef(gt_outflux_m[mask].flatten(),
                                           model_outflux_m[mask].flatten())[0, 1]
                            corr_outflux_per_month['corr'].append(corr)
                            corr_outflux_per_month['month'].append(m)
                            corr_outflux_per_month['trial'].append(t)
                        corr_influx.append(pd.DataFrame(corr_influx_per_month))
                        corr_outflux.append(pd.DataFrame(corr_outflux_per_month))

                        d2b_index = fluxes_per_dist2boundary(G, voronoi)
                        corr_per_d2b = dict(d2b=[], corr=[], trial=[])
                        for d2b, index in d2b_index.items():
                            model_net_fluxes_d2b = model_net_fluxes_t[index['idx'], index['jdx']]
                            gt_net_fluxes_d2b = gt_net_fluxes[index['idx'], index['jdx']]
                            mask = np.isfinite(gt_net_fluxes_d2b)
                            corr = stats.pearsonr(model_net_fluxes_d2b[mask].flatten(), gt_net_fluxes_d2b[mask].flatten())[0]
                            corr_per_d2b['d2b'].append(d2b)
                            corr_per_d2b['corr'].append(corr)
                            corr_per_d2b['trial'].append(t)
                        corr_d2b.append(pd.DataFrame(corr_per_d2b))


                        angle_index = fluxes_per_angle(G)
                        corr_per_angle = dict(angle=[], rad=[], corr=[], trial=[])
                        for angle, index in angle_index.items():
                            model_net_fluxes_a = model_net_fluxes_t[index['idx'], index['jdx']]
                            gt_net_fluxes_a = gt_net_fluxes[index['idx'], index['jdx']]
                            mask = np.isfinite(gt_net_fluxes_a)
                            corr = stats.pearsonr(model_net_fluxes_a[mask].flatten(), gt_net_fluxes_a[mask].flatten())[0]

                            corr_per_angle['angle'].append(angle)
                            corr_per_angle['rad'].append(angle / 360 * 2 * np.pi)
                            corr_per_angle['corr'].append(corr)
                            corr_per_angle['trial'].append(t)
                        corr_angles.append(pd.DataFrame(corr_per_angle))


                    #df_corr_d2b = flux_corr_per_dist2boundary(voronoi, model_net_fluxes_t, gt_net_fluxes)
                    #df_corr_d2b.to_csv(osp.join(output_dir, f'flux_corr_d2b_{t}.csv'))
                    #df_corr_d2b['trial'] = t
                    #corr_d2b.append(df_corr_d2b)


                    #fig, ax = plt.subplots(figsize=(4, 4))
                    #sb.boxplot(x='dist2boundary', y='corr', data=df_corr_d2b.dropna(), ax=ax, width=0.6, linewidth=2, color='lightgray')
                    #ax.set_xlabel('distance to boundary', fontsize=12)
                    #ax.set_ylabel('correlation coefficient', fontsize=12)
                    #plt.grid(color='gray', linestyle='--', alpha=0.5);
                    #fig.savefig(osp.join(output_dir, f'flux_corr_d2b_{t}.png'), bbox_inches='tight', dpi=200)



                    #df_corr_angles = flux_corr_per_angle(G, model_net_fluxes_t, gt_net_fluxes)
                    #df_corr_angles.to_csv(osp.join(output_dir, f'flux_corr_angles_{t}.csv'))
                    #df_corr_angles['trial'] = t

                    #bins = np.linspace(0, 360, 13)
                    #df_corr_angles['angle_bin'] = pd.cut(df_corr_angles['angle'], bins)
                    #df_corr_angles['angle_bin'] = df_corr_angles['angle_bin'].apply(lambda deg: (deg.left + deg.right) / 2)
                    #df_corr_angles['rad_bin'] = df_corr_angles['angle_bin'].apply(lambda deg: deg / 360 * 2 * np.pi)
                    #corr_angles.append(df_corr_angles)

                    #fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
                    #grouped = df_corr_angles.groupby('rad_bin')
                    #means = grouped.aggregate(np.nanmean).reset_index()
                    #stds = grouped.aggregate(np.nanstd).reset_index()
                    #bars = ax.bar(means['rad_bin'], means['corr'], width=0.3, bottom=0,
                    #              yerr=stds['corr'], ecolor='black', color='gray')

                    #max_corr = means['corr'].max() + stds['corr'].max()
                    #ax.set_rlim(0, max_corr)
                    #ax.set_theta_zero_location("N")
                    #ax.set_theta_direction(-1)
                    #fig.savefig(osp.join(output_dir, f'flux_corr_angles_{t}.png'), bbox_inches='tight', dpi=200)

                    if H_min == H_max:
                        agg_func = np.nansum
                    else:
                        agg_func = np.nanmean

                    G_model = fluxes_on_graph(voronoi, G, model_net_fluxes_t, agg_func=agg_func)
                    nx.write_gpickle(G_model, osp.join(output_dir, f'model_fluxes_{t}.gpickle'), protocol=4)
                    boundary_radars = voronoi.query('boundary == True').radar.values
                    G_boundary = fluxes_on_graph(voronoi, G, model_net_fluxes_t, agg_func=agg_func, radars=boundary_radars)
                    voronoi[f'net_flux_{t}'] = voronoi.apply(lambda row: total_net_flux(G_boundary, row.name), axis=1)
                    
                    if datasource == 'abm':
                        G_flux_corr = corr_on_graph(voronoi, G, gt_net_fluxes, model_net_fluxes_t)
                        nx.write_gpickle(G_flux_corr, osp.join(output_dir, f'flux_corr_{t}.gpickle'), protocol=4)

                        if t == 1:
                            G_gt = fluxes_on_graph(voronoi, G, gt_net_fluxes, agg_func=agg_func)
                            nx.write_gpickle(G_gt, osp.join(output_dir, 'gt_fluxes.gpickle'), protocol=4)


                voronoi.to_csv(osp.join(output_dir, 'voronoi_summary.csv'))

                
                if datasource == 'abm':
                    corr_d2b = pd.concat(corr_d2b)
                    corr_angles = pd.concat(corr_angles)
                    bin_fluxes = pd.concat(bin_fluxes)
                    corr_influx = pd.concat(corr_influx)
                    corr_outflux = pd.concat(corr_outflux)
                    corr_d2b.to_csv(osp.join(output_dir, 'agg_corr_d2b_per_trial.csv'))
                    corr_angles.to_csv(osp.join(output_dir, 'agg_corr_angles_per_trial.csv'))
                    bin_fluxes.to_csv(osp.join(output_dir, 'agg_bins_per_trial.csv'))
                    corr_influx.to_csv(osp.join(output_dir, 'agg_corr_influx_per_month.csv'))
                    corr_outflux.to_csv(osp.join(output_dir, 'agg_corr_outflux_per_month.csv'))

                    with open(osp.join(output_dir, 'agg_overall_corr.pickle'), 'wb') as f:
                        pickle.dump(overall_corr, f, pickle.HIGHEST_PROTOCOL)
                #
                # with open(osp.join(output_dir, 'corr_per_radar.pickle'), 'wb') as f:
                #     pickle.dump(corr_per_radar, f, pickle.HIGHEST_PROTOCOL)
                #
                # with open(osp.join(output_dir, 'corr_per_hour.pickle'), 'wb') as f:
                #     pickle.dump(corr_per_hour, f, pickle.HIGHEST_PROTOCOL)







