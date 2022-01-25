import torch
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import pickle5 as pickle
import itertools as it
import warnings
import datetime
warnings.filterwarnings("ignore")


def angle(x1, y1, x2, y2):
    # for coords given in lonlat crs
    y = y1 - y2
    x = x1 - x2
    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360
    return deg

def distance(x1, y1, x2, y2):
    # for coord1 and coord2 given in equidistant crs
    return np.linalg.norm(np.array([x1-x2, y1-y2])) / 10**3 # in kilometers


def rescale(features, min=None, max=None):
    if min is None:
        min = np.nanmin(features)
    if max is None:
        max = np.nanmax(features)
    if type(features) is not np.ndarray:
        features = np.array(features)

    rescaled = features - min
    if max != min:
        rescaled /= (max - min)
    return rescaled

def reshape(data, nights, mask, timesteps, use_nights=True):
    if use_nights:
        reshaped = reshape_nights(data, nights, mask, timesteps)
    else:
        reshaped = reshape_t(data, timesteps)
    return reshaped

def reshape_nights(data, nights, mask, timesteps):
    reshaped = [timeslice(data, night[0], mask, timesteps) for night in nights]
    reshaped = [d for d in reshaped if d.size > 0] # only use sequences that are fully available
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped

def reshape_t(data, timesteps):
    index = np.arange(0, data.shape[-1] - timesteps)
    reshaped = [data[..., t:t + timesteps] for t in index]
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped

def timeslice(data, start_night, mask, timesteps):
    data_night = data[..., start_night:]
    # remove hours during the day

    data_night = data_night[..., mask[start_night:]]
    if data_night.shape[-1] >= timesteps:
        data_night = data_night[..., :timesteps]
    else:
        data_night = np.empty(0)
    return data_night


class Normalization:
    def __init__(self, years, data_source, data_root, preprocessed_dirname, season='fall', t_unit='1H', **kwargs):
        self.root = data_root
        self.preprocessed_dirname = preprocessed_dirname
        self.data_source = data_source
        self.season = season
        self.t_unit = t_unit

        all_dfs = []
        for year in years:
            print('load year', year)
            dir = self.preprocessed_dir(year)
            if not osp.isdir(dir):
                # preprocessed data is not available
                print('Preprocessed data not available. Please run preprocessing script first.')

            # load features
            dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'dynamic_features.csv'))
            all_dfs.append(dynamic_feature_df)
        self.feature_df = pd.concat(all_dfs)

    def normalize(self, data, key):
        min = self.min(key)
        max = self.max(key)
        data = (data - min) / (max - min)
        return data

    def denormalize(self, data, key):
        min = self.min(key)
        max = self.max(key)
        data = data * (max - min) + min
        return data

    def min(self, key):
        return self.feature_df[key].dropna().min()

    def max(self, key):
        return self.feature_df[key].dropna().max()

    def absmax(self, key):
        return self.feature_df[key].dropna().abs().max()

    def root_min(self, key, root):
        root_transformed = self.feature_df[key].apply(lambda x: np.power(x, 1/root))
        return root_transformed.dropna().min()

    def root_max(self, key, root):
        root_transformed = self.feature_df[key].apply(lambda x: np.power(x, 1/root))
        return root_transformed.dropna().max()

    def preprocessed_dir(self, year):
        return osp.join(self.root, 'preprocessed', self.preprocessed_dirname,
                        self.data_source, self.season, str(year))

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

def angle(coord1, coord2):
    # coords should be in lonlat crs
    y = coord2[0] - coord1[0]
    x = coord2[1] - coord1[1]

    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360

    return deg

def compute_flux(dens, ff, dd, alpha, l=1):
    # compute number of birds crossing transect of length 'l' [km] and angle 'alpha' per hour
    mtr = dens * ff * np.cos(np.deg2rad(dd - alpha))
    flux = mtr * l * 3.6
    return flux


class RadarData(InMemoryDataset):

    def __init__(self, year, timesteps, preprocessed_dirname, processed_dirname,
                 transform=None, pre_transform=None, **kwargs):

        self.root = kwargs.get('data_root')
        self.preprocessed_dirname = preprocessed_dirname
        self.processed_dirname = processed_dirname
        if kwargs.get('device')['slurm']:
            self.sub_dir = str(datetime.datetime.utcnow())
        else:
            self.sub_dir = '' #kwargs.get('sub_dir', '')
        self.season = kwargs.get('season')
        self.year = str(year)
        self.timesteps = timesteps

        self.data_source = kwargs.get('data_source', 'radar')
        self.use_buffers = kwargs.get('use_buffers', False)
        self.bird_scale = kwargs.get('bird_scale', 1)
        self.env_points = kwargs.get('env_points', 100)
        self.radar_years = kwargs.get('radar_years', ['2015', '2016', '2017'])
        #self.env_vars = kwargs.get('env_vars', ['u', 'v'])
        self.env_vars = kwargs.get('env_vars', ['u', 'v', 'cc', 'tp', 'sp', 't2m', 'sshf'])
        #self.surface_vars = kwargs.get('surface_vars', [])
        self.multinight = kwargs.get('multinight', True)

        self.pref_dirs = kwargs.get('pref_dirs', {'spring': 58, 'fall': 223})
        self.wp_threshold = kwargs.get('wp_threshold', -0.5)
        self.root_transform = kwargs.get('root_transform', 0)
        self.missing_data_threshold = kwargs.get('missing_data_threshold', 0)

        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.normalize_dynamic = kwargs.get('normalize_dynamic', True)
        self.normalization = kwargs.get('normalization', None)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        self.max_distance = kwargs.get('max_distance', 216)
        self.t_unit = kwargs.get('t_unit', '1H')
        self.n_dummy_radars = kwargs.get('n_dummy_radars', 0)

        self.birds_per_km2 = kwargs.get('birds_per_km2', False)

        self.exclude = kwargs.get('exclude', [])

        self.compute_fluxes = kwargs.get('compute_fluxes', False)

        self.use_nights = kwargs.get('fixed_t0', True)
        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.default_rng(self.seed)
        self.data_perc = kwargs.get('data_perc', 1.0)
        self.importance_sampling = kwargs.get('importance_sampling', False)
        print('importance sampling', self.importance_sampling)


        super(RadarData, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(osp.join(self.processed_dir, self.info_file_name), 'rb') as f:
            self.info = pickle.load(f)

        print(f'processed data can be found here: {self.processed_dir}')

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def preprocessed_dir(self):
        return osp.join(self.root, 'preprocessed', self.preprocessed_dirname,
                        self.data_source, self.season, self.year)

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.processed_dirname,
                        self.data_source, self.season, self.year, self.sub_dir)

    @property
    def processed_file_names(self):
        return [f'data_timesteps={self.timesteps}.pt']

    @property
    def info_file_name(self):
        return f'info_timesteps={self.timesteps}.pkl'

    def download(self):
        pass


    def process(self):
        if not osp.isdir(self.preprocessed_dir):
            print('Preprocessed data not available. Please run preprocessing script first.')

        # load features
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        voronoi = pd.read_csv(osp.join(self.preprocessed_dir, 'static_features.csv'))

        if self.edge_type == 'voronoi':
            G = nx.read_gpickle(osp.join(self.preprocessed_dir, 'delaunay.gpickle'))
            edges = torch.tensor(list(G.edges()), dtype=torch.long)
            edge_index = edges.t().contiguous()
            n_edges = edge_index.size(1)
        else:
            G = nx.Graph()
            n_edges = 0
            edge_index = torch.zeros(0, dtype=torch.long)


        # boundary radars and boundary edges
        inner = voronoi['observed'].to_numpy()
        boundary2inner_edges = torch.tensor([(not inner[edge_index[0, idx]] and inner[edge_index[1, idx]])
                                            for idx in range(n_edges)])
        inner2boundary_edges = torch.tensor([(inner[edge_index[0, idx]] and not inner[edge_index[1, idx]])
                                             for idx in range(n_edges)])
        inner_edges = torch.tensor([(inner[edge_index[0, idx]] and inner[edge_index[1, idx]])
                                    for idx in range(n_edges)])
        boundary2boundary_edges = torch.tensor([(not inner[edge_index[0, idx]] and not inner[edge_index[1, idx]])
                                    for idx in range(n_edges)])

        reverse_edges = torch.zeros(n_edges, dtype=torch.long)
        for idx in range(n_edges):
            for jdx in range(n_edges):
                if (edge_index[:, idx] == torch.flip(edge_index[:, jdx], dims=[0])).all():
                    reverse_edges[idx] = jdx


        if self.edge_type == 'voronoi' and not self.birds_per_km2:
            input_col = 'birds'
        else:
            input_col = 'birds_km2'
        if self.use_buffers:
            input_col += '_from_buffer'

        # uv_cols = ['bird_u', 'bird_v', 'bird_direction', 'bird_speed']
        # uv_cols = [col for col in uv_cols if col in dynamic_feature_df.columns]

        # dynamic_feature_df['missing'] = dynamic_feature_df[input_col].isna() # remember which data was missing
        # #dynamic_feature_df[input_col].fillna(0, inplace=True)

        # # for each radar separately, replace missing observations with mean over all available nights
        # for radar in dynamic_feature_df.radar.unique():
        #     # bird densities
        #     mean = dynamic_feature_df.query(f'radar == "{radar}" & night == 1')[input_col].apply(np.nanmean)
        #     mean.fillna(0, inplace=True) # for dummy radars, fill with 0
        #     #mean = np.nan_to_num(mean) # for dummy radars, fill with 0
        #     dynamic_feature_df.iloc[(dynamic_feature_df.radar == radar).index][input_col].fillna(mean, inplace=True)
        #
        #     # bird velocities
        #     mean = dynamic_feature_df.query(f'radar == "{radar}" & night == 1')[uv_cols].apply(np.nanmean)
        #     mean.fillna(0, inplace=True)  # for dummy radars, fill with 0
        #     dynamic_feature_df.iloc[(dynamic_feature_df.radar == radar).index][uv_cols].fillna(mean, inplace=True)
        #
        # # set bird quantities to 0 during the day
        # for col in uv_cols + [input_col]:
        #     dynamic_feature_df[col] = dynamic_feature_df[col] * dynamic_feature_df['night']


        # apply root transform
        if self.root_transform > 0:
            dynamic_feature_df[input_col] = dynamic_feature_df[input_col].apply(
                                            lambda x: np.power(x, 1/self.root_transform))

        # normalize dynamic features
        cidx = ~dynamic_feature_df.columns.isin([input_col, 'birds_km2', 'birds_km2_from_buffer',
                                                 'bird_speed', 'bird_direction',
                                                 'bird_u', 'bird_v', 'u', 'v',
                                                 'radar', 'night', 'boundary',
                                                 'dusk', 'dawn', 'datetime', 'missing'])
        dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
                     lambda col: (col - self.normalization.min(col.name)) /
                                 (self.normalization.max(col.name) - self.normalization.min(col.name)), axis=0)

        if self.root_transform > 0:
            self.bird_scale = self.normalization.root_max(input_col, self.root_transform)
        else:
            self.bird_scale = self.normalization.max(input_col)

        print(f'bird scale = {self.bird_scale}')
        dynamic_feature_df[input_col] = dynamic_feature_df[input_col] / self.bird_scale
        if self.data_source == 'radar' and input_col != 'birds_km2':
            dynamic_feature_df['birds_km2'] = dynamic_feature_df['birds_km2'] / self.bird_scale

        uv_scale = self.normalization.absmax(['bird_u', 'bird_v']).max()
        print(f'uv scale = {uv_scale}')
        dynamic_feature_df[['bird_u', 'bird_v']] = dynamic_feature_df[['bird_u', 'bird_v']] / uv_scale

        if 'u' in self.env_vars and 'v' in self.env_vars:
            dynamic_feature_df[['u', 'v']] = dynamic_feature_df[['u', 'v']] / uv_scale

        # normalize static features
        coord_cols = ['x', 'y']
        xy_scale = voronoi[coord_cols].abs().max().max()
        voronoi[coord_cols] = voronoi[coord_cols] / xy_scale
        coords = voronoi[coord_cols].to_numpy()

        # check which radars are observed and which ones are dummy radars
        observed_idx = voronoi.observed.to_numpy()

        areas = voronoi[['area_km2']].apply(lambda col: col / col.max(), axis=0).to_numpy()
        if self.edge_type != 'voronoi':
            print('No graph structure used')
            areas = np.ones(areas.shape)
            edge_attr = torch.zeros(0)
        else:
            print('Use Voronoi tessellation')
            # get distances, angles and face lengths between radars
            distances = rescale(np.array([data['distance'] for i, j, data in G.edges(data=True)]))
            angles = rescale(np.array([data['angle'] for i, j, data in G.edges(data=True)]), min=0, max=360)
            delta_x = np.array([coords[j, 0] - coords[i, 0] for i, j in G.edges()])
            delta_y = np.array([coords[j, 1] - coords[i, 1] for i, j in G.edges()])

            face_lengths = rescale(np.array([data['face_length'] for i, j, data in G.edges(data=True)]))
            edge_attr = torch.stack([
                torch.tensor(distances, dtype=torch.float),
                torch.tensor(angles, dtype=torch.float),
                torch.tensor(delta_x, dtype=torch.float),
                torch.tensor(delta_y, dtype=torch.float),
                torch.tensor(face_lengths, dtype=torch.float)
            ], dim=1)


        # reorganize data
        target_col = input_col
        env_cols = self.env_vars
        acc_cols = ['acc_rain', 'acc_wind']

        def datetime_to_dayofyear(datetime):
            return pd.DatetimeIndex(datetime).dayofyear

        time = dynamic_feature_df.datetime.sort_values().unique()
        #dayofyear = pd.DatetimeIndex(dynamic_feature_df.datetime).dayofyear.values
        tidx = np.arange(len(time))
        #dayofyear = dayofyear / max(dayofyear)
        dynamic_feature_df['dayofyear'] = pd.DatetimeIndex(dynamic_feature_df.datetime).dayofyear
        dynamic_feature_df['dayofyear'] = dynamic_feature_df['dayofyear'] / dynamic_feature_df['dayofyear'].max()

        data = dict(inputs=[],
                    targets=[],
                    env=[],
                    acc=[],
                    nighttime=[],
                    missing=[],
                    bird_uv=[])

        if self.data_source == 'radar':
            data['speed'] = []
            data['direction'] = []
            data['birds_km2'] = []

        groups = dynamic_feature_df.groupby('radar')
        for name in voronoi.radar:
            df = groups.get_group(name).sort_values(by='datetime').reset_index(drop=True)
            data['inputs'].append(df[input_col].to_numpy())
            data['targets'].append(df[target_col].to_numpy())
            data['env'].append(df[env_cols].to_numpy().T)
            data['acc'].append(df[acc_cols].to_numpy().T)
            data['nighttime'].append(df.night.to_numpy())
            data['missing'].append(df.missing.to_numpy())
            data['bird_uv'].append(df[['bird_u', 'bird_v']].to_numpy().T)

            if self.data_source == 'radar':
                data['speed'].append(df.bird_speed.to_numpy())
                data['direction'].append(df.bird_direction.to_numpy())
                data['birds_km2'].append(df.birds_km2.to_numpy())

        for k, v in data.items():
            data[k] = np.stack(v, axis=0).astype(float)


        # find timesteps where it's night for all radars
        check_all = data['nighttime'].all(axis=0) # day/night mask
        # find timesteps where it's night for at least one radar
        check_any = data['nighttime'].any(axis=0)
        # also include timesteps immediately before dusk
        check_any = np.append(np.logical_or(check_any[:-1], check_any[1:]), check_any[-1])
        # dft = pd.DataFrame({'check_all': np.append(np.logical_and(check_all[:-1], check_all[1:]), False),
        #                     'check_any': np.append(np.logical_and(check_any[:-1], check_any[1:]), False),
        #                     'tidx': range(len(time))}, index=time)

        # group into nights
        groups = [list(g) for k, g in it.groupby(enumerate(check_all), key=lambda x: x[-1])]
        nights = [[item[0] for item in g] for g in groups if g[0][1]]

        # global_dusk_idx = [night[0] for night in nights]
        # global_dusk = np.zeros(tidx.shape)
        # global_dusk[global_dusk_idx] = 1

        if self.multinight:
            mask = np.ones(check_any.shape, dtype=bool)
        else:
            mask = check_all


        # reshape data into sequences
        for k, v in data.items():
            data[k] = reshape(v, nights, mask, self.timesteps, self.use_nights)

        tidx = reshape(tidx, nights, mask, self.timesteps, self.use_nights)
        #dayofyear = reshape(dayofyear, nights, mask, self.timesteps, self.use_nights)

        # remove sequences with too much missing data
        perc_missing = data['missing'][observed_idx].reshape(-1, data['missing'].shape[-1]).mean(0)
        valid_idx = perc_missing <= self.missing_data_threshold

        for k, v in data.items():
            data[k] = data[k][..., valid_idx]

        tidx = tidx[..., valid_idx]
        #dayofyear = dayofyear[..., valid_idx]



        if self.data_source == 'radar' and len(G.edges()) > 0:
            print('compute fluxes')
            fluxes = []
            mtr = []
            for i, j, e_data in G.edges(data=True):
                vid_i = data['birds_km2'][i]
                vid_j = data['birds_km2'][j]
                vid_i[np.isnan(vid_i)] = vid_j[np.isnan(vid_i)]
                vid_j[np.isnan(vid_j)] = vid_i[np.isnan(vid_j)]

                dd_i = data['direction'][i]
                dd_j = data['direction'][j]
                dd_i[np.isnan(dd_i)] = dd_j[np.isnan(dd_i)]
                dd_j[np.isnan(dd_j)] = dd_i[np.isnan(dd_j)]

                ff_i = data['speed'][i]
                ff_j = data['speed'][j]
                ff_i[np.isnan(ff_i)] = ff_j[np.isnan(ff_i)]
                ff_j[np.isnan(ff_j)] = ff_i[np.isnan(ff_j)]

                vid_interp = (vid_i + vid_j) / 2
                dd_interp = ((dd_i + 360) % 360 + (dd_j + 360) % 360) / 2
                ff_interp = (ff_i + ff_j) / 2
                length = e_data.get('face_length', 1)
                fluxes.append(compute_flux(vid_interp, ff_interp, dd_interp, e_data['angle'], length))
                mtr.append(compute_flux(vid_interp, ff_interp, dd_interp, e_data['angle'], 1))
            fluxes = np.stack(fluxes, axis=0)
            mtr = np.stack(mtr, axis=0)
        else:
            fluxes = np.zeros((len(G.edges()), data['inputs'].shape[1], data['inputs'].shape[2]))
            mtr = np.zeros((len(G.edges()), data['inputs'].shape[1], data['inputs'].shape[2]))

            data['direction'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))
            data['speed'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))

        data['direction'] = (data['direction'] + 360) % 360
        data['direction'] = rescale(data['direction'], min=0, max=360)

        min_speed = self.normalization.min('bird_speed') if self.data_source == 'radar' else 0
        max_speed = self.normalization.max('bird_speed') if self.data_source == 'radar' else 1
        data['speed'] = (data['speed'] - min_speed) / (max_speed - min_speed)


        # set bird densities during the day to zero
        data['inputs'] = data['inputs'] * data['nighttime']
        data['targets'] = data['targets'] * data['nighttime']

        # sample sequences
        if self.importance_sampling:
            print('use importance sampling')
            # reduce bias towards low migration intensity
            # compute total migration intensity per sequence
            agg = data['targets'].reshape(-1, data['targets'].shape[-1]).sum(0)

            # define importance weights
            thr = np.quantile(agg, 0.8)
            #s = agg.max() / 25
            #weight_func = lambda x: 1 / (1 + np.exp(-(x - thr) / s))
            weight_func = lambda x: 1 - np.exp(-x / thr)
            weights = weight_func(agg)
            weights /= weights.sum()

            # resample sequences according to importance weights
            n_seq = int(self.data_perc * valid_idx.sum())
            seq_index = self.rng.choice(np.arange(agg.size), n_seq, p=weights, replace=True)
            print('sampled seqIDs:', seq_index)
        else:
            # sample sequences uniformly
            n_seq = int(self.data_perc * valid_idx.sum())
            seq_index = self.rng.permutation(valid_idx.sum())[:n_seq]

        # create graph data objects per night
        data_list = [SensorData(edge_index=edge_index, reverse_edges=reverse_edges,
                          x=torch.tensor(data['inputs'][:, :, nidx], dtype=torch.float),
                          y=torch.tensor(data['targets'][:, :, nidx], dtype=torch.float),
                          coords=torch.tensor(coords, dtype=torch.float),
                          areas=torch.tensor(areas, dtype=torch.float),
                          boundary=torch.tensor(np.logical_not(inner), dtype=torch.bool),
                          env=torch.tensor(data['env'][..., nidx], dtype=torch.float),
                          acc=torch.tensor(data['acc'][..., nidx], dtype=torch.float),
                          boundary2inner_edges=boundary2inner_edges.bool(),
                          inner2boundary_edges=inner2boundary_edges.bool(),
                          boundary2boundary_edges=boundary2boundary_edges.bool(),
                          inner_edges=inner_edges.bool(),
                          edge_attr=edge_attr,
                          tidx=torch.tensor(tidx[:, nidx], dtype=torch.long),
                          #day_of_year=torch.tensor(dayofyear[:, nidx], dtype=torch.float),
                          local_night=torch.tensor(data['nighttime'][:, :, nidx], dtype=torch.bool),
                          missing=torch.tensor(data['missing'][:, :, nidx], dtype=torch.bool),
                          fluxes=torch.tensor(fluxes[:, :, nidx], dtype=torch.float),
                          mtr=torch.tensor(mtr[:, :, nidx], dtype=torch.float),
                          directions=torch.tensor(data['direction'][:, :, nidx], dtype=torch.float),
                          speeds=torch.tensor(data['speed'][:, :, nidx], dtype=torch.float),
                          bird_uv=torch.tensor(data['bird_uv'][..., nidx], dtype=torch.float))
                for nidx in seq_index]

        print(f'number of sequences = {len(data_list)}')

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)
        n_seq_discarded = valid_idx.size - valid_idx.sum()
        print(f'discarded {n_seq_discarded} sequences due to missing data')
        info = {'radars': voronoi.radar.values,
                'areas' : voronoi.area_km2.values,
                'env_vars': env_cols,
                 'timepoints': time,
                 'tidx': tidx,
                 'nights': nights,
                 'bird_scale': self.bird_scale,
                 'boundaries': voronoi['boundary'].to_dict(),
                 'root_transform': self.root_transform,
                 'n_seq_discarded': n_seq_discarded}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        if self.importance_sampling:
            np.save(osp.join(self.processed_dir, 'birds_per_seq.npy'), agg)
            np.save(osp.join(self.processed_dir, 'resampling_idx.npy'), seq_index)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SensorData(Data):
    def __init__(self, **kwargs):
        super(SensorData, self).__init__(**kwargs)

    def __inc__(self, key, value):
        # in mini-batches, increase edge indices in reverse_edges by the number of edges in the graph
        if key == 'reverse_edges':
            return self.num_edges
        else:
            return super().__inc__(key, value)



# prepare data for simple species distribution models that don't use neural nets
def get_training_data_gbt(dataset, timesteps, mask_daytime=False, use_acc_vars=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps):
            print(f'coord size = {seq.coords.detach().numpy().shape}')
            print(f'area size = {seq.areas.view(-1, 1).detach().numpy().shape}')
            print(f'env size = {seq.env[..., t].detach().numpy().shape}')
            features = [seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()]
            if use_acc_vars:
                features.append(seq.acc[..., t].detach().numpy())
            features = np.concatenate(features, axis=1) # shape (nodes, features)
            print(features.shape)
            X.append(features)
            y.append(seq.y[:, t])
            if mask_daytime:
                mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask.append(~seq.missing[:, t])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    mask = np.concatenate(mask, axis=0)

    return X, y, mask


def get_training_data_gam(dataset, timesteps, mask_daytime=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps):
            features = seq.env[:, -3:, t].detach().numpy()  # shape (nodes, features) where features are dayofyear, solarpos and solarpos_dt
            #doy = np.ones((env.shape[0], 1)) * seq.day_of_year[t].detach().numpy()
            #features = np.concatenate([env, doy], axis=-1)

            X.append(features)
            y.append(seq.y[:, t])
            if mask_daytime:
                mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask.append(~seq.missing[:, t])
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    mask = np.stack(mask, axis=0)

    return X, y, mask


def get_test_data_gbt(dataset, context, horizon, mask_daytime=False, use_acc_vars=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        X_night = []
        y_night = []
        mask_night = []
        for t in range(context, context+horizon):

            features = [seq.coords.detach().numpy(),
                 seq.areas.view(-1, 1).detach().numpy(),
                 seq.env[..., t].detach().numpy()]
            if use_acc_vars:
                features.append(seq.acc[..., t].detach().numpy())
            features = np.concatenate(features, axis=1) # shape (nodes, features)
            X_night.append(features)
            y_night.append(seq.y[:, t])
            if mask_daytime:
                mask_night.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask_night.append(~seq.missing[:, t])
        X.append(np.stack(X_night, axis=0)) # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0)) # shape (timesteps, nodes)
        mask.append(np.stack(mask_night, axis=0))  # shape (timesteps, nodes)

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    mask = np.stack(mask, axis=0)  # shape (nights, timesteps, nodes)

    return X, y, mask


def get_test_data_gam(dataset, context, horizon, mask_daytime=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        X_night = []
        y_night = []
        mask_night = []
        for t in range(context, context+horizon):
            features = seq.env[:, -3:, t].detach().numpy()  # shape (nodes, features)
            #doy = np.ones((env.shape[0], 1)) * seq.day_of_year[t].detach().numpy()
            #features = np.concatenate([env, doy], axis=-1)

            X_night.append(features)
            y_night.append(seq.y[:, t])
            if mask_daytime:
                mask_night.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask_night.append(~seq.missing[:, t])
        X.append(np.stack(X_night, axis=0)) # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0)) # shape (timesteps, nodes)
        mask.append(np.stack(mask_night, axis=0))  # shape (timesteps, nodes)

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    mask = np.stack(mask, axis=0)  # shape (nights, timesteps, nodes)

    return X, y, mask
