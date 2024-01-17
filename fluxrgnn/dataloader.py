import torch
from torch_geometric.data import Data, HeteroData, DataLoader, Dataset, InMemoryDataset
import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import pickle
import itertools as it
from omegaconf import DictConfig, OmegaConf
import warnings
import datetime
warnings.filterwarnings("ignore")


class Normalization:
    """Normalizer for neural network inputs"""

    def __init__(self, years, data_source, data_root, preprocessed_dirname, **kwargs):
        """
        Initialize normalizer with data for all given years.

        :param years: all years to use for normalization (should be training years)
        :param data_source: 'radar' or 'abm'
        :param data_root: directory containing all data
        :param preprocessed_dirname: name of directory containing data of interest (e.g. '1H_voronoi_ndummy=15')
        """

        self.root = data_root
        self.preprocessed_dirname = preprocessed_dirname
        self.data_source = data_source
        self.season = kwargs.get('season', 'fall')
        self.t_unit = kwargs.get('t_unit', '1H')

        all_features = []
        all_measurements = []
        print(years)
        for year in years:
            dir = self.preprocessed_dir(year)
            print(dir)
            if not osp.isdir(dir):
                print(f'Preprocessed data for year {year} not available. Please run preprocessing script first.')

            # load features
            dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'dynamic_features.csv'))
            all_features.append(dynamic_feature_df)

            measurement_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'measurements.csv'))
            all_measurements.append(measurement_df)

        self.feature_df = pd.concat(all_features)
        self.measurement_df = pd.concat(all_measurements)
        #self.measurement_df = pd.DataFrame()

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
        if key in self.measurement_df:
            return self.measurement_df[key].dropna().min()
        else:
            return self.feature_df[key].dropna().min()

    def max(self, key):
        if key in self.measurement_df:
            return self.measurement_df[key].dropna().max()
        else:
            return self.feature_df[key].dropna().max()

    def absmax(self, key):
        if key in self.measurement_df:
            return self.measurement_df[key].dropna().abs().max()
        else:
            return self.feature_df[key].dropna().abs().max()

    def quantile(self, key, q=0.99):
        if key in self.measurement_df:
            return self.measurement_df[key].dropna().quantile(q)
        else:
            return self.feature_df[key].dropna().quantile(q)

    def preprocessed_dir(self, year):
        return osp.join(self.root, 'preprocessed', self.preprocessed_dirname,
                        self.data_source, self.season, str(year))

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')


class SeasonalData(InMemoryDataset):

    def __init__(self, year, preprocessed_dirname, processed_dirname,
                 transform=None, pre_transform=None, **kwargs):

        self.root = kwargs.get('data_root')
        self.preprocessed_dirname = preprocessed_dirname
        self.processed_dirname = processed_dirname
        self.sub_dir = ''

        self.season = kwargs.get('season')
        self.year = str(year)
        self.data_source = kwargs.get('data_source', 'radar')
        self.use_buffers = kwargs.get('use_buffers', False)

        self.t_unit = kwargs.get('t_unit', '1H')
        self.birds_per_km2 = kwargs.get('birds_per_km2', False)
        self.exclude = kwargs.get('exclude', [])

        super(SeasonalData, self).__init__(self.root, transform, pre_transform)

        # run self.process() to generate dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

        # save additional info
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
        return [f'seasonal_data.pt']

    @property
    def info_file_name(self):
        return f'info.pkl'

    def download(self):
        pass

    def process(self):

        if not osp.isdir(self.preprocessed_dir):
            print('Preprocessed data not available. Please run preprocessing script first.')

        # load features
        # dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        measurement_df = pd.read_csv(osp.join(self.preprocessed_dir, 'measurements.csv'))
        cells = pd.read_csv(osp.join(self.preprocessed_dir, 'static_features.csv'))

        if not self.birds_per_km2:
            target_col = 'birds'
        else:
            target_col = 'birds_km2'
        if self.use_buffers:
            target_col += '_from_buffer'

        time = measurement_df.datetime.sort_values().unique()
        tidx = np.arange(len(time))

        print(tidx.min(), tidx.max())

        data = dict(targets=[], missing=[])

        groups = measurement_df.groupby('ID')
        for name in cells.ID:
            df = groups.get_group(name).sort_values(by='datetime').reset_index(drop=True)
            data['targets'].append(df[target_col].to_numpy())
            data['missing'].append(df.missing.to_numpy())

        for k, v in data.items():
            data[k] = np.stack(v, axis=0).astype(float)

        # create graph data objects per sequence
        data = [SensorData(
            # graph structure and edge features
            edge_index=torch.zeros(0, dtype=torch.long),

            # animal densities
            x=torch.tensor(data['targets'], dtype=torch.float),
            y=torch.tensor(data['targets'], dtype=torch.float),
            missing=torch.tensor(data['missing'], dtype=torch.bool),

            # time index of sequence
            tidx=torch.tensor(tidx, dtype=torch.long))]


        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)

        info = {'timepoints': time,
                'tidx': tidx}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])



class RadarData(InMemoryDataset):
    """
    Container for radar data (simulated or measured) that can be fed directly to FluxRGNN for training and testing.
    """

    def __init__(self, year, timesteps, preprocessed_dirname, processed_dirname,
                 transform=None, pre_transform=None, **kwargs):
        """
        Initialize data set.

        :param year: year of interest
        :param timesteps: number of timesteps per sequence (context + forecasting horizon)
        :param preprocessed_dirname: name of directory containing all preprocessed data of interest
        :param processed_dirname: name of directory to which the final dataset is written to
        :param transform: required for InMemoryDataset, but not used here
        :param pre_transform: required for InMemoryDataset, but not used here
        """

        self.root = kwargs.get('data_root')
        self.preprocessed_dirname = preprocessed_dirname
        self.processed_dirname = processed_dirname
        #if kwargs.get('device')['slurm']:
        #    # make sure to not overwrite data used by a parallel slurm process
        #    self.sub_dir = str(datetime.datetime.utcnow())
        #else:
        self.sub_dir = ''
        
        self.season = kwargs.get('season')
        self.year = str(year)
        self.timesteps = timesteps

        self.data_source = kwargs.get('data_source', 'radar')
        self.use_buffers = kwargs.get('use_buffers', False)
        self.bird_scale = kwargs.get('bird_scale', 1)
        self.env_points = kwargs.get('env_points', 100)
        self.radar_years = kwargs.get('radar_years', ['2015', '2016', '2017'])
        self.env_vars = kwargs.get('env_vars', ['dusk', 'dawn', 'night', 'dayofyear', 'solarpos', 'solarpos_dt'])

        self.pref_dirs = kwargs.get('pref_dirs', {'spring': 58, 'fall': 223})
        self.wp_threshold = kwargs.get('wp_threshold', -0.5)
        self.root_transform = kwargs.get('root_transform', 0)
        self.log_transform = kwargs.get('log_transform', 0)
        self.missing_data_threshold = kwargs.get('missing_data_threshold', 0)

        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.normalization = kwargs.get('normalization', None)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        self.t_unit = kwargs.get('t_unit', '1H')
        self.birds_per_km2 = kwargs.get('birds_per_km2', False)
        self.exclude = kwargs.get('exclude', [])

        self.use_nights = kwargs.get('fixed_t0', True)
        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.default_rng(self.seed)
        self.data_perc = kwargs.get('data_perc', 1.0)
        self.importance_sampling = kwargs.get('importance_sampling', False)


        super(RadarData, self).__init__(self.root, transform, pre_transform)

        # run self.process() to generate dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

        # save additional info
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
        """
        Prepare dataset consisting of a set of time sequences where for each sequence,
        a graph data object is created containing static and dynamic features for all sensors.
        """

        if not osp.isdir(self.preprocessed_dir):
            print('Preprocessed data not available. Please run preprocessing script first.')

        # load features
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        voronoi = pd.read_csv(osp.join(self.preprocessed_dir, 'static_features.csv'))

        if self.edge_type == 'voronoi':
            # load Delaunay triangulation with edge features
            G = nx.read_graphml(osp.join(self.preprocessed_dir, 'delaunay.graphml'), node_type=int)
            edges = torch.tensor(list(G.edges()), dtype=torch.long)
            edge_index = edges.t().contiguous()
            n_edges = edge_index.size(1)
        else:
            # don't use graph structure
            G = nx.Graph()
            n_edges = 0
            edge_index = torch.zeros(0, dtype=torch.long)

        print(f'number of nodes in graph = {G.number_of_nodes()}')

        # boundary radars and boundary edges
        boundary = voronoi['boundary'].to_numpy()
        boundary2inner_edges = torch.tensor([(boundary[edge_index[0, idx]] and not boundary[edge_index[1, idx]])
                                            for idx in range(n_edges)])
        inner2boundary_edges = torch.tensor([(not boundary[edge_index[0, idx]] and boundary[edge_index[1, idx]])
                                             for idx in range(n_edges)])
        inner_edges = torch.tensor([(not boundary[edge_index[0, idx]] and not boundary[edge_index[1, idx]])
                                    for idx in range(n_edges)])
        boundary2boundary_edges = torch.tensor([(boundary[edge_index[0, idx]] and boundary[edge_index[1, idx]])
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


        # normalize dynamic features
        if self.normalization is not None:
            dynamic_feature_df = self.normalize_dynamic(dynamic_feature_df, input_col)

        # normalize static features
        coord_cols = ['x', 'y']
        xy_scale = voronoi[coord_cols].abs().max().max()
        voronoi[coord_cols] = voronoi[coord_cols] / xy_scale
        coords = voronoi[coord_cols].to_numpy()

        areas = voronoi[['area_km2']].apply(lambda col: col / col.max(), axis=0).to_numpy()
        if self.edge_type != 'voronoi':
            print('No graph structure used')
            areas = np.ones(areas.shape)
            edge_attr = torch.zeros(0)
        else:
            print('Use Voronoi tessellation')
            # get distances, angles and face lengths between radars
            distances = rescale(np.array([data['distance'] for i, j, data in G.edges(data=True)]), min=0)
            angles = rescale(np.array([data['angle'] for i, j, data in G.edges(data=True)]), min=0, max=360)
            delta_x = np.array([coords[j, 0] - coords[i, 0] for i, j in G.edges()])
            delta_y = np.array([coords[j, 1] - coords[i, 1] for i, j in G.edges()])

            face_lengths = rescale(np.array([data['face_length'] for i, j, data in G.edges(data=True)]), min=0)
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

        time = dynamic_feature_df.datetime.sort_values().unique()
        tidx = np.arange(len(time))


        data = dict(inputs=[], targets=[], env=[], acc=[], nighttime=[], missing=[], bird_uv=[])

        if self.data_source == 'radar':
            data['speed'] = []
            data['direction'] = []
            data['birds_km2'] = []

        groups = dynamic_feature_df.groupby('ID')
        for id in voronoi.ID:
            df = groups.get_group(id).sort_values(by='datetime').reset_index(drop=True)
            data['inputs'].append(df[input_col].to_numpy())
            data['targets'].append(df[target_col].to_numpy())
            data['env'].append(df[env_cols].to_numpy().T)

            if len(set(acc_cols).intersection(set(df.columns))) == len(acc_cols):
                data['acc'].append(df[acc_cols].to_numpy().T)
            else:
                data['acc'].append(np.zeros((len(acc_cols), df.night.size)))

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
        check_all = data['nighttime'].all(axis=0)

        # group into nights
        groups = [list(g) for k, g in it.groupby(enumerate(check_all), key=lambda x: x[-1])]
        nights = [[item[0] for item in g] for g in groups if g[0][1]]

        # reshape data into sequences
        for k, v in data.items():
            data[k] = reshape(v, nights, np.ones(check_all.shape, dtype=bool), self.timesteps, self.use_nights)

        tidx = reshape(tidx, nights, np.ones(check_all.shape, dtype=bool), self.timesteps, self.use_nights)

        # remove sequences with too much missing data
        observed_idx = voronoi.observed.to_numpy()
        perc_missing = data['missing'][observed_idx].reshape(-1, data['missing'].shape[-1]).mean(0)
        valid_idx = perc_missing <= self.missing_data_threshold

        for k, v in data.items():
            data[k] = data[k][..., valid_idx]

        tidx = tidx[..., valid_idx]

        if self.data_source == 'radar' and len(G.edges()) > 0:
            # estimate fluxes between Voronoi cells based on MTR
            fluxes, mtr = self.compute_fluxes(data, G)
        else:
            fluxes = np.zeros((len(G.edges()), data['inputs'].shape[1], data['inputs'].shape[2]))
            mtr = np.zeros((len(G.edges()), data['inputs'].shape[1], data['inputs'].shape[2]))

            data['direction'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))
            data['speed'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))

        data['direction'] = (data['direction'] + 360) % 360
        data['direction'] = rescale(data['direction'], min=0, max=360)

        if self.data_source in ['radar', 'nexrad'] and self.normalization is not None:
            min_speed = self.normalization.min('bird_speed')
            max_speed = self.normalization.max('bird_speed')
            data['speed'] = (data['speed'] - min_speed) / (max_speed - min_speed)


        # set animal densities during the day to zero
        data['inputs'] = data['inputs'] * data['nighttime']
        data['targets'] = data['targets'] * data['nighttime']

        # sample sequences
        if self.importance_sampling:
            print('use importance sampling')
            n_seq, seq_index = self.importance_sampling(data, valid_idx)

        else:
            # sample sequences uniformly
            n_seq = int(self.data_perc * valid_idx.sum())
            seq_index = self.rng.permutation(valid_idx.sum())[:n_seq]

        # create graph data objects per sequence
        data_list = [SensorData(
                                # graph structure and edge features
                                edge_index=edge_index,
                                reverse_edges=reverse_edges,
                                boundary2inner_edges=boundary2inner_edges.bool(),
                                inner2boundary_edges=inner2boundary_edges.bool(),
                                boundary2boundary_edges=boundary2boundary_edges.bool(),
                                inner_edges=inner_edges.bool(),
                                edge_attr=edge_attr,

                                # static node features
                                coords=torch.tensor(coords, dtype=torch.float),
                                areas=torch.tensor(areas, dtype=torch.float),
                                boundary=torch.tensor(boundary, dtype=torch.bool),
                                ridx=torch.arange(len(voronoi.radar), dtype=torch.long),

                                # input animal densities
                                x=torch.tensor(data['inputs'][:, :, nidx], dtype=torch.float),
                                # target animal densities
                                y=torch.tensor(data['targets'][:, :, nidx], dtype=torch.float),

                                # dynamic node features
                                env=torch.tensor(data['env'][..., nidx], dtype=torch.float),
                                acc=torch.tensor(data['acc'][..., nidx], dtype=torch.float),
                                local_night=torch.tensor(data['nighttime'][:, :, nidx], dtype=torch.bool),
                                missing=torch.tensor(data['missing'][:, :, nidx], dtype=torch.bool),
                                fluxes=torch.tensor(fluxes[:, :, nidx], dtype=torch.float),
                                mtr=torch.tensor(mtr[:, :, nidx], dtype=torch.float),
                                directions=torch.tensor(data['direction'][:, :, nidx], dtype=torch.float),
                                speeds=torch.tensor(data['speed'][:, :, nidx], dtype=torch.float),
                                bird_uv=torch.tensor(data['bird_uv'][..., nidx], dtype=torch.float),

                                # time index of sequence
                                tidx=torch.tensor(tidx[:, nidx], dtype=torch.long),
                                )
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
                #'root_transform': self.root_transform,
                #'log_transform': self.log_transform,
                'n_seq_discarded': n_seq_discarded}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        if self.importance_sampling:
            np.save(osp.join(self.processed_dir, 'birds_per_seq.npy'), n_seq)
            np.save(osp.join(self.processed_dir, 'resampling_idx.npy'), seq_index)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def normalize_dynamic(self, dynamic_feature_df, input_col):
        """Normalize dynamic features to range between 0 and 1."""

        #if self.root_transform > 0:
        #    dynamic_feature_df[input_col] = dynamic_feature_df[input_col].apply(
        #        lambda x: np.power(x, 1 / self.root_transform))
        #elif self.log_transform:
        #    dynamic_feature_df[input_col] = dynamic_feature_df[input_col].apply(
        #        lambda x: np.log(x + 1e-5))

        cidx = ~dynamic_feature_df.columns.isin([input_col, 'birds_km2', 'birds_km2_from_buffer',
                                                 'bird_speed', 'bird_direction',
                                                 'bird_u', 'bird_v', 'u', 'v', 'u10', 'v10',
                                                 'radar', 'night', 'boundary',
                                                 'dusk', 'dawn', 'datetime', 'missing'])#,
                                                 #'cc', 'tp', 'acc_wind', 'acc_rain'])
        dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
            lambda col: (col - self.normalization.min(col.name)) /
                        (self.normalization.max(col.name) - self.normalization.min(col.name)), axis=0)

        #if self.root_transform > 0:
        #    self.bird_scale = self.normalization.root_max(input_col, self.root_transform)
        #else:
        #    self.bird_scale = self.normalization.max(input_col)
            # self.bird_scale = self.normalization.quantile(input_col, 0.99)
        
        #if self.log_transform:
        #    # TODO: find a more elegant way to do the log transform
        self.bird_scale = 1

        #dynamic_feature_df[input_col] = dynamic_feature_df[input_col] / self.bird_scale
        #if self.data_source == 'radar' and input_col != 'birds_km2':
        #    dynamic_feature_df['birds_km2'] = dynamic_feature_df['birds_km2'] / self.bird_scale

        uv_scale = max(self.normalization.absmax('bird_u'), 
                self.normalization.absmax('bird_v'))
        dynamic_feature_df[['bird_u', 'bird_v']] = dynamic_feature_df[['bird_u', 'bird_v']] / uv_scale

        if 'u' in self.env_vars and 'v' in self.env_vars:
            uv_scale = max(self.normalization.absmax('u'), self.normalization.absmax('v'))
            dynamic_feature_df[['u', 'v']] = dynamic_feature_df[['u', 'v']] / uv_scale

        if 'u10' in self.env_vars and 'v10' in self.env_vars:
            uv_scale = max(self.normalization.absmax('u10'), self.normalization.absmax('v10'))
            dynamic_feature_df[['u10', 'v10']] = dynamic_feature_df[['u10', 'v10']] / uv_scale

        #if 'tp' in self.env_vars:
        #    tp_scale = self.normalization.quantile('tp', 0.99)
        #    dynamic_feature_df['tp'] = dynamic_feature_df['tp'] / tp_scale

        if 'dayofyear' in self.env_vars:
            dynamic_feature_df['dayofyear'] /= self.normalization.max('dayofyear') # always use 365?

        #if 'acc_rain' in dynamic_feature_df.columns:
        #    acc_scale = self.normalization.quantile('acc_rain', 0.99)
        #    dynamic_feature_df['acc_rain'] = dynamic_feature_df['acc_rain'] / acc_scale

        #if 'acc_wind' in dynamic_feature_df.columns:
        #    acc_scale = self.normalization.quantile('acc_wind', 0.99)
        #    dynamic_feature_df['acc_wind'] = dynamic_feature_df['acc_wind'] / acc_scale

        return dynamic_feature_df

    def compute_fluxes(self, data, G):
        """Estimate fluxes across Voronoi faces based on radar MTR"""

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

        return fluxes, mtr

    def importance_sampling(self, data, valid_idx):
        """
        Use importance sampling to reduce bias towards low migration intensity.

        Importance weights are computed based on total migration intensities per sequence.
        """

        agg = data['targets'].reshape(-1, data['targets'].shape[-1]).sum(0)

        # define importance weights
        thr = np.quantile(agg, 0.8)
        weight_func = lambda x: 1 - np.exp(-x / thr)
        weights = weight_func(agg)
        weights /= weights.sum()

        # resample sequences according to importance weights
        n_seq = int(self.data_perc * valid_idx.sum())
        seq_index = self.rng.choice(np.arange(agg.size), n_seq, p=weights, replace=True)

        return n_seq, seq_index


class RadarHeteroData(InMemoryDataset):
    """
    Container for radar data (simulated or measured) that can be fed directly to FluxRGNN for training and testing.
    """

    def __init__(self, year, timesteps, preprocessed_dirname, processed_dirname,
                 transform=None, pre_transform=None, **kwargs):
        """
        Initialize data set.

        :param year: year of interest
        :param timesteps: number of timesteps per sequence (context + forecasting horizon)
        :param preprocessed_dirname: name of directory containing all preprocessed data of interest
        :param processed_dirname: name of directory to which the final dataset is written to
        :param transform: required for InMemoryDataset, but not used here
        :param pre_transform: required for InMemoryDataset, but not used here
        """

        self.root = kwargs.get('data_root')
        self.preprocessed_dirname = preprocessed_dirname
        self.processed_dirname = processed_dirname

        self.season = kwargs.get('season')
        self.year = str(year)
        self.timesteps = timesteps

        self.data_source = kwargs.get('data_source', 'radar')
        self.env_vars = kwargs.get('env_vars', ['dusk', 'dawn', 'night', 'dayofyear', 'solarpos', 'solarpos_dt'])

        self.wp_threshold = kwargs.get('wp_threshold', -0.5)
        self.missing_data_threshold = kwargs.get('missing_data_threshold', 0)

        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.normalization = kwargs.get('normalization', None)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        self.t_unit = kwargs.get('t_unit', '1H')
        self.birds_per_km2 = kwargs.get('birds_per_km2', True)
        self.exclude = kwargs.get('exclude', [])

        self.use_nights = kwargs.get('fixed_t0', True)
        self.tidx_start = kwargs.get('tidx_start', 0)
        self.tidx_step = kwargs.get('tidx_step', 1)

        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.default_rng(self.seed)
        self.data_perc = kwargs.get('data_perc', 1.0)

        print(kwargs)

        super(RadarHeteroData, self).__init__(self.root, transform, pre_transform)

        # run self.process() to generate dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

        # save additional info
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
                        self.data_source, self.season, self.year)

    @property
    def processed_file_names(self):
        return [f'data_timesteps={self.timesteps}.pt']

    @property
    def info_file_name(self):
        return f'info_timesteps={self.timesteps}.pkl'

    def download(self):
        pass

    def process(self):
        """
        Prepare dataset consisting of a set of time sequences where for each sequence,
        a graph data object is created containing static and dynamic features for all sensors.
        """

        if not osp.isdir(self.preprocessed_dir):
            print('Preprocessed data not available. Please run preprocessing script first.')

        # load features
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        measurement_df = pd.read_csv(osp.join(self.preprocessed_dir, 'measurements.csv'))
        cells = pd.read_csv(osp.join(self.preprocessed_dir, 'static_features.csv'))

        # relationship between cells and radars
        if self.edge_type in ['voronoi', 'none']:
            cell_to_radar_edge_index = torch.stack([torch.arange(len(cells)), torch.arange(len(cells))], dim=0).contiguous()
            cell_to_radar_weights = torch.ones(len(cells))
            
            radar_to_cell_edge_index = torch.stack([torch.arange(len(cells)), torch.arange(len(cells))], dim=0).contiguous()
            radar_to_cell_weights = torch.ones(len(cells))
        else:
            cell_to_radar_edges = pd.read_csv(osp.join(self.preprocessed_dir, 'cell_to_radar_edges.csv'))
            radar_to_cell_edges = pd.read_csv(osp.join(self.preprocessed_dir, 'radar_to_cell_edges.csv'))

            cell_to_radar_edge_index = torch.tensor(cell_to_radar_edges[['cidx', 'ridx']].values, dtype=torch.long)
            cell_to_radar_edge_index = cell_to_radar_edge_index.t().contiguous()
            cell_to_radar_weights = torch.tensor(cell_to_radar_edges['weight'].values, dtype=torch.float)

            radar_to_cell_edge_index = torch.tensor(radar_to_cell_edges[['ridx', 'cidx']].values, dtype=torch.long)
            radar_to_cell_edge_index = radar_to_cell_edge_index.t().contiguous()
            radar_to_cell_weights = torch.tensor(radar_to_cell_edges['weight'].values, dtype=torch.float)

        graph_file = osp.join(self.preprocessed_dir, 'delaunay.graphml')
        if osp.isfile(graph_file) and not self.edge_type == 'none':
            # load Delaunay triangulation with edge features
            G = nx.read_graphml(graph_file, node_type=int)
            edges = torch.tensor(list(G.edges()), dtype=torch.long)
            edge_index = edges.t().contiguous()
            n_edges = edge_index.size(1)
        else:
            # don't use graph structure
            G = nx.Graph()
            n_edges = 0
            edge_index = torch.zeros(0, dtype=torch.long)

        print(f'number of nodes in graph = {G.number_of_nodes()}')

        # boundary cells and boundary edges
        boundary = cells['boundary'].to_numpy()
        boundary2inner_edges = torch.tensor([(boundary[edge_index[0, idx]] and not boundary[edge_index[1, idx]])
                                             for idx in range(n_edges)])
        inner2boundary_edges = torch.tensor([(not boundary[edge_index[0, idx]] and boundary[edge_index[1, idx]])
                                             for idx in range(n_edges)])
        inner_edges = torch.tensor([(not boundary[edge_index[0, idx]] and not boundary[edge_index[1, idx]])
                                    for idx in range(n_edges)])
        boundary2boundary_edges = torch.tensor([(boundary[edge_index[0, idx]] and boundary[edge_index[1, idx]])
                                                for idx in range(n_edges)])

        reverse_edges = torch.zeros(n_edges, dtype=torch.long)
        for idx in range(n_edges):
            for jdx in range(n_edges):
                if (edge_index[:, idx] == torch.flip(edge_index[:, jdx], dims=[0])).all():
                    reverse_edges[idx] = jdx

        if self.birds_per_km2:
            target_col = 'birds_km2'
        else:
            target_col = 'birds'

        print(f'target col = {target_col}')

        # normalize dynamic features
        if self.normalization is not None:
            dynamic_feature_df = self.normalize_dynamic(dynamic_feature_df)
            measurement_df = self.normalize_dynamic(measurement_df)

        # normalize static features
        coord_cols = ['x', 'y']
        xy_scale = cells[coord_cols].abs().max().max()
        cells[coord_cols] = cells[coord_cols] / xy_scale
        local_pos = cells[coord_cols].to_numpy()

        lonlat_encoding = np.stack([np.sin(cells['lon'].to_numpy()), 
                                    np.cos(cells['lon'].to_numpy()),
                                    np.sin(cells['lat'].to_numpy()),
                                    np.cos(cells['lat'].to_numpy())], axis=1)

        areas = cells[['area_km2']].apply(lambda col: col / col.max(), axis=0).to_numpy()
        area_scale = cells['area_km2'].max() # [km^2]
        length_scale = np.sqrt(area_scale) # [km]

        if self.edge_type == 'none':
            print('No graph structure used')
            areas = np.ones(areas.shape)
            edge_attr = torch.zeros(0)
            n_ij = torch.zeros(0)
            face_lengths = torch.zeros(0)
        else:
            print('Use tessellation')
            # get distances, angles and face lengths between radars
            # distances = rescale(np.array([data['distance'] for i, j, data in G.edges(data=True)]), min=0)
            distances = np.array([data['distance'] for i, j, data in G.edges(data=True)]) / (length_scale * 1e3)
            angles = rescale(np.array([data['angle'] for i, j, data in G.edges(data=True)]), min=0, max=360)
            delta_x = np.array([local_pos[j, 0] - local_pos[i, 0] for i, j in G.edges()])
            delta_y = np.array([local_pos[j, 1] - local_pos[i, 1] for i, j in G.edges()])
            n_ij = np.stack([delta_x, delta_y], axis=1)
            n_ij = n_ij / np.linalg.norm(n_ij, ord=2, axis=1).reshape(-1, 1) # normalize to unit vectors

            face_lengths = np.array([data['face_length'] for i, j, data in G.edges(data=True)]) / (length_scale * 1e3)
            print(f'max face length: {face_lengths.max()}, min face length: {face_lengths.min()}')
            print(f'max distance: {distances.max()}, min distance: {distances.min()}')
            print(f'max area: {areas.max()}, min area: {areas.min()}')

            edge_attr = torch.stack([
                torch.tensor(rescale(distances, min=0), dtype=torch.float),
                torch.tensor(angles, dtype=torch.float),
                torch.tensor(delta_x, dtype=torch.float),
                torch.tensor(delta_y, dtype=torch.float),
                torch.tensor(rescale(face_lengths, min=0), dtype=torch.float)
            ], dim=1)


        time = dynamic_feature_df.datetime.sort_values().unique()
        tidx = np.arange(len(time))

        data = {'env': [], 'cell_nighttime': [], 'radar_nighttime': [], target_col: [], 'bird_uv': [], 'missing': []}

        # process dynamic cell features
        for cid, group_df in dynamic_feature_df.groupby('ID'):
            df = group_df.sort_values(by='datetime').reset_index(drop=True)
            data['env'].append(df[self.env_vars].to_numpy().T)
            data['cell_nighttime'].append(df.night.to_numpy())

        # process radar measurements
        radar_ids = measurement_df.ID.unique()
        for rid, group_df in measurement_df.groupby('ID'):
            group_df = group_df.sort_values(by='datetime').reset_index(drop=True)
            data[target_col].append(group_df[target_col].to_numpy())
            data['bird_uv'].append(group_df[['bird_u', 'bird_v']].to_numpy().T)
            data['missing'].append(group_df['missing'].to_numpy())
            data['radar_nighttime'].append(group_df.night.to_numpy())

        for k, v in data.items():
            data[k] = np.stack(v, axis=0).astype(float)
            print(k, data[k].shape)


        if self.timesteps == 'all':
            # use a single sequence for the entire year
            for k, v in data.items():
                print(f'{k}: {v.shape}')
                data[k] = np.expand_dims(v, axis=-1)
            tidx = np.expand_dims(tidx, axis=-1)
        else:
            # find timesteps where it's night for all cells
            check_all = data['cell_nighttime'].all(axis=0)

            # group into nights
            groups = [list(g) for k, g in it.groupby(enumerate(check_all), key=lambda x: x[-1])]
            nights = [[item[0] for item in g] for g in groups if g[0][1]]

            # reshape data into sequences
            for k, v in data.items():
                data[k] = reshape(v, nights, np.ones(check_all.shape, dtype=bool), self.timesteps, self.use_nights,
                                  self.tidx_start, self.tidx_step)

            tidx = reshape(tidx, nights, np.ones(check_all.shape, dtype=bool), self.timesteps, self.use_nights,
                           self.tidx_start, self.tidx_step)


        # remove sequences with too much missing data
        perc_missing = data['missing'].reshape(-1, data['missing'].shape[-1]).mean(0)
        print(perc_missing)
        valid_idx = perc_missing <= self.missing_data_threshold
        #valid_idx = np.ones(tidx.shape[-1], dtype='int')

        for k, v in data.items():
            data[k] = data[k][..., valid_idx]
        tidx = tidx[..., valid_idx]
        print(tidx.min(), tidx.max())


        # sample sequences uniformly
        if self.data_perc < 1.0:
            n_seq = int(self.data_perc * valid_idx.sum())
            seq_index = self.rng.permutation(valid_idx.sum())[:n_seq]
        else:
            seq_index = np.arange(tidx.shape[-1])
            print(seq_index)

        # Delaunay triangulation features
        cell2cell_edges = {
            'edge_index': edge_index,
            'reverse_edges': reverse_edges,
            'boundary2inner_edges': boundary2inner_edges.bool(),
            'inner2boundary_edges': inner2boundary_edges.bool(),
            'boundary2boundary_edges': boundary2boundary_edges.bool(),
            'inner_edges': inner_edges.bool(),
            'edge_attr': edge_attr,
            'edge_normals': torch.tensor(n_ij, dtype=torch.float),
            'edge_face_lengths': torch.tensor(face_lengths, dtype=torch.float)
        }

        # observation model structure
        cell2radar_edges = {
            'edge_index': cell_to_radar_edge_index,
            'edge_weight': cell_to_radar_weights
        }

        # interpolation model structure
        radar2cell_edges = {
            'edge_index': radar_to_cell_edge_index,
            'edge_weight': radar_to_cell_weights
        }

        # create graph data objects per sequence
        data_list = []
        for idx in seq_index:

            cell_data = {
                # static cell features
                'pos': torch.tensor(local_pos, dtype=torch.float),
                'coords': torch.tensor(lonlat_encoding, dtype=torch.float),
                'areas': torch.tensor(areas, dtype=torch.float),
                'boundary': torch.tensor(boundary, dtype=torch.bool),
                'cidx': torch.arange(len(cells), dtype=torch.long),

                # dynamic cell features
                'env': torch.tensor(data['env'][..., idx], dtype=torch.float),
                'local_night': torch.tensor(data['cell_nighttime'][..., idx], dtype=torch.bool),
                'tidx': torch.tensor(tidx[:, idx], dtype=torch.long)
            }

            if self.edge_type in ['voronoi', 'none']:
                cell_data['x'] = torch.tensor(data[target_col][..., idx], dtype=torch.float)
                cell_data['bird_uv'] = torch.tensor(data['bird_uv'][..., idx], dtype=torch.float)

            radar_data = {
                # static radar features
                'ridx': torch.arange(len(radar_ids), dtype=torch.long),

                # dynamic radar features
                'x': torch.tensor(data[target_col][..., idx], dtype=torch.float),
                'missing': torch.tensor(data['missing'][..., idx], dtype=torch.bool),
                'local_night': torch.tensor(data['radar_nighttime'][..., idx], dtype=torch.bool),
                'bird_uv': torch.tensor(data['bird_uv'][..., idx], dtype=torch.float),
                'tidx': torch.tensor(tidx[:, idx], dtype=torch.long)
            }

            # heterogeneous graph with two types of nodes: cells and radars
            data_list.append(SensorHeteroData(
                cell = cell_data,
                radar = radar_data,
                cell__to__cell = cell2cell_edges,
                cell__to__radar = cell2radar_edges,
                radar__to__cell = radar2cell_edges
            ))


        print(f'number of sequences = {len(data_list)}')

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)
        n_seq_discarded = valid_idx.size - valid_idx.sum()
        print(f'discarded {n_seq_discarded} sequences due to missing data')

        info = {
                'env_vars': self.env_vars,
                'timepoints': time,
                'tidx': tidx,
                'n_seq_discarded': n_seq_discarded
        }

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def normalize_dynamic(self, dynamic_feature_df):
        """Normalize dynamic features to range between 0 and 1."""

        cidx = ~dynamic_feature_df.columns.isin(['birds', 'birds_km2', 'birds_km2_from_buffer',
                                                 'bird_speed', 'bird_direction',
                                                 'bird_u', 'bird_v', 'u', 'v', 'u10', 'v10',
                                                 'radar', 'ID', 'night', 'boundary',
                                                 'dusk', 'dawn', 'datetime', 'missing'])

        dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
            lambda col: (col - self.normalization.min(col.name)) /
                        (self.normalization.max(col.name) - self.normalization.min(col.name)), axis=0)

        if 'bird_u' in dynamic_feature_df and 'bird_v' in dynamic_feature_df:
            uv_scale = max(self.normalization.absmax('bird_u'), self.normalization.absmax('bird_v'))
            dynamic_feature_df[['bird_u', 'bird_v']] = dynamic_feature_df[['bird_u', 'bird_v']] / uv_scale

        if 'u' in dynamic_feature_df and 'v' in dynamic_feature_df:
            uv_scale = max(self.normalization.absmax('u'), self.normalization.absmax('v'))
            dynamic_feature_df[['u', 'v']] = dynamic_feature_df[['u', 'v']] / uv_scale

        if 'u10' in dynamic_feature_df and 'v10' in dynamic_feature_df:
            uv_scale = max(self.normalization.absmax('u10'), self.normalization.absmax('v10'))
            dynamic_feature_df[['u10', 'v10']] = dynamic_feature_df[['u10', 'v10']] / uv_scale

        if 'dayofyear' in dynamic_feature_df:
            dynamic_feature_df['dayofyear'] /= self.normalization.max('dayofyear')  # always use 365?

        return dynamic_feature_df


class SensorData(Data):
    """Graph data object where reverse edges are treated the same as the regular 'edge_index'."""

    def __init__(self, **kwargs):
        super(SensorData, self).__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # in mini-batches, increase edge indices in reverse_edges by the number of edges in the graph
        if key == 'reverse_edges':
            return self.num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

class SensorHeteroData(HeteroData):
    """Heterogeneous graph data object where reverse edges are treated the same as the regular 'edge_index'."""

    def __init__(self, **kwargs):
        super(SensorHeteroData, self).__init__(**kwargs)

    def __inc__(self, key, value, store=None, *args, **kwargs):
        # in mini-batches, increase edge indices in reverse_edges by the number of edges in the graph
        if key == 'reverse_edges':
            return store.size(0) #torch.tensor(store.size()) #.view(2, 1)
        else:
            return super().__inc__(key, value, store, *args, **kwargs)


def load_dataset(cfg: DictConfig, output_dir: str, training: bool, transform=None):
    """
    Load training or testing data, initialize normalizer, setup and save configuration

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which config is written to
    :return: pytorch Dataset
    """
    context = cfg.model.get('context', 0)
    if context == 0 and not training:
        context = cfg.model.get('test_context', 0)

    # seq_len = context + (cfg.model.horizon if training else cfg.model.test_horizon)
    seq_len = context + max(cfg.model.get('horizon', 1), cfg.model.get('test_horizon')) + cfg.datasource.get('tidx_step', 1) - 1
    seed = cfg.seed + cfg.get('job_id', 0)

    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.datasource.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_log={cfg.model.use_log_transform}_' \
                        f'pow={cfg.model.get("pow_exponent", 1.0)}_maxT0={cfg.model.max_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_ndummy={cfg.datasource.n_dummy_radars}_dataperc={cfg.data_perc}'
    data_dir = osp.join(cfg.device.root, 'data')

    # if cfg.model.birds_per_km2:
    #     input_col = 'birds_km2'
    # else:
    #     if cfg.datasource.use_buffers:
    #         input_col = 'birds_from_buffer'
    #     else:
    #         input_col = 'birds'

    if training:
        # initialize normalizer
        years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
        normalization = Normalization(years, cfg.datasource.name, data_dir, preprocessed_dirname, **cfg)

        # complete config and write it together with normalizer to disk
        # cfg.datasource.bird_scale = float(normalization.max(input_col))
        cfg.model_seed = seed
        with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(config=cfg, f=f)
        with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
            pickle.dump(normalization, f)
    else:
        years = [cfg.datasource.test_year]
        norm_path = osp.join(cfg.get('model_dir', output_dir), 'normalization.pkl')
        if osp.isfile(norm_path):
            with open(norm_path, 'rb') as f:
                normalization = pickle.load(f)
        else:
            normalization = None

    # load training/validation/test data
    # data = [RadarData(year, seq_len, preprocessed_dirname, processed_dirname,
    #                              **cfg, **cfg.model,
    #                              data_root=data_dir,
    #                              data_source=cfg.datasource.name,
    #                              normalization=normalization,
    #                              env_vars=cfg.datasource.env_vars,
    #                              transform=transform
    #                              )
    #         for year in years]

    data = [RadarHeteroData(year, seq_len, preprocessed_dirname, processed_dirname,
                      **cfg, **cfg.model,
                      data_root=data_dir,
                      data_source=cfg.datasource.name,
                      normalization=normalization,
                      env_vars=cfg.datasource.env_vars,
                      tidx_start=cfg.datasource.get('tidx_start', 0),
                      tidx_step=cfg.datasource.get('tidx_step', 1),
                      transform=transform
                      )
            for year in years]

    # return data, input_col, context, seq_len
    return data, context, seq_len


def load_xgboost_dataset(cfg: DictConfig, output_dir: str, transform=None):
    """
    Load training data for XGBoost model, for which no time sequences are needed.

    :param cfg: DictConfig specifying model, data and training details
    :param output_dir: directory to which config is written to
    :return: pytorch Dataset
    """

    seq_len = 'all'
    seed = cfg.seed + cfg.get('job_id', 0)

    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.datasource.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_log={cfg.model.use_log_transform}_' \
                        f'pow={cfg.model.pow_exponent}_scale={cfg.model.scale}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_ndummy={cfg.datasource.n_dummy_radars}_dataperc={cfg.data_perc}'
    data_dir = osp.join(cfg.device.root, 'data')

    # initialize normalizer
    years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
    normalization = Normalization(years, cfg.datasource.name, data_dir, preprocessed_dirname, **cfg)

    # complete config and write it together with normalizer to disk
    cfg.model_seed = seed
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)


    data = [RadarHeteroData(year, seq_len, preprocessed_dirname, processed_dirname,
                      **cfg, **cfg.model,
                      data_root=data_dir,
                      data_source=cfg.datasource.name,
                      normalization=normalization,
                      env_vars=cfg.datasource.env_vars,
                      transform=transform
                      )
            for year in years]

    return data


def load_seasonal_dataset(cfg: DictConfig, output_dir: str, training: bool, transform=None):
    """
    Load seasonal data, initialize normalizer, setup and save configuration

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which config is written to
    :return: pytorch Dataset
    """
    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.datasource.n_dummy_radars}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_ndummy={cfg.datasource.n_dummy_radars}'
    data_dir = osp.join(cfg.device.root, 'data')

    if training:
        # initialize normalizer
        years = set(cfg.datasource.years) - set([cfg.datasource.test_year])

        with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(config=cfg, f=f)
    else:
        years = [cfg.datasource.test_year]

    # load training and validation data
    data = [SeasonalData(year, preprocessed_dirname, processed_dirname,
                                 **cfg, **cfg.model,
                                 data_root=data_dir,
                                 data_source=cfg.datasource.name,
                                 transform=transform
                                 )
            for year in years]

    return data




def rescale(features, min=None, max=None):
    """Rescaling of static features"""

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

def reshape(data, nights, mask, timesteps, use_nights=True, t0=0, step=1):
    """Reshape data to have dimensions [nodes, features, timesteps, sequences]"""

    if use_nights:
        reshaped = reshape_nights(data, nights, mask, timesteps)
    else:
        reshaped = reshape_t(data, timesteps, t0, step)
    return reshaped

def reshape_nights(data, nights, mask, timesteps):
    """Reshape into sequences that all start at the first hour of the night."""

    reshaped = [timeslice(data, night[0], mask, timesteps) for night in nights]
    reshaped = [d for d in reshaped if d.size > 0] # only use sequences that are fully available
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped


def reshape_t(data, timesteps, t0=0, step=1):
    """
    Reshape into sequences of length 'timesteps', starting at all possible time points in the data (step=1),
    or at regular intervals of size `step`
    """

    index = np.arange(t0, data.shape[-1] - timesteps, step)
    reshaped = [data[..., t:t + timesteps] for t in index]
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped

def timeslice(data, start_night, mask, timesteps):
    """Extract a sequence of length 'timesteps', that starts at time 'start_night'."""
    data_night = data[..., start_night:]

    data_night = data_night[..., mask[start_night:]]
    if data_night.shape[-1] >= timesteps:
        data_night = data_night[..., :timesteps]
    else:
        data_night = np.empty(0)
    return data_night


def compute_flux(dens, ff, dd, alpha, l=1):
    """Compute number of birds crossing transect of length 'l' [km] and angle 'alpha' per hour"""
    mtr = dens * ff * np.cos(np.deg2rad(dd - alpha))
    flux = mtr * l * 3.6
    return flux


def get_training_data(model_name, dataset, timesteps, mask_daytime=False, use_acc_vars=False):
    """Prepare training data for baseline model"""

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps):
            if model_name == 'GBT':
                features = [seq.coords.detach().numpy(),
                            seq.areas.view(-1,1).detach().numpy(),
                            seq.env[..., t].detach().numpy()]
                if use_acc_vars:
                    features.append(seq.acc[..., t].detach().numpy())
                features = np.concatenate(features, axis=1) # shape (nodes, features)
            else:
                # extract dayofyear, solarpos and solarpos_dt from env features
                features = seq.env[:, -3:, t].detach().numpy()  # shape (nodes, features)

            X.append(features)
            y.append(seq.y[:, t])

            if mask_daytime:
                mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask.append(~seq.missing[:, t])

    if model_name == 'GBT':
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        mask = np.concatenate(mask, axis=0)
    else:
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        mask = np.stack(mask, axis=0)

    return X, y, mask



def get_test_data(model_name, dataset, context, horizon, mask_daytime=False, use_acc_vars=False):
    """Prepare test data for baseline model."""

    X = []
    y = []
    mask = []
    for seq in dataset:
        X_night = []
        y_night = []
        mask_night = []
        for t in range(context, context+horizon):
            if model_name == 'GBT':
                features = [seq.coords.detach().numpy(),
                     seq.areas.view(-1, 1).detach().numpy(),
                     seq.env[..., t].detach().numpy()]
                if use_acc_vars:
                    features.append(seq.acc[..., t].detach().numpy())
                features = np.concatenate(features, axis=1)
            else:
                features = seq.env[:, -3:, t].detach().numpy()

            X_night.append(features)
            y_night.append(seq.y[:, t])
            if mask_daytime:
                mask_night.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask_night.append(~seq.missing[:, t])
        X.append(np.stack(X_night, axis=0))
        y.append(np.stack(y_night, axis=0))
        mask.append(np.stack(mask_night, axis=0))

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    mask = np.stack(mask, axis=0)  # shape (nights, timesteps, nodes)

    return X, y, mask

