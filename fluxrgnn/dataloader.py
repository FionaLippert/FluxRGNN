import torch
from torch_geometric.data import Data, HeteroData, DataLoader, Dataset, InMemoryDataset
import torch_geometric as ptg
from torch_geometric.nn import knn
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
from typing import Any, List, Optional, Sequence, Union
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

        for year in years:
            dir = self.preprocessed_dir(year)
            print(dir)
            if not osp.isdir(dir):
                print(f'Preprocessed data for year {year} not available. Please run preprocessing script first.')

            # load features
            dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'dynamic_cell_features.csv'))
            all_features.append(dynamic_feature_df)

            measurement_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'dynamic_radar_features.csv'))
            all_measurements.append(measurement_df)

        feature_df = pd.concat(all_features)
        measurement_df = pd.concat(all_measurements)

        self.feature_stats = feature_df.describe()
        self.feature_stats.loc['absmax'] = feature_df[self.feature_stats.columns].apply(lambda x: np.max(np.abs(x)))

        self.measurement_stats = measurement_df.describe()
        self.measurement_stats.loc['absmax'] = measurement_df[self.measurement_stats.columns].apply(lambda x: np.max(np.abs(x)))


    def normalize(self, data, key):

        key_min = self.min(key)
        key_max = self.max(key)

        data = (data - key_min) / (key_max - key_min)
        return data

    def denormalize(self, data, key):
        key_min = self.min(key)
        key_max = self.max(key)
        data = data * (key_max - key_min) + key_min
        return data

    def min(self, key):
        if key in self.measurement_stats:
            return self.measurement_stats[key]['min']
        else:
            return self.feature_stats[key]['min']

    def max(self, key):
        if key in self.measurement_stats:
            return self.measurement_stats[key]['max']
        else:
            return self.feature_stats[key]['max']
    
    def mean(self, key):
        if key in self.measurement_stats:
            return self.measurement_stats[key]['mean']
        else:
            return self.feature_stats[key]['mean']

    def std(self, key):
        if key in self.measurement_stats:
            return self.measurement_stats[key]['std']
        else:
            return self.feature_stats[key]['std']
    
    def absmax(self, key):
        if key in self.measurement_stats:
            return self.measurement_stats[key]['absmax']
        else:
            return self.feature_stats[key]['absmax']

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

        # generate dataset if it does not exist yet
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
        measurement_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_radar_features.csv'))

        print('loaded measurements')

        if not self.birds_per_km2:
            target_col = 'birds'
        else:
            target_col = 'birds_km2'

        time = measurement_df.datetime.sort_values().unique()
        tidx = np.arange(len(time))
        time_scale = pd.Timedelta(self.t_unit).total_seconds()

        data = dict(targets_x=[], targets_uv=[], missing_x=[], missing_uv=[])

        groups = measurement_df.groupby('ID')
        for name in measurement_df.ID:
            df = groups.get_group(name).sort_values(by='datetime').reset_index(drop=True)
            data['targets_x'].append(df[target_col].to_numpy())
            data['missing_x'].append(df['missing_birds_km2'].to_numpy())

            bird_uv = df[['bird_u', 'bird_v']].to_numpy().T  # in m/s
            bird_uv = bird_uv * time_scale / 1e3  # in km/[t_unit]
            data['targets_uv'].append(bird_uv)
            data['missing_uv'].append(df['missing_birds_uv'].to_numpy())

        for k, v in data.items():
            data[k] = np.stack(v, axis=0).astype(float)

        print('created data dict')

        # create graph data objects per sequence
        data = [SensorData(
            # graph structure and edge features
            edge_index=torch.zeros(0, dtype=torch.long),

            # animal densities
            x=torch.tensor(data['targets_x'], dtype=torch.float),
            missing_x=torch.tensor(data['missing_x'], dtype=torch.bool),

            bird_uv=torch.tensor(data['targets_uv'], dtype=torch.float),
            missing_bird_uv=torch.tensor(data['missing_uv'], dtype=torch.bool),

            # time index of sequence
            tidx=torch.tensor(tidx, dtype=torch.long))]

        print('created SensorData')

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)

        info = {'timepoints': time,
                'tidx': tidx}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])





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
        self.permute_env_vars = set(kwargs.get('permute_env_vars', [])).intersection(set(self.env_vars))

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
        self.tidx_max = kwargs.get('tidx_max', -1)

        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.default_rng(self.seed)
        self.data_perc = kwargs.get('data_perc', 1.0)

        # self.test_radars = kwargs.get('test_radars', [])
        self.n_cv_folds = kwargs.get('n_cv_folds', 0)
        self.cv_fold = kwargs.get('cv_fold', 0)

        super(RadarHeteroData, self).__init__(self.root, transform, pre_transform)

        # run self.process() to generate dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

        # save additional info
        with open(osp.join(self.processed_dir, self.info_file_name), 'rb') as f:
            self.info = pickle.load(f)

        # TODO: add test_radar info only when loading already processed data
        #  (to prevent storing different data for all CV folds)

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
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_cell_features.csv'))
        measurement_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_radar_features.csv'))
        cells = pd.read_csv(osp.join(self.preprocessed_dir, 'static_cell_features.csv'))
        radars = pd.read_csv(osp.join(self.preprocessed_dir, 'static_radar_features.csv'))

        # excluded_radars = np.concatenate([test_radars, radars[radars.radar.isin(self.exclude)].ID.values], axis=0)
        excluded_radars = radars[radars.radar.isin(self.exclude)].ID.values
        excluded_radars = torch.tensor(excluded_radars, dtype=torch.long)

        # relationship between cells and radars
        if self.edge_type in ['voronoi', 'none']:
            cell_to_radar_edge_index = torch.stack([torch.arange(len(cells)), torch.arange(len(cells))], dim=0).contiguous()
            cell_to_radar_weights = torch.ones(len(cells))
            
        else:
            cell_to_radar_edges = pd.read_csv(osp.join(self.preprocessed_dir, 'cell_to_radar_edges.csv'))
            # radar_to_cell_edges = pd.read_csv(osp.join(self.preprocessed_dir, 'radar_to_cell_edges.csv'))
            
            cell_to_radar_edge_index = torch.tensor(cell_to_radar_edges[['cidx', 'ridx']].values, dtype=torch.long)
            cell_to_radar_edge_index = cell_to_radar_edge_index.t().contiguous()
            cell_to_radar_dist = torch.tensor(cell_to_radar_edges['distance'].values, dtype=torch.float)

            if 'intersection' in cell_to_radar_edges.columns:
                cell_to_radar_weights = torch.tensor(cell_to_radar_edges['intersection'].values, dtype=torch.float)
            else:
                print('data on radar-cell intersections not available')
                cell_to_radar_weights = 1 / cell_to_radar_dist


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

        if not 'dayofyear' in dynamic_feature_df:
            dynamic_feature_df['dayofyear'] = pd.DatetimeIndex(dynamic_feature_df.datetime).dayofyear

        if self.normalization is not None:
            dynamic_feature_df = self.normalize_dynamic(dynamic_feature_df)
            measurement_df = self.normalize_dynamic(measurement_df)

        print(dynamic_feature_df.columns)

        # normalize static features
        coord_cols = ['x', 'y']
        xy_scale = cells[coord_cols].abs().max().max()
        cells[coord_cols] = cells[coord_cols] / xy_scale
        local_pos = cells[coord_cols].to_numpy()

        local_radar_pos = (radars[coord_cols] / xy_scale).to_numpy()

        lonlat_encoding = np.stack([np.sin(cells['lon'].to_numpy() * np.pi / 180.),
                                    np.cos(cells['lon'].to_numpy() * np.pi / 180.),
                                    np.sin(cells['lat'].to_numpy() * np.pi / 180.),
                                    np.cos(cells['lat'].to_numpy() * np.pi / 180.)], axis=1)

        lonlat_radar_encoding = np.stack([np.sin(radars['lon'].to_numpy() * np.pi / 180.),
                                    np.cos(radars['lon'].to_numpy() * np.pi / 180.),
                                    np.sin(radars['lat'].to_numpy() * np.pi / 180.),
                                    np.cos(radars['lat'].to_numpy() * np.pi / 180.)], axis=1)

        
        lonlat_encoding_old = (lonlat_encoding - lonlat_encoding.mean(0)) / lonlat_encoding.std(0)
        print(f'lonlat min = {lonlat_encoding_old.min(0)}, lonlat max = {lonlat_encoding_old.max(0)}')
        lonlat_encoding = lonlat_encoding - lonlat_encoding.mean(0)
        #lonlat_radar_encoding = (lonlat_radar_encoding - lonlat_encoding.mean(0)) / lonlat_encoding.std(0)    
        lonlat_radar_encoding = lonlat_radar_encoding - lonlat_encoding.mean(0)

        print(f'lonlat min = {lonlat_encoding.min(0)}, lonlat max = {lonlat_encoding.max(0)}')
        # land cover
        landcover_cols = [col for col in cells.columns if col.startswith('nlcd_c')]
        if len(landcover_cols) > 0:
            land_cover = torch.tensor(cells[landcover_cols].values, dtype=torch.float) # shape [cells, classes]
        else:
            land_cover = torch.zeros(0)
        print(f'landcover size: {land_cover.size()}')

        if 'nlcd_water' in cells.columns:
            water = torch.tensor(cells['nlcd_water'].values, dtype=torch.bool)
        else:
            water = torch.zeros(0)
        
        areas = cells[['area_km2']].to_numpy()
        length_scale = 1.0

        time_scale = pd.Timedelta(self.t_unit).total_seconds() # number of seconds per time step

        if self.edge_type == 'none':
            print('No graph structure used')
            areas = np.ones(areas.shape)
            edge_attr = torch.zeros(0)
            n_ij = torch.zeros(0)
            face_lengths = torch.zeros(0)
            radar_to_cell_edge_attr = torch.zeros(0)
            edge_weights = torch.zeros(0)
        else:
            print('Use tessellation')
            # get distances, angles and face lengths between radars
            distances = np.array([data['distance'] for i, j, data in G.edges(data=True)]) / length_scale

            edge_weights = rescale(1. / distances**2, min=0)

            angles = np.array([data['angle'] for i, j, data in G.edges(data=True)])
            angles_sin = np.sin(angles * np.pi / 180.)
            angles_cos = np.cos(angles * np.pi / 180.)

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
                torch.tensor(angles_sin, dtype=torch.float),
                torch.tensor(angles_cos, dtype=torch.float),
                torch.tensor(delta_x, dtype=torch.float),
                torch.tensor(delta_y, dtype=torch.float),
                torch.tensor(rescale(face_lengths, min=0), dtype=torch.float)
            ], dim=1)
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)


        time = dynamic_feature_df.datetime.sort_values().unique()
        tidx = np.arange(len(time))

        data = {'cell_nighttime': [], 'radar_nighttime': [], target_col: [], 'bird_uv': [],
                'missing_x': [], 'missing_uv': []}

        for var in self.env_vars:
            data[var] = []

        # process dynamic cell features
        for cid, group_df in dynamic_feature_df.groupby('ID'):
            df = group_df.sort_values(by='datetime').reset_index(drop=True)
            data['cell_nighttime'].append(df.night.to_numpy())

            for var in self.env_vars:
                data[var].append(df[var].to_numpy())


        # process radar measurements
        radar_ids = measurement_df.ID.unique()
        for rid, group_df in measurement_df.groupby('ID'):
            group_df = group_df.sort_values(by='datetime').reset_index(drop=True)
            data[target_col].append(group_df[target_col].to_numpy())
            data['missing_x'].append(group_df['missing_birds_km2'].to_numpy())
            data['missing_uv'].append(group_df['missing_birds_uv'].to_numpy())
            data['radar_nighttime'].append(group_df.night.to_numpy())

            bird_uv = group_df[['bird_u', 'bird_v']].to_numpy().T  # in m/s
            bird_uv = bird_uv * time_scale / 1e3  # in km/[t_unit]
            data['bird_uv'].append(bird_uv)


        for k, v in data.items():
            data[k] = np.stack(v, axis=0).astype(float)

        print(f'bird_uv min = {data["bird_uv"].min()}, max = {data["bird_uv"].max()}')


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
                                  self.tidx_start, self.tidx_step, self.tidx_max)

            tidx = reshape(tidx, nights, np.ones(check_all.shape, dtype=bool), self.timesteps, self.use_nights,
                           self.tidx_start, self.tidx_step, self.tidx_max)

        # remove sequences with too much missing data
        data['missing'] = np.logical_and(data['missing_x'], data['missing_uv'])
        print(f'missing shape = {data["missing"].shape}')
        perc_missing = data['missing']
        if self.edge_type == 'voronoi':
            # ignore dummy radars for computation of missing data
            observed = cells['observed'].to_numpy()
            perc_missing = perc_missing[observed]
        perc_missing = perc_missing.reshape(-1, perc_missing.shape[-1]).mean(0)
        print(perc_missing)
        valid_idx = perc_missing <= self.missing_data_threshold

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

        # tessellation structure
        cell2cell_edges = {
            'edge_index': edge_index,
            'reverse_edges': reverse_edges,
            'boundary2inner_edges': boundary2inner_edges.bool(),
            'inner2boundary_edges': inner2boundary_edges.bool(),
            'boundary2boundary_edges': boundary2boundary_edges.bool(),
            'inner_edges': inner_edges.bool(),
            'edge_attr': edge_attr,
            'edge_weights': edge_weights,
            'edge_normals': torch.tensor(n_ij, dtype=torch.float),
            'edge_face_lengths': torch.tensor(face_lengths, dtype=torch.float)
        }

        # observation model structure
        cell2radar_edges = {
            'edge_index': cell_to_radar_edge_index,
            'edge_weight': cell_to_radar_weights
        }

        # ignore excluded radars during both training and testing
        radar_mask = torch.ones(len(radar_ids), dtype=torch.bool)
        radar_mask[excluded_radars] = False

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
                'land_cover': land_cover,
                'water': water,

                # dynamic cell features
                'local_night': torch.tensor(data['cell_nighttime'][..., idx], dtype=torch.bool),
                'tidx': torch.tensor(tidx[:, idx], dtype=torch.long),
                'length_scale': torch.tensor(length_scale)
            }

            for var in self.env_vars:
                cell_data[var] = torch.tensor(data[var][..., idx], dtype=torch.float)

            if self.edge_type in ['voronoi', 'none']:
                cell_data['x'] = torch.tensor(data[target_col][..., idx], dtype=torch.float)
                cell_data['bird_uv'] = torch.tensor(data['bird_uv'][..., idx], dtype=torch.float)

            
            x = torch.tensor(data[target_col][..., idx], dtype=torch.float)
            bird_uv = torch.tensor(data['bird_uv'][..., idx], dtype=torch.float)

            radar_data = {
                # static radar features
                'ridx': torch.arange(len(radar_ids), dtype=torch.long),
                'pos': torch.tensor(local_radar_pos, dtype=torch.float),
                'coords': torch.tensor(lonlat_radar_encoding, dtype=torch.float),
                'train_mask': radar_mask,
                'test_mask': radar_mask,

                # dynamic radar features
                'x': x,
                'missing_x': torch.tensor(data['missing_x'][..., idx], dtype=torch.bool),
                'missing_bird_uv': torch.tensor(data['missing_uv'][..., idx], dtype=torch.bool),
                'missing': torch.tensor(data['missing'][..., idx], dtype=torch.bool),
                'local_night': torch.tensor(data['radar_nighttime'][..., idx], dtype=torch.bool),
                'bird_uv': bird_uv,
                'tidx': torch.tensor(tidx[:, idx], dtype=torch.long),
                'year': torch.tensor(int(self.year), dtype=torch.long)
            }

            # heterogeneous graph with two types of nodes: cells and radars
            data_list.append(SensorHeteroData(
                cell = cell_data,
                radar = radar_data,
                cell__to__cell = cell2cell_edges,
                cell__to__radar = cell2radar_edges,
            ))


        print(f'number of sequences = {len(data_list)}')

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)
        n_seq_discarded = valid_idx.size - valid_idx.sum()
        print(f'discarded {n_seq_discarded} sequences due to missing data')

        info = {
                'env_vars': self.env_vars,
                'n_seq_discarded': n_seq_discarded,
                'length_scale': length_scale,
                'time_scale': time_scale
        }

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def normalize_dynamic(self, dynamic_feature_df):
        """Normalize dynamic features to range between 0 and 1."""

        cidx = ~dynamic_feature_df.columns.isin(['birds', 'birds_km2', 'birds_km2_from_buffer',
                                                 'bird_speed', 'bird_direction', 'wind_u', 'wind_v',
                                                 'bird_u', 'bird_v', 
                                                 #'u', 'v', 'u10', 'v10',
                                                 #'cc', 'sshf', 
                                                 'dayofyear',
                                                 'radar', 'ID', 'night', 'boundary',
                                                 'dusk', 'dawn', 'datetime', 'missing',
                                                 'missing_birds_km2', 'missing_birds_uv'])


        # rescale q, t2m, t, sp to [-1, 1] 
        dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
             lambda col: 2 * ((col - self.normalization.min(col.name)) /
                         (self.normalization.max(col.name) - self.normalization.min(col.name))) - 1, axis=0)

        if 'dayofyear' in dynamic_feature_df:
            dynamic_feature_df['dayofyear'] /= 365.0

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




def load_dataset(cfg: DictConfig, output_dir: str, split: str, transform=None, seqID_min=0, seqID_max=-1, normalization=None):
    """
    Load training or testing data, initialize normalizer, setup and save configuration

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which config is written to
    :return: pytorch Dataset
    """
    context = cfg.model.get('context', 0)
    if (context == 0) and (split == 'test'):
        context = cfg.model.get('test_context', 0)

    seq_len = context + max(cfg.model.get('horizon', 1), cfg.model.get('test_horizon')) \
              + cfg.datasource.get('tidx_step', 1) #- 1

    if cfg.model.edge_type == 'hexagons':
        res_info = f'_{cfg.datasource.buffer}_res={cfg.datasource.h3_resolution}'
    elif cfg.model.edge_type == 'voronoi':
        res_info = f'_ndummy={cfg.datasource.n_dummy_radars}'
    else:
        res_info = ''

    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}' + res_info

    processed_dirname = f'maxT0={cfg.model.max_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_dataperc={cfg.data_perc}_' \
                        f'missing_thr={cfg.missing_data_threshold}_' \
                        f'seed={cfg.seed}' + res_info
                        #f'fold={n_cv_folds}-{cv_fold}_seed={cfg.seed}'

    n_excl = len(cfg.datasource.get('excluded_radars', []))
    if n_excl > 0:
        processed_dirname += f'_excluded={n_excl}'
    
    data_dir = osp.join(cfg.device.root, 'data')

    if normalization is None:
        print('setup feature normalization')
        norm_years = cfg.datasource['train_years']
        normalization = Normalization(norm_years, cfg.datasource.name, data_dir, preprocessed_dirname, **cfg)
    
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    
    years = cfg.datasource[f'{split}_years']

    data = []
    for year in years:
        data_y = RadarHeteroData(year, seq_len, preprocessed_dirname, processed_dirname,
                      **cfg, **cfg.model, **cfg.task,
                      data_root=data_dir,
                      data_source=cfg.datasource.name,
                      normalization=normalization,
                      exclude=cfg.datasource.get('excluded_radars', []),
                      tidx_start=cfg.datasource.get('tidx_start', 0),
                      tidx_step=cfg.datasource.get('tidx_step', 1),
                      transform=transform
                      )
        print(f'len data = {len(data_y)}')
        indices = range(0, len(data_y))
        print(indices)
        print(seqID_min, seqID_max)
        if seqID_max < 0:
            seqID_max = len(data_y) - 1
        data.append(torch.utils.data.Subset(data_y, indices[seqID_min: seqID_max + 1]))
    
    return data, normalization #context, seq_len


def load_xgboost_dataset(cfg: DictConfig, output_dir: str, split: str = 'train', transform=None):
    """
    Load training data for XGBoost model, for which no time sequences are needed.

    :param cfg: DictConfig specifying model, data and training details
    :param output_dir: directory to which config is written to
    :return: pytorch Dataset
    """

    seq_len = 'all'

    preprocessed_dirname = f'{cfg.t_unit}_none'

    model_cfg = dict(cfg.model)
    model_cfg['edge_type'] = 'none'

    processed_dirname = f'buffers={cfg.datasource.use_buffers}_log={cfg.model.use_log_transform}_' \
                        f'pow={cfg.model.get("pow_exponent", 1.0)}_maxT0={cfg.model.max_t0}_timepoints={seq_len}_' \
                        f'edges={cfg.model.edge_type}_dataperc={cfg.data_perc}' \
                        f'_seed={cfg.seed}'
    
    data_dir = osp.join(cfg.device.root, 'data')

    # initialize normalizer
    normalization = Normalization(cfg.datasource.train_years, cfg.datasource.name,
                                  data_dir, preprocessed_dirname, **cfg)

    # complete config and write it together with normalizer to disk
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)


    data = [RadarHeteroData(year, seq_len, preprocessed_dirname, processed_dirname,
                      **cfg, **model_cfg, **cfg.task,
                      data_root=data_dir,
                      data_source=cfg.datasource.name,
                      normalization=normalization,
                      transform=transform
                      )
            for year in cfg.datasource[f'{split}_years']]

    return data


def load_seasonal_dataset(cfg: DictConfig, output_dir: str, split: str, transform=None):
    """
    Load seasonal data, initialize normalizer, setup and save configuration

    :param cfg: DictConfig specifying model, data and training/testing details
    :param output_dir: directory to which config is written to
    :return: pytorch Dataset
    """
    preprocessed_dirname = f'{cfg.t_unit}_{cfg.model.edge_type}'

    if cfg.model.edge_type == 'hexagons':
        res_info = f'_{cfg.datasource.buffer}_res={cfg.datasource.h3_resolution}'
    elif cfg.model.edge_type == 'voronoi':
        res_info = f'_ndummy={cfg.datasource.n_dummy_radars}'
    else:
        res_info = ''

    processed_dirname = f'seasonal_buffers={cfg.datasource.use_buffers}_log={cfg.model.use_log_transform}_' \
                        f'pow={cfg.model.get("pow_exponent", 1.0)}_maxT0={cfg.model.max_t0}_dataperc={cfg.data_perc}'
    
    preprocessed_dirname += res_info
    processed_dirname += res_info
    
    data_dir = osp.join(cfg.device.root, 'data')

    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    years = cfg.datasource[f'{split}_years']
    print(f'load data for {years}')

    # load training and validation data
    data = [SeasonalData(year, preprocessed_dirname, processed_dirname,
                                 **cfg, **cfg.model, **cfg.task,
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

def reshape(data, nights, mask, timesteps, use_nights=True, t0=0, step=1, t_max=-1):
    """Reshape data to have dimensions [nodes, features, timesteps, sequences]"""

    if use_nights:
        reshaped = reshape_nights(data, nights, mask, timesteps)
    else:
        reshaped = reshape_t(data, timesteps, t0, step, t_max)
    return reshaped

def reshape_nights(data, nights, mask, timesteps):
    """Reshape into sequences that all start at the first hour of the night."""

    reshaped = [timeslice(data, night[0], mask, timesteps) for night in nights]
    reshaped = [d for d in reshaped if d.size > 0] # only use sequences that are fully available
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped


def reshape_t(data, timesteps, t0=0, step=1, t_max=-1):
    """
    Reshape into sequences of length 'timesteps', starting at all possible time points in the data (step=1),
    or at regular intervals of size `step`
    """
    if t_max < 0:
        t_max = data.shape[-1] - 1

    index = np.arange(t0, t_max + 1 - timesteps, step)
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

