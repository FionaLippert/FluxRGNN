import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch_geometric.nn import MessagePassing, inits
from torch_geometric.utils import to_dense_adj
import pytorch_lightning as pl
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from fluxrgnn import utils


class ForecastModel(pl.LightningModule):
    """
    Base model class for forecasting bird migration
    """

    def __init__(self, observation_model=None, **kwargs):
        super(ForecastModel, self).__init__()

        # save config
        self.save_hyperparameters()
        self.config = kwargs

        # forecasting settings
        self.horizon = max(1, kwargs.get('horizon', 1))
        self.t_context = max(1, kwargs.get('context', kwargs.get('test_context', 1)))
        self.tf_start = min(1.0, kwargs.get('teacher_forcing', 0.0))

        # observation model mapping cell densities to radar observations
        self.observation_model = observation_model

        self.use_log_transform = kwargs.get('use_log_transform', False)
        self.scale = kwargs.get('scale', 1.0)
        self.log_offset = kwargs.get('log_offset', 1e-8)
        self.pow_exponent = kwargs.get('pow_exponent', 0.3333)
        
        # if self.use_log_transform:
        #     self.zero_value = np.log(self.log_offset) * self.scale
        # else:
        #     self.zero_value = 0

        print(f'during initialization, the model is on {self.device}')

        self.transforms = kwargs.get('transforms', [])
        self.zero_value = self.apply_forward_transforms(torch.tensor(0, device=self.device))
        print(f'zero value = {self.zero_value}')


    def forecast(self, data, horizon, teacher_forcing=0, t0=0):
        """
        Setup prediction for given data and run model until the max forecasting horizon is reached.

        :param data: SensorData instance containing information on static and dynamic features for one time sequence
        :param teacher_forcing: teacher forcing probability
        :return: predicted migration intensities for all cells and time points
        """

        # make sure t0 is tensor of dimension 2
        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0, device=self.device)
        t0 = t0.view(-1, 1)

        # initialize forecast
        model_states = self.initialize(data, t0)
        y_hat = []
        
        # TODO: include initial model state in forecast and training as well (but how to do this for MLP?)

        # only cell data is needed for forecasting
        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()
        

        # predict until the max forecasting horizon is reached
        forecast_horizon = range(self.t_context, self.t_context + horizon)
        for tidx in forecast_horizon:

            t = t0 + tidx

            # use gt data instead of model output with probability 'teacher_forcing'
            r = torch.rand(1)
            #if hasattr(data, 'x') and (r < teacher_forcing):
            if r < teacher_forcing:
                #print('use teacher forcing')
                # TODO: map measurements in node_storage 'radar' to cells in node_storage 'cell',
                #  or use smaller horizon instead of teacher forcing?
                #model_states['x'] = data.x[..., t - 1].view(-1, 1)
                radars_to_cells = data['radar', 'cell']
                interpolated = self.observation_model(data['radar'].x, radars_to_cells)
                model_states['x'] = tidx_select(interpolated, t - 1).view(-1, 1)
            #else:
            #    print('no ground truth data available to use for teacher forcing')
            
            
            # make prediction for next time step
            model_states = self.forecast_step(model_states, cell_data, t, teacher_forcing)
            y_hat.append(model_states['x'])

        prediction = torch.cat(y_hat, dim=-1)

        return prediction


    def initialize(self, data, t0=0):

        model_states = {}

        return model_states


    def forecast_step(self, model_states, data, t, teacher_forcing):

        raise NotImplementedError


    def training_step(self, batch, batch_idx):


        # get teacher forcing probability for current epoch
        tf = self.tf_start * pow(self.config.get('teacher_forcing_gamma', 1), 
                                 self.current_epoch)
        self.log('teacher_forcing', tf)
        
        h_rate = self.config.get('increase_horizon_rate', 0)
        if h_rate > 0:
            epoch = max(0, self.current_epoch - self.config.get('increase_horizon_start', 0))
            horizon = min(self.horizon, int(epoch * h_rate) + 1)
        else:
            horizon = self.horizon

        self.log('horizon', horizon)
        
        t0 = torch.randint(0, self.config.get('max_t0', 1), (batch.num_graphs,), device=self.device)
        
        # extract relevant data from batch
        #cell_data = batch.node_type_subgraph(['cell']).to_homogeneous()
        radar_data = batch['radar']
        
        # make predictions for all cells
        prediction = self.forecast(batch, horizon, teacher_forcing=tf, t0=t0)

        # map cell predictions to radar observations
        if self.observation_model is not None:
            cells_to_radars = batch['cell', 'radar']
            prediction = self.observation_model(prediction, cells_to_radars)
            prediction = prediction[:radar_data.num_nodes]

        #compute loss
        eval_dict = self._eval_step(radar_data, prediction, horizon, prefix='train', t0=t0)
        self.log_dict(eval_dict, batch_size=batch.num_graphs)

        return eval_dict['train/loss']


    def validation_step(self, batch, batch_idx):

        t0 = torch.randint(0, self.config.get('max_t0', 1), (batch.num_graphs,), device=self.device)

        # extract relevant data from batch
        radar_data = batch['radar']
        
        # make predictions for all cells
        prediction = self.forecast(batch, self.horizon, t0=t0)

        # apply observation model to forecast
        if self.observation_model is not None:
            cells_to_radars = batch['cell', 'radar']
            prediction = self.observation_model(prediction, cells_to_radars)
            prediction = prediction[:radar_data.num_nodes]

        # evaluate forecast
        eval_dict = self._eval_step(radar_data, prediction, self.horizon, prefix='val', t0=t0)
        self.log_dict(eval_dict, batch_size=batch.num_graphs)


    def on_test_epoch_start(self):

        self.test_results = {
                'test/mask': [], 
                'test/measurements': [], 
                'test/predictions': []
                }
        self.test_metrics = {}


    def test_step(self, batch, batch_idx):

        # extract relevant data from batch
        #cell_data = batch.node_type_subgraph(['cell']).to_homogeneous()
        radar_data = batch['radar']

        for t0 in range(self.config.get('max_t0', 1)):
            #print(t0)
            # make predictions for all cells
            prediction = self.forecast(batch, self.horizon, t0=t0)

            # apply observation model to forecast
            if self.observation_model is not None:
                cells_to_radars = batch['cell', 'radar']
                prediction = self.observation_model(prediction, cells_to_radars)
                prediction = prediction[:radar_data.num_nodes]

            # compute evaluation metrics
            eval_dict = self._eval_step(radar_data, prediction, self.horizon, prefix='test', t0=t0)
            self.log_dict(eval_dict, batch_size=batch.num_graphs)

            # compute evaluation metrics as a function of the forecasting horizon
            eval_dict_per_t = self._eval_step(radar_data, prediction, self.horizon, prefix='test', aggregate_time=False, t0=t0)

            for m, values in eval_dict_per_t.items():
                if m in self.test_metrics:
                    self.test_metrics[m].append(values)
                else:
                    self.test_metrics[m] = [values]

            self.test_results['test/mask'].append(
                torch.logical_not(radar_data.missing)[:, t0:t0 +self.t_context + self.horizon]
            )
            self.test_results['test/measurements'].append(
                self.transformed2raw(radar_data.x)[:, t0:t0 + self.t_context + self.horizon]
            )
            self.test_results['test/predictions'].append(
                self.transformed2raw(prediction)
            )


    def on_test_epoch_end(self):

        for m, value_list in self.test_metrics.items():
            self.test_metrics[m] = torch.concat(value_list, dim=0).reshape(-1, self.horizon)

        for m, value_list in self.test_results.items():
            self.test_results[m] = torch.stack(value_list)


    def predict_step(self, batch, batch_idx):

        # extract relevant data from batch
        cell_data = batch['cell']
        radar_data = batch['radar']


        #for t0 in range(self.config.get('max_t0', 1)):
        
        # make predictions for all cells
        prediction = self.forecast(batch, self.horizon, t0=0)

        # apply observation model to forecast
        if self.observation_model is not None:
            cells_to_radars = batch['cell', 'radar']
            prediction = self.observation_model(prediction, cells_to_radars)
            prediction = prediction[:radar_data.num_nodes]

        result = {
            'predictions': self.transformed2raw(prediction),
            'measurements': self.transformed2raw(radar_data.x),
            'local_night': cell_data.local_night,
            'missing': radar_data.missing,
            'tidx': cell_data.tidx
        }
        return result

    def apply_forward_transforms(self, values: torch.Tensor):

        out = values
        for t in self.transforms:
            out = t.tensor_forward(out)

        return out

    def apply_backward_transforms(self, values: torch.Tensor):

        out = values
        for t in reversed(self.transforms):
            out = t.tensor_backward(out)

        return out

    # def to_raw(self, values):
    def transformed2raw(self, values):

        values = torch.clamp(values, min=self.zero_value)
        raw = self.apply_backward_transforms(values)

        # assert torch.allclose(self.apply_forward_transforms(raw), values)

        return raw

    # def to_log(self, values):
    def raw2log(self, values):

        values = torch.clamp(values, min=0)
        log = torch.log(values + self.log_offset)

        # if self.use_log_transform:
        #     log = values / self.scale
        # else:
        #     raw = values / self.scale
        #     log = torch.log(raw + self.log_offset)

        return log

    def raw2pow(self, values):

        pow = torch.pow(values, 1/3)
        # pow = torch.pow(values, self.pow_exponent)

        return pow

    def _regularizer(self):
        return 0

    def _eval_step(self, radar_data, output, horizon, prefix='', aggregate_time=True, t0=0):

        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0, device=self.device)
        t0 = t0.view(-1, 1)
        

        if self.config.get('force_zeros', False):
            mask = torch.logical_and(radar_data.local_night, torch.logical_not(radar_data.missing))
        else:
            mask = torch.logical_not(radar_data.missing)

        gt = tidx_select(radar_data.x, t0, steps=(self.t_context + horizon))
        gt = gt.view(radar_data.x.size(0), -1)[:, self.t_context: self.t_context + horizon]
        mask = tidx_select(mask, t0, steps=(self.t_context + horizon)).view(mask.size(0), -1)[:, self.t_context: self.t_context + horizon]
        output = output[:, :horizon]
        
        if aggregate_time:
            gt = gt.reshape(-1)
            mask = mask.reshape(-1)
            output = output.reshape(-1)

        loss = utils.MSE(output, gt, mask)
        
        if self.training:
            regularizer = self._regularizer()
            loss = loss + self.config.get('regularizer_weight', 1.0) * regularizer
            eval_dict = {f'{prefix}/loss': loss,
                         f'{prefix}/log-loss': torch.log(loss),
                         f'{prefix}/regularizer': regularizer}
        else:
            eval_dict = {f'{prefix}/loss': loss,
                         f'{prefix}/log-loss': torch.log(loss)}

        if not self.training:
            raw_gt = self.transformed2raw(gt)
            raw_output = self.transformed2raw(output)
            self._add_eval_metrics(eval_dict, raw_gt, raw_output, mask, prefix=f'{prefix}/raw')

            log_gt = self.raw2log(raw_gt)
            log_output = self.raw2log(raw_output)
            self._add_eval_metrics(eval_dict, log_gt, log_output, mask, prefix=f'{prefix}/log')

            pow_gt = self.raw2pow(raw_gt)
            pow_output = self.raw2pow(raw_output)
            self._add_eval_metrics(eval_dict, pow_gt, pow_output, mask, prefix=f'{prefix}/pow')

            # print(f'min output = {output.min()}')
            # print(f'min output (raw) = {raw_output.min()}')
            #
            # print(f'min gt = {gt.min()}')
            # print(f'min gt (raw) = {raw_gt.min()}')

        return eval_dict

    def _add_eval_metrics(self, eval_dict, gt, output, mask, prefix=''):

        # root mean squared error
        rmse = torch.sqrt(utils.MSE(output, gt, mask))
        eval_dict.update({f'{prefix}/RMSE': rmse})

        # mean absolute error
        mae = utils.MAE(output, gt, mask)
        eval_dict.update({f'{prefix}/MAE': mae})

        # symmetric mean absolute percentage error
        smape = utils.SMAPE(output, gt, mask)
        eval_dict.update({f'{prefix}/SMAPE': smape})

        # mean absolute percentage error
        mape = utils.MAPE(output, gt, mask)
        eval_dict.update({f'{prefix}/MAPE': mape})
        
        # avg residuals
        mean_res = ((output - gt) * mask).sum(0) / mask.sum(0)
        eval_dict.update({f'{prefix}/mean_residual': mean_res})

        # pseudo R squared (a.k.a "variance explained")
        r2 = utils.R2(output, gt, mask)
        eval_dict.update({f'{prefix}/R2': r2})




    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get('lr', 0.01))
        optimizer = self.config.get('optimizer')(self.parameters())
        #scheduler = StepLR(optimizer,
        #                   step_size=self.config.get('lr_decay', 1000),
        #                   gamma=self.config.get('lr_gamma', 1))

        scheduler_list = self.config.get('lr_schedulers', None)
        milestone_list = self.config.get('lr_milestones', None)
        if scheduler_list is not None and milestone_list is not None:
            scheduler = SequentialLR(optimizer, [s_partial(optimizer) for s_partial in scheduler_list], milestones=milestone_list)
        else:
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1e8)

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler}




class FluxRGNN(ForecastModel):
    """
    Recurrent graph neural network based on a mechanistic description of population-level movements
    on the Voronoi tesselation of sensor network sites.
    """

    def __init__(self, decoder, flux_model=None, source_sink_model=None,
                 encoder=None, boundary_model=None, initial_model=None, **kwargs):
        """
        Initialize FluxRGNN and all its components.

        :param dynamics: transition model (e.g. FluxRGNNTransition, LSTMTransition or Persistence)
        :param encoder: encoder model (e.g. RecurrentEncoder)
        :param boundary_model: model handling boundary cells (e.g. Extrapolation)
        :param initial_model: model predicting the initial state
        """

        super(FluxRGNN, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.flux_model = flux_model
        self.source_sink_model = source_sink_model
        self.boundary_model = boundary_model
        self.initial_model = initial_model


    def initialize(self, graph_data, t0=0):

        # TODO: use separate topographical / location embedding model whose output is fed to all other model components?

        cell_data = graph_data['cell']
        cell_edges = graph_data['cell', 'cell']

        if self.encoder is not None:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(cell_data, t0)
        else:
            h_t = [torch.zeros(cell_data.num_nodes, self.decoder.n_hidden, device=cell_data.coords.device) for
                   _ in range(self.decoder.n_lstm_layers)]
            c_t = [torch.zeros(cell_data.num_nodes, self.decoder.n_hidden, device=cell_data.coords.device) for
                   _ in range(self.decoder.n_lstm_layers)]

        self.decoder.initialize(h_t, c_t)

        self.regularizers = []

        # setup model components
        if self.boundary_model is not None:
            self.boundary_model.initialize(cell_edges)

        # relevant info for later
        if not self.training and self.config.get('store_fluxes', False):
            self.edge_fluxes = []#torch.zeros((cell_edges.edge_index.size(1), 1, self.horizon), device=cell_data.coords.device)
            self.node_flux = []#torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)
            self.node_sink = []#torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)
            self.node_source = []#torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)

        # initial system state
        if self.initial_model is not None:
            x = self.initial_model(graph_data, max(0, self.t_context - 1) + t0, h_t[-1])
        #elif hasattr(cell_data, 'x'):
        #    x = cell_data.x[..., self.t_context - 1].view(-1, 1)
        else:
            x = torch.zeros_like(cell_data.coords[:, 0]).unsqueeze(-1)
        #print(f'initial x size = {x.size()}')

        model_states = {'x': x,
                        'hidden': h_t[-1],
                        'boundary_nodes': cell_data.boundary.view(-1, 1),
                        'inner_nodes': torch.logical_not(cell_data.boundary).view(-1, 1)}

        # if self.ground_model is not None:
        #     model_states['ground_states'] = self.ground_model(h_t)

        return model_states

    def forecast_step(self, model_states, cell_data, t, teacher_forcing=0):

        # use gt data instead of model output with probability 'teacher_forcing'
        #r = torch.rand(1)
        #if cell_data.x is not None and (r < teacher_forcing):
            # TODO: map measurements in node_storage 'radar' to cells in node_storage 'cell',
            #  or use smaller horizon instead of teacher forcing?
            #x = cell_data.x[torch.arange(cell_data.num_nodes, device=cell_data.x.device), t - 1].view(-1, 1)
        #else:
        x = model_states['x']
        
        hidden = model_states['hidden']
        #tidx = t - self.t_context

        if self.boundary_model is not None:
            x_boundary = self.boundary_model(x)
            h_boundary = self.boundary_model(hidden)
            x = x * model_states['inner_nodes'] + x_boundary * model_states['boundary_nodes']
            hidden = hidden * model_states['inner_nodes'] + h_boundary * model_states['boundary_nodes']
        #print(f'x after boundary model: {x.size()}')

        # update hidden states
        hidden = self.decoder(x, hidden, cell_data, t)

        # predict movements
        if self.flux_model is not None:

            # predict fluxes between neighboring cells
            net_flux = self.flux_model(x, hidden, cell_data, t)
            x = x + net_flux

            if not self.training and self.config.get('store_fluxes', False):
                # save model component outputs
                self.edge_fluxes.append(self.flux_model.edge_fluxes)
                self.node_flux.append(self.flux_model.node_flux)

        if self.source_sink_model is not None:

            # predict local source/sink terms
            delta = self.source_sink_model(x, hidden, cell_data, t,
                                           ground_states=model_states.get('ground_states', None))
            x = x + delta

            if not self.training and self.config.get('store_fluxes', False):
                # save model component outputs
                self.node_source.append(self.source_sink_model.node_source)
                self.node_sink.append(self.source_sink_model.node_sink)
            else:
                #self.regularizers.append(delta)
                self.regularizers.append(self.source_sink_model.node_source + self.source_sink_model.node_sink)

        if self.config.get('force_zeros', False):
            x = x * cell_data.local_night[torch.arange(cell_data.num_nodes, device=cell_data.local_night.device), t]
            x = x + self.zero_value * torch.logical_not(cell_data.local_night[
                    torch.arange(cell_data.num_nodes, device=cell_data.local_night.device), t])
        
        model_states['x'] = x
        model_states['hidden'] = hidden

        return model_states

    def _regularizer(self):

        if len(self.regularizers) > 0:
            regularizers = torch.cat(self.regularizers, dim=0)
            penalty = regularizers.pow(2).mean()
        else:
            penalty = 0

        return penalty


    def on_predict_start(self):

        self.predict_results = {
                'prediction': [],
                'node_source': [],
                'node_sink': [],
                'node_flux': [],
                'edge_flux': []
                }

    def on_predict_end(self):

        for m, value_list in self.predict_results.items():
            self.predict_results[m] = torch.stack(value_list)


    def predict_step(self, batch, batch_idx):

        # extract relevant data from batch
        cell_data = batch['cell']
        radar_data = batch['radar']
        
        # make predictions for all cells
        prediction = self.forecast(batch, self.horizon)

        # apply observation model to forecast
        #if self.observation_model is not None:
        #    cells_to_radars = batch['cell', 'radar']
        #    prediction = self.observation_model(prediction, cells_to_radars)
        #    prediction = prediction[:radar_data.num_nodes]

        self.predict_results['prediction'].append(self.transformed2raw(prediction))
        self.predict_results['node_source'].append(self.node_source)
        self.predict_results['node_sink'].append(self.node_sink)
        self.predict_results['node_flux'].append(self.node_flux)
        self.predict_results['edge_flux'].append(self.edge_fluxes)

        #result = {
        #    'predictions': self.to_raw(prediction),
        #    'measurements': self.to_raw(radar_data.x),
        #    'local_night': cell_data.local_night,
        #    'missing': radar_data.missing,
        #    'tidx': cell_data.tidx
        #}
        
        return prediction

    #def predict_step(self, batch, batch_idx):

    #    # make predictions
    #    output = self.to_raw(self.forecast(batch))
    #    gt = self.to_raw(batch.y) if hasattr(batch, 'y') else None

    #    # get fluxes along edges
    #    #adj = to_dense_adj(batch.edge_index, edge_attr=self.edge_fluxes)
    #    #edge_fluxes = adj.view(batch.num_nodes, batch.num_nodes, -1)

    #    # get net fluxes per node
    #    #influxes = edge_fluxes.sum(1)
    #    #outfluxes = edge_fluxes.permute(1, 0, 2).sum(1)

    #    #if hasattr(batch, 'fluxes'):
    #    #    # compute approximate fluxes from radar data
    #    #    radar_fluxes = to_dense_adj(batch.edge_index, edge_attr=batch.fluxes).view(
    #    #        batch.num_nodes, batch.num_nodes, -1)
    #    #else:
    #    #    radar_fluxes = None

    #    result = {
    #        'y_hat': output,
    #        'y': gt,
    #        #'influx': influxes,
    #        #'outflux': outfluxes,
    #        #'source': self.node_source,
    #        #'sink': self.node_sink,
    #        #'edge_fluxes': edge_fluxes,
    #        #'radar_fluxes': radar_fluxes,
    #        'local_night': batch.local_night,
    #        'missing': batch.missing,
    #        'tidx': batch.tidx
    #    }
    #    return result


class LocalMLPForecast(ForecastModel):
    """
    Forecast model using a local MLP with parameters shared across time and space.
    """

    def __init__(self, node_features, dynamic_features, **kwargs):
        """
        Initialize LocalMLPForecast and all its components.
        """

        super(LocalMLPForecast, self).__init__(**kwargs)

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        # setup model
        n_in = sum(node_features.values()) + sum(dynamic_features.values())
        self.mlp = NodeMLP(n_in, **kwargs)


    def forecast_step(self, model_states, data, t, *args, **kwargs):

        # static features
        node_features = torch.cat([data.get(feature).reshape(data.num_nodes, -1) for
                                   feature in self.node_features], dim=1)

        # dynamic features for current time step t
        dynamic_features = torch.cat([tidx_select(data.get(feature), t).reshape(data.num_nodes, -1) for
                                      feature in self.dynamic_features], dim=1)

        # combined features
        inputs = torch.cat([node_features, dynamic_features], dim=1).detach().numpy()

        x = self.mlp(inputs)

        if self.config.get('square_output', False):
            x = torch.pow(x, 2)

        if self.config.get('force_zeros', False):
            x = x * tidx_select(data.local_night, t)
            x = x + self.zero_value * torch.logical_not(tidx_select(data.local_night, t))

        model_states['x'] = x

        return model_states



class NodeMLP(torch.nn.Module):
    """Standard MLP mapping concatenated features of a single nodes at time t to migration intensities at time t."""

    def __init__(self, n_in, **kwargs):
        super(NodeMLP, self).__init__()

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_layers = kwargs.get('n_fc_layers', 1)

        self.fc_in = torch.nn.Linear(n_in, self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)


    def forward(self, features):

        # use only location-specific features to predict migration intensities

        x = F.relu(self.fc_in(features))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = F.relu(l(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)

        return x


# class Persistence(torch.nn.Module):
#
#     def __init__(self):
#
#         super(Persistence, self).__init__()
#
#     def initialize(self, *args):
#
#         return None
#
#     def forward(self, data, x, hidden, *args, **kwargs):
#
#         return x, hidden


class SeasonalityForecast(ForecastModel):
    """
    Forecast model using the seasonal patterns from the training data to predict animal densities.
    """

    def __init__(self, **kwargs):
        """
        Initialize SeasonalityForecast model.
        """

        super(SeasonalityForecast, self).__init__(**kwargs)

        self.automatic_optimization = False

    def forecast_step(self, model_states, data, t, *args, **kwargs):
        #print(data.ridx.device, self.seasonal_patterns.device)
        # get typical density for each radars at the given time point
        model_states['x'] = self.seasonal_patterns[data.ridx, data.tidx[t]].view(-1, 1)

        return model_states

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return None


class XGBoostForecast(ForecastModel):
    """
    Forecast model using XGBoost to predict local animal densities.
    """

    def __init__(self, xgboost, node_features, dynamic_features, **kwargs):
        """
        Initialize XGBoostForecast model.
        """

        super(XGBoostForecast, self).__init__(**kwargs)

        self.automatic_optimization = False

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        self.xgboost = xgboost

    def fit_xgboost(self, X, y):

        self.xgboost.fit(X, y)

    def forecast_step(self, model_states, data, t, *args, **kwargs):

        # static graph features
        node_features = torch.cat([data.get(feature).reshape(data.coords.size(0), -1) for
                                   feature in self.node_features], dim=1)

        # dynamic features for current and previous time step
        dynamic_features = torch.cat([tidx_select(data.get(feature), t).reshape(data.coords.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)

        # combined features
        inputs = torch.cat([node_features, dynamic_features], dim=1).detach().numpy()

        # apply XGBoost
        model_states['x'] = torch.tensor(self.xgboost.predict(inputs)).view(-1, 1)

        return model_states

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return None


# class Dynamics(MessagePassing):
#
#     def __init__(self, **kwargs):
#         super(Dynamics, self).__init__(aggr='add', node_dim=0)
#
#         self.use_log_transform = kwargs.get('use_log_transform', False)
#
#         self.scale = kwargs.get('scale', 1.0)
#         self.log_offset = kwargs.get('log_offset', 1e-8)
#
#     def to_raw(self, values):
#
#         if self.use_log_transform:
#             log = values / self.scale
#             raw = torch.exp(log) - self.log_offset
#         else:
#             raw = values / self.scale
#
#         return raw
#
#     def to_log(self, values):
#
#         if self.use_log_transform:
#             log = values / self.scale
#         else:
#             raw = values / self.scale
#             log = torch.log(raw + self.log_offset)
#
#         return log
#
#     def initialize(self, data, *args):
#         pass
#
#     def forward(self, x, hidden, graph_data, t):
#         raise NotImplementedError


class Fluxes(MessagePassing):
    """
    Predicts fluxes for time step t -> t+1, given previous predictions and hidden states.
    """

    def __init__(self, node_features, edge_features, dynamic_features, **kwargs):
        """
        Initialize Fluxes.

        :param node_features: tensor containing all static node features
        :param edge_features: tensor containing all static edge features
        :param dynamic_features: tensor containing all dynamic node features
        :param n_graph_layers: number of graph NN layers to use for hidden representations
        """

        super(Fluxes, self).__init__(aggr='add', node_dim=0)

        self.node_features = node_features
        self.edge_features = edge_features
        self.dynamic_features = dynamic_features

        n_edge_in = sum(edge_features.values()) + 2 * sum(dynamic_features.values())

        # setup model components
        self.edge_mlp = EdgeFluxMLP(n_edge_in, **kwargs)
        #self.input2hidden = torch.nn.Linear(n_edge_in, kwargs.get('n_hidden'), bias=False)
        #self.edge_mlp = MLP(2 * kwargs.get('n_hidden'), 1, **kwargs)
        n_graph_layers = kwargs.get('n_graph_layers', 0)
        self.graph_layers = nn.ModuleList([GraphLayer(**kwargs) for l in range(n_graph_layers)])

        self.use_log_transform = kwargs.get('use_log_transform', False)

        # self.scale = kwargs.get('scale', 1.0)
        # self.log_offset = kwargs.get('log_offset', 1e-8)

        self.transforms = kwargs.get('transforms', [])


    def apply_backward_transforms(self, values: torch.Tensor):

        out = values
        for t in self.transforms:
            out = t.tensor_backward(out)

        return out

    def transformed2raw(self, values):

        #values = torch.clamp(values, min=self.zero_value)
        raw = self.apply_backward_transforms(values)

        return raw


    def forward(self, x, hidden, graph_data, t):
        """
        Predict fluxes for one time step.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        """

        # propagate hidden states through graph to combine spatial information
        hidden_sp = hidden
        for layer in self.graph_layers:
            hidden_sp = layer([graph_data.edge_index, hidden_sp])

        # static graph features
        node_features = torch.cat([graph_data.get(feature).reshape(x.size(0), -1) for
                                   feature in self.node_features], dim=1)
        edge_features = torch.cat([graph_data.get(feature).reshape(graph_data.edge_index.size(1), -1) for
                                   feature in self.edge_features], dim=1)

        # dynamic features for current and previous time step
        dynamic_features_t0 = torch.cat([tidx_select(graph_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)
        dynamic_features_t1 = torch.cat([tidx_select(graph_data.get(feature), t-1).reshape(x.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)

        # message passing through graph
        net_flux = self.propagate(graph_data.edge_index,
                                          reverse_edges=graph_data.reverse_edges,
                                          x=x,
                                          hidden=hidden,
                                          hidden_sp=hidden_sp,
                                          node_features=node_features,
                                          edge_features=edge_features,
                                          dynamic_features_t0=dynamic_features_t0,
                                          dynamic_features_t1=dynamic_features_t1,
                                          areas=graph_data.areas)

        if not self.training:
            if self.use_log_transform:
                raw_net_flux = self.transformed2raw(x) * net_flux
            else:
                raw_net_flux = self.transformed2raw(net_flux)
            self.node_flux = raw_net_flux  # birds/km2 flying in/out of cell i

        return net_flux


    def message(self, x_i, x_j, hidden_sp_j, dynamic_features_t0_i, dynamic_features_t1_j,
                edge_features, reverse_edges, areas_i, areas_j):
        """
        Construct message from node j to node i (for all edges in parallel)

        :param x_i: features of nodes i with shape [#edges, #node_features]
        :param x_j: features of nodes j with shape [#edges, #node_features]
        :param hidden_sp_j: hidden features of nodes j with shape [#edges, #hidden_features]
        :param dynamic_features_t0_i: dynamic features for nodes i with shape [#edges, #features]
        :param dynamic_features_t1_j: dynamic features for nodes j from the previous time step with shape [#edges, #features]
        :param edge_features: edge attributes for edges (j->i) with shape [#edges, #features]
        :param reverse_edges: edge index for reverse edges (i->j)
        :param areas_i: Voronoi cell areas for nodes i
        :param areas_j: Voronoi cell areas for nodes j
        :return: edge fluxes with shape [#edges, 1]
        """

        inputs = [dynamic_features_t0_i, dynamic_features_t1_j, edge_features]
        #inputs = [dynamic_features_t0_i, dynamic_features_t1_j, edge_features, hidden_sp_j]
        inputs = torch.cat(inputs, dim=1)

        #embedding = self.input2hidden(inputs)
        #inputs = torch.cat([embedding, hidden_sp_j], dim=1)

        # total flux from cell j to cell i
        # TODO: use relative face length as input (face length / total cell boundary)
        flux = self.edge_mlp(inputs, hidden_sp_j)
        #flux = self.edge_mlp(inputs)
        #flux = torch.tanh(flux).pow(2)  # between 0 and 1, with initial random outputs close to 0

        # TODO: add flux for self-edges and then use pytorch_geometric.utils.softmax(flux, edge_index[0])
        #  to make sure that mass is conserved
        # flux = flux * x_j * areas_j.view(-1, 1)

        if self.use_log_transform:
            total_i = torch.exp(x_i) * areas_i.view(-1, 1)
            total_j = torch.exp(x_j) * areas_j.view(-1, 1)
            in_flux = flux * total_j / total_i
            out_flux = flux[reverse_edges]
            net_flux = in_flux - out_flux
        else:
            in_flux = flux * x_j * areas_j.view(-1, 1)
            out_flux = in_flux[reverse_edges]
            net_flux = (in_flux - out_flux) / areas_i.view(-1, 1)  # net influx into cell i per km2

        if not self.training:
            # convert to raw quantities
            if self.use_log_transform:
                raw_out_flux = out_flux * self.transformed2raw(x_i) * areas_i.view(-1, 1)
                self.edge_fluxes = raw_out_flux[reverse_edges] - raw_out_flux
            else:
                self.edge_fluxes = self.transformed2raw(in_flux - out_flux)

        return net_flux.view(-1, 1)



class SourceSink(torch.nn.Module):
    """
    Predict source and sink terms for time step t -> t+1, given previous predictions and hidden states.
    """

    def __init__(self, node_features, dynamic_features, **kwargs):
        """
        Initialize RecurrentDecoder module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(SourceSink, self).__init__()

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        n_node_in = sum(self.node_features.values()) + \
                    sum(self.dynamic_features.values()) + \
                    1 + kwargs.get('n_hidden')

        # setup model components
        # self.node_lstm = NodeLSTM(n_node_in, **kwargs)
        #self.source_sink_mlp = SourceSinkMLP(n_node_in, **kwargs)
        self.source_sink_mlp = MLP(n_node_in, 2, **kwargs)

        self.use_log_transform = kwargs.get('use_log_transform', False)

        # self.scale = kwargs.get('scale', 1.0)
        # self.log_offset = kwargs.get('log_offset', 1e-8)

        self.transforms = kwargs.get('transforms', [])

    def apply_backward_transforms(self, values: torch.Tensor):

        out = values
        for t in self.transforms:
            out = t.tensor_backward(out)

        return out

    def transformed2raw(self, values):

        # values = torch.clamp(values, min=self.zero_value)
        raw = self.apply_backward_transforms(values)

        return raw


    def forward(self, x, hidden, graph_data, t, ground_states=None):
        """
        Predict source and sink terms for one time step.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        :param ground_states: estimates of birds on the ground
        """

        # static graph features
        node_features = torch.cat([graph_data.get(feature).reshape(x.size(0), -1) for
                                          feature in self.node_features], dim=1)

        # dynamic features for current time step
        dynamic_features_t0 = torch.cat([tidx_select(graph_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)

        inputs = torch.cat([x.view(-1, 1), node_features, dynamic_features_t0], dim=1)
        inputs = torch.cat([hidden, inputs], dim=1)

        # hidden = self.node_lstm(inputs)
        #source, frac_sink = self.source_sink_mlp(hidden, inputs)
        source_sink = self.source_sink_mlp(inputs)

        if ground_states is None:
            # total density of birds taking off (must be positive)
            source = source_sink[:, 0].view(-1, 1).pow(2)
        else:
            # fraction of birds taking off (between 0 and 1, with initial random outputs close to 0)
            frac_source = torch.tanh(source_sink[:, 0].view(-1, 1)).pow(2)

        # fraction of birds landing (between 0 and 1, with initial random outputs close to 0)
        frac_sink = torch.tanh(source_sink[:, 1].view(-1, 1)).pow(2)


        if self.use_log_transform:
            # both source and sink are fractions (total source/sink divided by current density x)
            if ground_states is not None:
                source = frac_source * torch.exp(ground_states) / torch.exp(x)
                ground_states = ground_states - frac_source + frac_sink * torch.exp(x) / torch.exp(ground_states)

            delta = source - frac_sink

        else:
            # source is the total density while sink is a fraction
            sink = frac_sink * x
            if ground_states is not None:
                source = frac_source * ground_states
                ground_states = ground_states - source + sink
            delta = source - sink


        if not self.training:
            # convert to raw quantities
            if self.use_log_transform:
                raw_x = self.transformed2raw(x)
                # TODO: make sure this conversion is correct
                self.node_source = raw_x * source # birds/km2 taking-off in cell i
                self.node_sink = raw_x * frac_sink # birds/km2 landing in cell i
            else:
                self.node_source = self.transformed2raw(source) # birds/km2 taking-off in cell i
                self.node_sink = self.transformed2raw(sink) # birds/km2 landing in cell i
        else:
            self.node_source = source
            self.node_sink = frac_sink if self.use_log_transform else sink
    
        return delta


class ObservationModel(MessagePassing):

    def __init__(self):
        
        super(ObservationModel, self).__init__(aggr='add', node_dim=0)
        
    def forward(self, cell_states, cells_to_radars):
        
        predictions = self.propagate(cells_to_radars.edge_index, x=cell_states, edge_weight=cells_to_radars.edge_weight)

        return predictions

    def message(self, x_j, edge_weight):
        
        return x_j * edge_weight.view(-1, 1)




class InitialState(MessagePassing):
    """
    Base class for estimating initial bird densities.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize InitialState module.
        """

        super(InitialState, self).__init__(aggr='add', node_dim=0)

        self.use_log_transform = kwargs.get('use_log_transform', False)

        # self.scale = kwargs.get('scale', 1.0)
        # self.log_offset = kwargs.get('log_offset', 1e-8)

        self.transforms = kwargs.get('transforms', [])
        self.zero_value = self.apply_forward_transforms(torch.tensor(0, device=self.device))

    def apply_forward_transforms(self, values: torch.Tensor):

        out = values
        for t in self.transforms:
            out = t.tensor_forward(out)

        return out

    def apply_backward_transforms(self, values: torch.Tensor):

        out = values
        for t in self.transforms:
            out = t.tensor_backward(out)

        return out

    def transformed2raw(self, values):

        # values = torch.clamp(values, min=self.zero_value)
        raw = self.apply_backward_transforms(values)

        return raw

    def forward(self, graph_data, t, *args, **kwargs):
        """
        Determine initial bird densities.

        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index of first forecasting step
        :return x0: initial bird density estimates
        """

        x0 = self.initial_state(graph_data, t, *args, **kwargs)

        if self.use_log_transform:
            x0 = torch.clamp(x0, min=self.zero_value)
        else:
            x0 = x0.pow(2)

        return x0

    def initial_state(self, graph_data, t, *args, **kwargs):

        raise NotImplementedError
    


class RadarToCellInterpolation(InitialState):

    def __init__(self, mlp=None, **kwargs):
        
        super(RadarToCellInterpolation, self).__init__(**kwargs)

        self.mlp = mlp
        
    def initial_state(self, graph_data, t, hidden):
        
        measurements = tidx_select(graph_data['radar'].x, t)
        radars_to_cells = graph_data['radar', 'cell']

        n_radars = graph_data['radar'].num_nodes
        n_cells = graph_data['cell'].num_nodes
        embedded_measurements = torch.cat([measurements.view(n_radars, 1),
                                           torch.zeros((n_cells-n_radars, 1), device=measurements.device)], dim=0)

        cell_states = self.propagate(radars_to_cells.edge_index,
                                     x=embedded_measurements,
                                     edge_weight=radars_to_cells.edge_weight)

        if self.mlp is not None:
            inputs = torch.cat([cell_states, hidden], dim=1)
            cell_states = cell_states + self.mlp(inputs)

        return cell_states

    def message(self, x_j, edge_weight):
        
        return x_j * edge_weight.view(-1, 1)




class InitialStateMLP(InitialState):
    """
    Predict initial bird densities for all cells based on encoder hidden states.
    """

    def __init__(self, node_features, dynamic_features, **kwargs):
        """
        Initialize InitialState module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(InitialStateMLP, self).__init__(**kwargs)

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        n_node_in = sum(self.node_features.values()) + \
                    sum(self.dynamic_features.values()) + \
                    kwargs.get('n_hidden')

        self.mlp = MLP(n_node_in, 1, **kwargs)


    def initial_state(self, graph_data, t, hidden):
        """
        Predict initial bird densities.

        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index of first forecasting step
        :param hidden: updated hidden states for all cells and time points
        :return x0: initial state estimate
        """

        cell_data = graph_data['cell']
        num_nodes = cell_data.num_nodes

        # static graph features
        node_features = torch.cat([cell_data.get(feature).reshape(num_nodes, -1) for
                                          feature in self.node_features], dim=1)

        # dynamic features for current and previous time step
        dynamic_features_t0 = torch.cat([tidx_select(cell_data.get(feature), t).reshape(num_nodes, -1) for
                                         feature in self.dynamic_features], dim=1)

        inputs = torch.cat([node_features, dynamic_features_t0, hidden], dim=1)

        x0 = self.mlp(inputs)

        return x0


class ObservationCopy(InitialState):
    """
    Copies observations to cells.
    """

    def __init__(self, **kwargs):
        """
        Initialize ObservationCopy.
        """

        super(ObservationCopy, self).__init__(**kwargs)


    def initial_state(self, graph_data, t, *args, **kwargs):

        cell_data = graph_data['cell']
        assert hasattr(cell_data, 'x')

        return tidx_select(cell_data.x, t).view(-1, 1)


class GraphInterpolation(InitialState):
    """
    Interpolates values on a partially observed graph.
    """

    def __init__(self, **kwargs):
        """
        Initialize GraphInterpolation.

        :param n_graph_layers: number of message passing rounds to use
        """

        super(GraphInterpolation, self).__init__(**kwargs)

        self.n_graph_layers = kwargs.get('n_graph_layers', 1)

    def initial_state(self, graph_data, t, *args, **kwargs):
        """
        Interpolate graph signals.

        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index of first forecasting step
        :return x0: initial state estimate
        """

        cell_data = graph_data['cell']

        assert hasattr(cell_data, 'x')

        observation_mask = tidx_select(cell_data.missing, t)
        validity = tidx_select(cell_data.missing, t)

        # propagate data through graph
        for _ in range(self.n_graph_layers):
            # message passing through graph
            x, validity = self.propagate(graph_data['cell', 'cell'].edge_index, x=tidx_select(cell_data.x, t), mask=observation_mask,
                                         validity=validity)#, edge_weight=graph_data.edge_weight)

        return x

    def message(self, x_j, validity_j): #, edge_weight):
        """
        Construct message from node j to node i (for all edges in parallel)
        """

        # message from node j to node i
        value = x_j * validity_j # * edge_weight
        weight = validity_j # * edge_weight

        return value, weight

    def update(self, agg_out, x, mask):
        agg_value, agg_weight = agg_out

        # fix observed nodes
        x_observed = mask * x

        # weighted average for unobserved nodes
        validity = agg_weight > 0
        agg_weight = agg_weight + torch.logical_not(validity)  # to avoid division by zero
        x_unobserved = torch.logical_not(mask) * validity * agg_value / agg_weight

        x = x_observed + x_unobserved

        return x, validity


class RecurrentDecoder(torch.nn.Module):
    """
    Recurrent neural network predicting the hidden states during forecasting.
    """

    def __init__(self, node_features, dynamic_features, **kwargs):
        """
        Initialize RecurrentDecoder module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(RecurrentDecoder, self).__init__()

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        n_node_in = sum(node_features.values()) + sum(dynamic_features.values()) + 1

        self.node_lstm = NodeLSTM(n_node_in, **kwargs)

    def initialize(self, h_t, c_t):

        self.node_lstm.setup_states(h_t, c_t)


    def forward(self, x, hidden, graph_data, t):
        """
        Predict next hidden state.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        """

        # static graph features
        node_features = torch.cat([graph_data.get(feature).reshape(x.size(0), -1) for
                                          feature in self.node_features], dim=1)

        # dynamic features for current and previous time step
        dynamic_features_t0 = torch.cat([tidx_select(graph_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)

        #print(x.size(), node_features.size(), dynamic_features_t0.size())
        inputs = torch.cat([x.view(-1, 1), node_features, dynamic_features_t0], dim=1)

        hidden = self.node_lstm(inputs)

        return hidden




# class FluxRGNNTransition(Dynamics):
#     """
#     Implements a single FluxRGNN transition from t to t+1, given the previous predictions and hidden states.
#     """
#
#     def __init__(self, node_features, edge_features, dynamic_features,
#                  n_graph_layers=0, **kwargs):
#         """
#         Initialize FluxRGNNTransition.
#
#         :param node_features: tensor containing all static node features
#         :param edge_features: tensor containing all static edge features
#         :param dynamic_features: tensor containing all dynamic node features
#         :param n_graph_layers: number of graph NN layers to use for hidden representations
#         """
#
#         super(FluxRGNNTransition, self).__init__(**kwargs)
#
#         self.node_features = node_features
#         self.edge_features = edge_features
#         self.dynamic_features = dynamic_features
#
#         # self.use_log_transform = kwargs.get('use_log_transform', False)
#
#         n_node_in = sum(node_features.values()) + sum(dynamic_features.values()) + 1
#         n_edge_in = sum(edge_features.values()) + 2 * sum(dynamic_features.values())
#
#         # setup model components
#         self.node_lstm = NodeLSTM(n_node_in, **kwargs)
#         self.source_sink_mlp = SourceSinkMLP(n_node_in, **kwargs)
#         self.edge_mlp = EdgeFluxMLP(n_edge_in, **kwargs)
#         self.graph_layers = nn.ModuleList([GraphLayer(**kwargs) for l in range(n_graph_layers)])
#
#     def initialize(self, data, h_t, c_t):
#
#         if h_t is None or c_t is None:
#             # start with all zeros
#             h_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=data.device) for
#                    _ in range(self.node_lstm.n_lstm_layers)]
#             c_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=data.device) for
#                    _ in range(self.node_lstm.n_lstm_layers)]
#
#         self.node_lstm.setup_states(h_t, c_t)
#
#     def forward(self, x, hidden, graph_data, t):
#         """
#         Run FluxRGNN prediction for one time step.
#
#         :return x: predicted migration intensities for all cells and time points
#         :return hidden: updated hidden states for all cells and time points
#         :param graph_data: SensorData instance containing information on static and dynamic features
#         :param t: time index
#         """
#
#         # propagate hidden states through graph to combine spatial information
#         hidden_sp = hidden
#         for layer in self.graph_layers:
#             hidden_sp = layer([graph_data.edge_index, hidden_sp])
#
#         # static graph features
#         node_features = torch.cat([graph_data.get(feature).reshape(x.size(0), -1) for
#                                    feature in self.node_features], dim=1)
#         edge_features = torch.cat([graph_data.get(feature).reshape(graph_data.edge_index.size(1), -1) for
#                                    feature in self.edge_features], dim=1)
#
#         # dynamic features for current and previous time step
#         dynamic_features_t0 = torch.cat([graph_data.get(feature)[..., t].reshape(x.size(0), -1) for
#                                          feature in self.dynamic_features], dim=1)
#         dynamic_features_t1 = torch.cat([graph_data.get(feature)[..., t - 1].reshape(x.size(0), -1) for
#                                          feature in self.dynamic_features], dim=1)
#
#         # message passing through graph
#         x, hidden, delta = self.propagate(graph_data.edge_index,
#                                           reverse_edges=graph_data.reverse_edges,
#                                           x=x,
#                                           hidden=hidden,
#                                           hidden_sp=hidden_sp,
#                                           node_features=node_features,
#                                           edge_features=edge_features,
#                                           dynamic_features_t0=dynamic_features_t0,
#                                           dynamic_features_t1=dynamic_features_t1,
#                                           areas=graph_data.areas)
#
#         return x, hidden, delta
#
#     def message(self, x_i, x_j, hidden_sp_j, dynamic_features_t0_i, dynamic_features_t1_j,
#                 edge_features, reverse_edges, areas_i, areas_j):
#         """
#         Construct message from node j to node i (for all edges in parallel)
#
#         :param x_i: features of nodes i with shape [#edges, #node_features]
#         :param x_j: features of nodes j with shape [#edges, #node_features]
#         :param hidden_sp_j: hidden features of nodes j with shape [#edges, #hidden_features]
#         :param dynamic_features_t0_i: dynamic features for nodes i with shape [#edges, #features]
#         :param dynamic_features_t1_j: dynamic features for nodes j from the previous time step with shape [#edges, #features]
#         :param edge_features: edge attributes for edges (j->i) with shape [#edges, #features]
#         :param reverse_edges: edge index for reverse edges (i->j)
#         :param areas_i: Voronoi cell areas for nodes i
#         :param areas_j: Voronoi cell areas for nodes j
#         :return: edge fluxes with shape [#edges, 1]
#         """
#
#         inputs = [dynamic_features_t0_i, dynamic_features_t1_j, edge_features]
#         inputs = torch.cat(inputs, dim=1)
#
#         # total flux from cell j to cell i
#         # TODO: use relative face length as input (face length / total cell boundary)
#         flux = self.edge_mlp(inputs, hidden_sp_j)
#
#         # TODO: add flux for self-edges and then use pytorch_geometric.utils.softmax(flux, edge_index[0])
#         #  to make sure that mass is conserved
#         # flux = flux * x_j * areas_j.view(-1, 1)
#
#         if self.use_log_transform:
#             total_i = torch.exp(x_i) * areas_i.view(-1, 1)
#             total_j = torch.exp(x_j) * areas_j.view(-1, 1)
#             in_flux = flux * total_j / total_i
#             out_flux = flux[reverse_edges]
#             net_flux = in_flux - out_flux
#             raw_out_flux = out_flux * self.to_raw(x_i) * areas_i.view(-1, 1)
#             raw_net_flux = raw_out_flux[reverse_edges] - raw_out_flux
#         else:
#             in_flux = flux * x_j * areas_j.view(-1, 1)
#             out_flux = in_flux[reverse_edges]
#             net_flux = (in_flux - out_flux) / areas_i.view(-1, 1)  # net influx into cell i per km2
#             raw_net_flux = self.to_raw(in_flux - out_flux)
#
#         if not self.training:
#             self.edge_fluxes = raw_net_flux
#
#         return net_flux.view(-1, 1)
#
#     def update(self, aggr_out, x, node_features, dynamic_features_t0, areas):
#         """
#         Aggregate all received messages (fluxes) and combine them with local source/sink
#         terms into a single prediction per node.
#
#         :param aggr_out: sum of incoming messages (fluxes)
#         :param x: local densities from previous time step
#         :param node_features: tensor containing all static node features
#         :param dynamic_features_t0: tensor containing all dynamic node features
#         :param areas: Voronoi cell areas
#         :return: prediction and updated hidden states for all nodes
#         """
#
#         inputs = torch.cat([x.view(-1, 1), node_features, dynamic_features_t0], dim=1)
#
#         hidden = self.node_lstm(inputs)
#         source, sink = self.source_sink_mlp(hidden, inputs)
#
#         if self.use_log_transform:
#             # both source and sink are fractions (total source/sink divided by current density x)
#             delta = source - sink
#             raw_x = self.to_raw(x)
#             raw_source = raw_x * source
#             raw_sink = raw_x * sink
#             raw_node_flux = raw_x * aggr_out
#         else:
#             # source is the total density while sink is a fraction
#             delta = source - sink * x
#             raw_source = self.to_raw(source)
#             raw_sink = self.node_sink = self.to_raw(x) * sink
#             raw_node_flux = aggr_out
#
#         if not self.training:
#             self.node_source = raw_source  # birds/km2 taking-off in cell i
#             self.node_sink = raw_sink  # birds/km2 landing in cell i
#             self.node_flux = raw_node_flux  # birds/km2 flying in/out of cell i
#
#         influx = aggr_out
#
#         pred = x + delta + influx
#
#         return pred, hidden, delta




# class LSTMTransition(Dynamics):
#     """
#     Implements a single LSTM transition from t to t+1, given the previous predictions and hidden states.
#     """
#
#     def __init__(self, node_features, dynamic_features, *args, **kwargs):
#         """
#         Initialize LSTMransition.
#
#         :param node_features: tensor containing all static node features
#         :param dynamic_features: tensor containing all dynamic node features
#         """
#
#         super(LSTMTransition, self).__init__(**kwargs)
#
#         self.node_features = node_features
#         self.dynamic_features = dynamic_features
#
#         n_node_in = sum(node_features.values()) + sum(dynamic_features.values()) + 1
#
#         # setup model components
#         self.node_lstm = NodeLSTM(n_node_in, **kwargs)
#         self.source_sink_mlp = SourceSinkMLP(n_node_in, **kwargs)
#
#
#     def initialize(self, data, h_t, c_t):
#         if h_t is None or c_t is None:
#             # start with all zeros
#             h_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=data.device) for
#                    _ in range(self.node_lstm.n_lstm_layers)]
#             c_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=data.device) for
#                    _ in range(self.node_lstm.n_lstm_layers)]
#
#         self.node_lstm.setup_states(h_t, c_t)
#
#
#     def forward(self, x, hidden, graph_data, t):
#         """
#         Run LSTM prediction for one time step.
#
#         :return x: predicted migration intensities for all cells and time points
#         :return hidden: updated hidden states for all cells and time points
#         :param graph_data: SensorData instance containing information on static and dynamic features for one time sequence
#         :param t: time index
#         """
#
#         # static graph features
#         node_features = torch.cat([graph_data.get(feature).reshape(x.size(0), -1) for
#                                    feature in self.node_features], dim=1)
#
#         # dynamic features for current time step
#         dynamic_features = torch.cat([graph_data.get(feature)[..., t].reshape(x.size(0), -1) for
#                                       feature in self.dynamic_features], dim=1)
#
#         inputs = torch.cat([x.view(-1, 1), node_features, dynamic_features], dim=1)
#
#         hidden = self.node_lstm(inputs)
#         source, sink = self.source_sink_mlp(hidden, inputs)
#
#         if self.use_log_transform:
#             # both source and sink are fractions (total source/sink divided by current density x)
#             delta = source - sink
#             raw_source = self.to_raw(x) * source
#             raw_sink = self.to_raw(x) * sink
#         else:
#             # source is the total density while sink is a fraction
#             delta = source - sink * x
#             raw_source = self.to_raw(source)
#             raw_sink = self.node_sink = self.to_raw(x) * sink
#
#         if not self.training:
#             self.node_source = raw_source  # birds/km2 taking-off in cell i
#             self.node_sink = raw_sink  # birds/km2 landing in cell i
#
#         x = x + delta
#
#
#         return x, hidden, delta


class EdgeFluxMLP(torch.nn.Module):
    """MLP predicting relative movements (between 0 and 1) along the edges of a graph."""

    def __init__(self, n_in, **kwargs):
        super(EdgeFluxMLP, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.activation = kwargs.get('activation', torch.nn.ReLU())

        self.input2hidden = torch.nn.Linear(n_in, self.n_hidden, bias=False)
        #self.fc_edge_in = torch.nn.Linear(self.n_hidden * 2, self.n_hidden)
        #self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
        #                                     for _ in range(self.n_fc_layers - 1)])
        
        #self.hidden2output = torch.nn.Linear(self.n_hidden, 1)

        self.edge_mlp = MLP(self.n_hidden * 2, 1, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.input2hidden)
        #init_weights(self.fc_edge_in)
        #self.fc_edge_hidden.apply(init_weights)
        #init_weights(self.hidden2output)
        #self.edge_mlp.apply(init_weights)

    def forward(self, inputs, hidden_j):
        inputs = self.input2hidden(inputs)
        inputs = torch.cat([inputs, hidden_j], dim=1)

        flux = self.edge_mlp(inputs)

        #flux = self.fc_edge_in(inputs)
        #flux = self.activation(flux)

        # flux = F.dropout(flux, p=self.dropout_p, training=self.training, inplace=False)

        #for layer in self.fc_edge_hidden:
        #    flux = layer(flux)
        #    flux = self.activation(flux)
        #    flux = F.dropout(flux, p=self.dropout_p, training=self.training, inplace=False)

        # map hidden state to relative flux
        #flux = self.hidden2output(flux)
        # flux = torch.sigmoid(flux) # fraction of birds moving from cell j to cell i
        flux = torch.tanh(flux).pow(2) # between 0 and 1, with initial random outputs close to 0
        
        return flux


class SourceSinkMLP(torch.nn.Module):
    """MLP predicting local source and sink terms"""

    def __init__(self, n_in, **kwargs):
        super(SourceSinkMLP, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.hidden2sourcesink = torch.nn.Sequential(torch.nn.Linear(self.n_hidden + n_in, self.n_hidden),
                                                     torch.nn.Dropout(p=self.dropout_p),
                                                     torch.nn.LeakyReLU(),
                                                     torch.nn.Linear(self.n_hidden, 2))

        self.reset_parameters()

    def reset_parameters(self):
        self.hidden2sourcesink.apply(init_weights)

    def forward(self, hidden, inputs):
        inputs = torch.cat([hidden, inputs], dim=1)

        source_sink = self.hidden2sourcesink(inputs)

        source = source_sink[:, 0].view(-1, 1).pow(2) # total density of birds taking off (per km2)

        sink = torch.tanh(source_sink[:, 1].view(-1, 1)).pow(2) # between 0 and 1, with initial random outputs close to 0

        return source, sink


class DeltaMLP(torch.nn.Module):
    """MLP predicting local delta terms"""

    def __init__(self, n_in, **kwargs):
        super(DeltaMLP, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden + n_in, self.n_hidden),
                                                     torch.nn.Dropout(p=self.dropout_p),
                                                     torch.nn.LeakyReLU(),
                                                     torch.nn.Linear(self.n_hidden, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.hidden2delta.apply(init_weights)

    def forward(self, hidden, inputs):
        inputs = torch.cat([hidden, inputs], dim=1)

        delta = self.hidden2delta(inputs).view(-1, 1)

        return delta


class MLP(torch.nn.Module):
    """Simple MLP"""

    def __init__(self, n_in, n_out, activation, **kwargs):
        super(MLP, self).__init__()

        n_hidden = kwargs.get('n_hidden', 64)
        n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.activation = activation
        #self.layer_norm = torch.nn.LayerNorm(n_hidden)
        #self.layer_norm = torch.nn.BatchNorm1d(n_hidden)


        self.input2hidden = torch.nn.Linear(n_in, n_hidden)
        self.hidden_layers = nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden)
                                             for _ in range(n_fc_layers - 1)])
        self.hidden2output = torch.nn.Linear(n_hidden, n_out)

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        self.hidden_layers.apply(init_weights)
        init_weights(self.hidden2output)

    def forward(self, inputs):

        hidden = self.input2hidden(inputs)
        #hidden = self.layer_norm(hidden)
        hidden = F.dropout(hidden, p=self.dropout_p, training=self.training, inplace=False) 
        hidden = self.activation(hidden)

        for layer in self.hidden_layers:
            hidden = layer(hidden)
            #hidden = self.layer_norm(hidden)
            hidden = F.dropout(hidden, p=self.dropout_p, training=self.training, inplace=False)
            hidden = self.activation(hidden)

        output = self.hidden2output(hidden)

        return output



class NodeLSTM(torch.nn.Module):
    """Decoder LSTM combining hidden states with additional inputs."""

    def __init__(self, n_in, **kwargs):
        super(NodeLSTM, self).__init__()

        self.n_in = n_in
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 2)
        self.use_encoder = kwargs.get('use_encoder', True)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # node embedding
        self.input2hidden = torch.nn.Linear(self.n_in, self.n_hidden, bias=False)

        if self.use_encoder:
            self.lstm_in = torch.nn.LSTMCell(self.n_hidden * 2, self.n_hidden)
        else:
            self.lstm_in = torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_lstm_layers - 1)])

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        init_weights(self.lstm_in)
        self.lstm_layers.apply(init_weights)

    def setup_states(self, h, c):
        self.h = h
        self.c = c
        self.alphas = []
        self.enc_state = h[-1]

    def get_alphas(self):
        alphas = torch.stack(self.alphas)
        return alphas

    def get_hidden(self):
        return self.h[-1]

    def forward(self, inputs):

        inputs = self.input2hidden(inputs)

        if self.use_encoder:
            inputs = torch.cat([inputs, self.enc_state], dim=1)

        # lstm layers
        self.h[0], self.c[0] = self.lstm_in(inputs, (self.h[0], self.c[0]))
        for l in range(self.n_lstm_layers - 1):
            self.h[0] = F.dropout(self.h[0], p=self.dropout_p, training=self.training, inplace=False)
            self.c[0] = F.dropout(self.c[0], p=self.dropout_p, training=self.training, inplace=False)
            self.h[l + 1], self.c[l + 1] = self.lstm_layers[l](self.h[l], (self.h[l + 1], self.c[l + 1]))

        return self.h[-1]



class GraphLayer(MessagePassing):
    """
    Message passing layer for further propagation of features through the graph.

    This could help capturing long-range dependencies of fluxes on environmental conditions in non-adjacent cells.
    """

    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', node_dim=0)

        # model settings
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # model components
        self.fc_edge = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.fc_node = torch.nn.Linear(self.n_hidden, self.n_hidden)

        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.fc_edge)
        init_weights(self.fc_node)

    def forward(self, data):
        edge_index, inputs = data

        # message passing through graph
        out = self.propagate(edge_index, inputs=inputs)

        return out

    def message(self, inputs_i):
        # construct messages to node i for each edge (j,i)
        out = self.fc_edge(inputs_i)
        out = F.relu(out)

        return out

    def update(self, aggr_out):
        out = self.fc_node(aggr_out)

        return out


class Extrapolation(MessagePassing):
    """Boundary model that extrapolates features from inner (observed) cells/nodes to unobserved boundary cells."""

    def __init__(self, edge_index=None):
        super(Extrapolation, self).__init__(aggr='mean', node_dim=0)

        self.edge_index = edge_index

    def initialize(self, cell_edges):
        self.edge_index = cell_edges.edge_index[:, torch.logical_not(cell_edges.boundary2boundary_edges)]

    def forward(self, var):
        var = self.propagate(self.edge_index, var=var)
        return var

    def message(self, var_j):
        return var_j


class RecurrentEncoder(torch.nn.Module):
    """Encoder LSTM extracting relevant information from sequences of past environmental conditions and system states"""

    def __init__(self, node_features, dynamic_features, **kwargs):
        super(RecurrentEncoder, self).__init__()

        self.t_context = kwargs.get('context', 24)
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        n_node_in = sum(node_features.values()) + sum(dynamic_features.values())

        self.input2hidden = torch.nn.Linear(n_node_in, self.n_hidden, bias=False)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_lstm_layers)])

        self.reset_parameters()

    def reset_parameters(self):

        self.lstm_layers.apply(init_weights)
        init_weights(self.input2hidden)

    def forward(self, data, t0=0):
        """Run encoder until the given number of context time steps has been reached."""

        # initialize lstm variables
        h_t = [torch.zeros(data.num_nodes, self.n_hidden, device=data.coords.device) for _ in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.num_nodes, self.n_hidden, device=data.coords.device) for _ in range(self.n_lstm_layers)]

        node_features = torch.cat([data.get(feature).reshape(data.num_nodes, -1) for
                                   feature in self.node_features], dim=1)
        # TODO: push node_features through GNN or MLP to extract spatial representations?
        
        for tidx in range(self.t_context):
            
            t = tidx + t0
            
            # dynamic features for current time step
            dynamic_features = torch.cat([tidx_select(data.get(feature), t).reshape(data.num_nodes, -1) for
                                          feature in self.dynamic_features], dim=1)
            inputs = torch.cat([node_features, dynamic_features], dim=1)

            h_t, c_t = self.update(inputs, h_t, c_t)

        return h_t, c_t

    def update(self, inputs, h_t, c_t):
        """Include information on the current time step into the hidden state."""

        inputs = self.input2hidden(inputs)

        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l - 1] = F.dropout(h_t[l - 1], p=self.dropout_p, training=self.training, inplace=False)
            c_t[l - 1] = F.dropout(c_t[l - 1], p=self.dropout_p, training=self.training, inplace=False)
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        return h_t, c_t


class LocalLSTM(torch.nn.Module):
    """FluxRGNN variant without spatial fluxes."""

    def __init__(self, n_env, coord_dim=2, **kwargs):
        super(LocalLSTM, self).__init__()

        self.horizon = kwargs.get('horizon', 40)
        self.t_context = max(1, kwargs.get('context', 1))
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)
        self.use_encoder = kwargs.get('use_encoder', True)

        # model components
        n_in = n_env + coord_dim + 2
        if self.use_encoder:
            self.encoder = RecurrentEncoder(n_in, **kwargs)
        self.node_lstm = NodeLSTM(n_in, **kwargs)
        self.output_mlp = SourceSinkMLP(n_in, **kwargs)


    def forward(self, data):

        x = data.x[..., self.t_context - 1].view(-1, 1)
        y_hat = []

        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
            self.node_lstm.setup_states(h_t, c_t)  # , enc_states)
        else:
            # start from scratch
            h_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=x.device) for
                   _ in range(self.node_lstm.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=x.device) for
                   _ in range(self.node_lstm.n_lstm_layers)]
            self.node_lstm.setup_states(h_t, c_t)

        forecast_horizon = range(self.t_context, self.t_context + self.horizon)

        if not self.training:
            self.node_source = torch.zeros((data.x.size(0), 1, self.horizon), device=x.device)
            self.node_sink = torch.zeros((data.x.size(0), 1, self.horizon), device=x.device)

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < self.teacher_forcing:
                x = data.x[..., t - 1].view(-1, 1)

            inputs = torch.cat([x.view(-1, 1), data.coords, data.env[..., t], data.areas.view(-1, 1)], dim=1)

            hidden = self.node_lstm(inputs)
            source, sink = self.output_mlp(hidden, inputs)
            sink = sink * x
            x = x + source - sink
            y_hat.append(x)

            if not self.training:
                self.node_source[..., t - self.t_context] = source
                self.node_sink[..., t - self.t_context] = sink

        prediction = torch.cat(y_hat, dim=-1)

        return prediction



class LSTM(torch.nn.Module):
    """Standard LSTM taking all observed/predicted bird densities and environmental features as input to LSTM"""

    def __init__(self, **kwargs):

        super(LSTM, self).__init__()

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_in = 5 + kwargs.get('n_env', 4)
        self.n_nodes = kwargs.get('n_nodes', 22)
        self.n_layers = kwargs.get('n_layers', 1)
        self.force_zeros = kwargs.get('force_zeros', False)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        self.fc_in = torch.nn.Linear(self.n_in*self.n_nodes, self.n_hidden)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden) for l in range(self.n_layers)])
        self.fc_out = torch.nn.Linear(self.n_hidden, self.n_nodes)


    def forward(self, data):

        x = data.x[:, 0]
        h_t = [torch.zeros(1, self.n_hidden, device=x.device) for l in range(self.n_layers)]
        c_t = [torch.zeros(1, self.n_hidden, device=x.device) for l in range(self.n_layers)]

        y_hat = [x]
        for t in range(self.horizon):
            r = torch.rand(1)
            if r < self.teacher_forcing:
                x = data.x[:, t]

            # use both bird prediction/observation and environmental features as input to LSTM
            inputs = torch.cat([data.coords.flatten(),
                                data.env[..., t+1].flatten(),
                                data.local_dusk[:, t].float().flatten(),
                                data.local_dawn[:, t+1].float().flatten(),
                                x], dim=0).view(1, -1)

            # multi-layer LSTM
            inputs = self.fc_in(inputs)
            h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
            for l in range(1, self.n_layers):
                h_t[l], c_t[l] = self.lstm_layers[l](h_t[l-1], (h_t[l], c_t[l]))

            x = x + self.fc_out(h_t[-1]).tanh().view(-1)

            if self.force_zeros:
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t+1]

            y_hat.append(x)

        return torch.stack(y_hat, dim=1)


# class MLP(torch.nn.Module):
#     """
#     Standard MLP mapping concatenated features of all nodes at time t to migration intensities
#     of all nodes at time t
#     """
#
#     def __init__(self, **kwargs):
#         super(MLP, self).__init__()
#
#         in_channels = kwargs.get('n_env') + kwargs.get('coord_dim', 2)
#         hidden_channels = kwargs.get('n_hidden', 64)
#         out_channels = kwargs.get('out_channels', 1)
#         horizon = kwargs.get('horizon', 1)
#         n_layers = kwargs.get('n_fc_layers', 1)
#         dropout_p = kwargs.get('dropout_p', 0.0)
#
#         self.fc_in = torch.nn.Linear(in_channels, hidden_channels)
#         self.fc_hidden = nn.ModuleList([torch.nn.Linear(hidden_channels, hidden_channels) for _ in range(n_layers - 1)])
#         self.fc_out = torch.nn.Linear(hidden_channels, out_channels)
#         self.horizon = horizon
#         self.dropout_p = dropout_p
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#
#         init_weights(self.fc_in)
#         init_weights(self.fc_out)
#         self.fc_hidden.apply(init_weights)
#
#     def forward(self, data):
#
#         y_hat = []
#         for t in range(self.horizon + 1):
#
#             features = torch.cat([data.coords.flatten(),
#                                   data.env[..., t].flatten()], dim=0)
#             x = self.fc_in(features)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_p, training=self.training)
#
#             for l in self.fc_hidden:
#                 x = l(x)
#                 x = F.relu(x)
#                 x = F.dropout(x, p=self.dropout_p, training=self.training)
#
#             x = self.fc_out(x)
#             x = x.sigmoid()
#
#             # for locations where it is night: set birds in the air to zero
#             x = x * data.local_night[:, t]
#
#             y_hat.append(x)
#
#         return torch.stack(y_hat, dim=1)
#

class LocalMLP(torch.nn.Module):
    """Standard MLP mapping concatenated features of a single nodes at time t to migration intensities at time t."""

    def __init__(self, n_env, coord_dim=2, **kwargs):
        super(LocalMLP, self).__init__()

        self.horizon = kwargs.get('horizon', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.use_acc = kwargs.get('use_acc_vars', False)
        self.n_layers = kwargs.get('n_fc_layers', 1)
        self.force_zeros = kwargs.get('force_zeros', False)

        self.n_in = n_env + coord_dim + self.use_acc * 2

        self.fc_in = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)


    def forward(self, data):

        y_hat = []

        for t in range(self.horizon):

            x = self.step(data.coords, data.env[..., t], acc=data.acc[..., t])

            if self.force_zeros:
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def step(self, coords, env, acc):
        # use only location-specific features to predict migration intensities
        if self.use_acc:
            features = torch.cat([coords, env, acc], dim=1)
        else:
            features = torch.cat([coords, env], dim=1)
        x = F.relu(self.fc_in(features))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = F.relu(l(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)
        x = x.relu()

        return x


def init_weights(m):
    """Initialize model weights with Kaiming method for relu activations"""
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


def encode_position(pos):

    return torch.cat([torch.sin(pos), torch.cos(pos)])


def tidx_select(features, indices, steps=0):
        
    shape = torch.tensor(features.size()) # [graphs * nodes, ..., time steps]
    
    if indices.size(0) == 1:
        full_indices = indices.repeat(shape[0], 1) # [graphs * nodes, 1)
    else:
        full_indices = indices.repeat(1, int(shape[0] / indices.size(0))).view(-1, 1) # [graphs * nodes, 1]
    #if len(shape) > 2:
    #    full_indices = full_indices.repeat(1, torch.prod(shape[1:-1])).view(-1, 1)

    #full_indices = full_indices.view(-1)

    tidx = torch.arange(shape[-1], device=features.device).view(1, -1).repeat(shape[0], 1)
   
    #print(tidx.device, features.device, full_indices.device)
    mask = torch.logical_and(tidx >= full_indices, tidx <= full_indices + steps)

    if len(shape) > 2:
        dim1 = torch.prod(shape[1:-1])
        f = features.view(shape[0], -1, shape[-1])[mask.view(shape[0], 1, shape[-1]).repeat(1, dim1, 1)]
        #f = f.view(*shape[:-1], -1)
    else:
        f = features[mask]


    #f = features.view(-1, shape[-1])[torch.arange(torch.prod(shape[:-1]), device=features.device), full_indices]
    
    f = f.view(*shape[:-1], -1)
    
    #print(f'original shape = {shape}, after selection = {f.size()}')

    return f


def angle(x1, y1, x2, y2):
    """
    Compute angle between point (x1, y1) and point (x2, y2).

    :return: angle in degrees
    """
    y = y1 - y2
    x = x1 - x2
    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360
    return deg

def distance(x1, y1, x2, y2):
    """
    Compute distance between point (x1, y1) and point (x2, y2).

    Coordinates should be given in the local CRS.

    :return: distance in kilometers
    """
    return np.linalg.norm(np.array([x1-x2, y1-y2])) / 10**3

def MSE(output, gt):
    """Compute mean squared error."""
    return torch.mean((output - gt)**2)


def train(model, train_loader, optimizer, loss_func, device, teacher_forcing=0, **kwargs):
    """Train model using the given optimizer and loss function."""

    model.train()
    loss_all = 0
    #flux_loss_weight = kwargs.get('flux_loss_weight', 0)
    for nidx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if hasattr(model, 'teacher_forcing'):
            model.teacher_forcing = teacher_forcing
        output = model(data)
        gt = data.y

        # if flux_loss_weight > 0:
        #     penalty = flux_penalty(model, data, flux_loss_weight)
        # else:
        #     penalty = 0

        if kwargs.get('force_zeros', False):
            mask = torch.logical_and(data.local_night, torch.logical_not(data.missing))
        else:
            mask = torch.logical_not(data.missing)

        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]

        loss = loss_func(output, gt, mask) #+ penalty
        loss_all += data.num_graphs * float(loss)
        loss.backward()
        optimizer.step()

        del loss, output

    return loss_all


def test(model, test_loader, loss_func, device, **kwargs):
    """Run model on test data and compute loss function for each forecasting horizon separately."""

    model.eval()
    loss_all = []

    if hasattr(model, 'teacher_forcing'):
        model.teacher_forcing = 0

    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        gt = data.y

        if kwargs.get('fixed_boundary', False):
            output = output[~data.boundary]
            gt = gt[~data.boundary]

        if kwargs.get('force_zeros', False):
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing

        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]

        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t]).detach()
                                      for t in range(model.horizon)]))

    return torch.stack(loss_all)


# def flux_penalty(model, data, weight):
#     """Compute penalty for fluxes that do not obey mass-balance."""
#
#     inferred_fluxes = model.local_fluxes.squeeze()
#     inferred_fluxes = inferred_fluxes - inferred_fluxes[data.reverse_edges]
#     observed_fluxes = data.fluxes[..., model.t_context:].squeeze()
#
#     diff = observed_fluxes - inferred_fluxes
#     diff = torch.square(observed_fluxes) * diff  # weight timesteps with larger fluxes more
#
#     edges = data.boundary2inner_edges + data.inner2boundary_edges + data.inner_edges
#     diff = diff[edges]
#     penalty = (torch.square(diff[~torch.isnan(diff)])).mean()
#     penalty = weight * penalty
#
#     return penalty
