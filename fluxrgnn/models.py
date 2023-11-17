import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import MessagePassing, inits
from torch_geometric.utils import to_dense_adj
import pytorch_lightning as pl
import numpy as np
from fluxrgnn import utils


class ForecastModel(pl.LightningModule):
    """
    Base model class for forecasting bird migration
    """

    def __init__(self, **kwargs):
        super(ForecastModel, self).__init__()

        # save config
        self.save_hyperparameters()
        self.config = kwargs

        # forecasting settings
        self.horizon = max(1, kwargs.get('horizon', 1))
        self.t_context = max(1, kwargs.get('context', 1))
        self.tf_start = min(1.0, kwargs.get('teacher_forcing', 1.0))

        self.use_log_transform = kwargs.get('use_log_transform', False)
        self.scale = kwargs.get('scale', 1.0)
        self.log_offset = kwargs.get('log_offset', 1e-8)
        
        if self.use_log_transform:
            self.zero_value = np.log(self.log_offset) * self.scale
        else:
            self.zero_value = 0

    def forecast(self, data, teacher_forcing=0):
        """
        Setup prediction for given data and run model until the max forecasting horizon is reached.

        :param data: SensorData instance containing information on static and dynamic features for one time sequence
        :param teacher_forcing: teacher forcing probability
        :return: predicted migration intensities for all cells and time points
        """

        # initialize forecast
        model_states = self.initialize(data)
        y_hat = []

        # predict until the max forecasting horizon is reached
        forecast_horizon = range(self.t_context, self.t_context + self.horizon)
        for t in forecast_horizon:

            # use gt data instead of model output with probability 'teacher_forcing'
            r = torch.rand(1)
            if r < teacher_forcing:
                model_states['x'] = data.x[..., t - 1].view(-1, 1)

            # make prediction for next time step
            model_states = self.forecast_step(model_states, data, t)
            y_hat.append(model_states['x'])

        prediction = torch.cat(y_hat, dim=-1)

        return prediction


    def initialize(self, data):

        model_states = {'x': data.x[..., self.t_context - 1].view(-1, 1)}

        return model_states


    def forecast_step(self, model_states, data, t):

        raise NotImplementedError


    def training_step(self, batch, batch_idx):

        # get teacher forcing probability for current epoch
        tf = self.tf_start * pow(self.config.get('teacher_forcing_gamma', 1), 
                                 self.current_epoch)
        self.log('teacher_forcing', tf)

        # make predictions and compute loss
        output = self.forecast(batch, teacher_forcing=tf)
        eval_dict = self._eval_step(batch, output, prefix='train')
        self.log_dict(eval_dict)

        return eval_dict['train/loss']


    def validation_step(self, batch, batch_idx):

        # make predictions and compute loss
        prediction = self.forecast(batch)
        eval_dict = self._eval_step(batch, prediction, prefix='val')
        self.log_dict(eval_dict)

        output = {'y_hat': prediction,
                  'y': batch.y if hasattr(batch, 'y') else None}

        return output

    def on_test_epoch_start(self):

        self.test_results = {}
        #self.residuals = []
        self.test_gt = []
        self.test_predictions = []
        self.test_masks = []


    def test_step(self, batch, batch_idx):

        # make predictions and compute evaluation metrics
        prediction = self.forecast(batch)
        eval_dict = self._eval_step(batch, prediction, prefix='test')
        self.log_dict(eval_dict)

        # compute evaluation metrics as a function of the forecasting horizon
        eval_dict_per_t = self._eval_step(batch, prediction, prefix='test', aggregate_time=False)

        for m, values in eval_dict_per_t.items():
            if m in self.test_results:
                self.test_results[m].append(values)
            else:
                self.test_results[m] = [values]

        # gt = batch.y[:, self.t_context: self.t_context + self.horizon]
        # mask = torch.logical_not(batch.missing[:, self.t_context: self.t_context + self.horizon])
        gt = batch.y
        mask = torch.logical_not(batch.missing)
        self.test_gt.append(gt)
        self.test_predictions.append(prediction)
        self.test_masks.append(mask)


    def on_test_epoch_end(self):

        for m, value_list in self.test_results.items():
            self.test_results[m] = torch.concat(value_list, dim=0).reshape(-1, self.horizon)

        # self.residuals = torch.stack(self.residuals, dim=0)
        self.test_gt = torch.stack(self.test_gt)
        self.test_predictions = torch.stack(self.test_predictions)
        self.test_masks = torch.stack(self.test_masks)


    def predict_step(self, batch, batch_idx):

        # make predictions
        output = self.forecast(batch)

        result = {
            'y_hat': self.to_raw(output),
            'y': self.to_raw(batch.y) if hasattr(batch, 'y') else None,
            'local_night': batch.local_night,
            'missing': batch.missing,
            'tidx': batch.tidx
        }
        return result

    def to_raw(self, values):

        values = torch.clamp(values, min=self.zero_value)

        if self.use_log_transform:
            log = values / self.scale
            raw = torch.exp(log) - self.log_offset
        else:
            raw = values / self.scale

        return raw

    def to_log(self, values):

        values = torch.clamp(values, min=self.zero_value)

        if self.use_log_transform:
            log = values / self.scale
        else:
            raw = values / self.scale
            log = torch.log(raw + self.log_offset)

        return log

    def _get_residuals(self, batch, output):
        
        gt = batch.y[:, self.t_context: self.t_context + self.horizon]
        mask = torch.logical_not(batch.missing)[:, self.t_context: self.t_context + self.horizon]

        residuals = (output - gt) * mask

        return residuals

    def _regularizer(self):
        return 0

    def _eval_step(self, batch, output, prefix='', aggregate_time=True):

        if self.config.get('force_zeros', False):
            mask = torch.logical_and(batch.local_night, torch.logical_not(batch.missing))
        else:
            mask = torch.logical_not(batch.missing)

        gt = batch.y[:, self.t_context: self.t_context + self.horizon]
        mask = mask[:, self.t_context: self.t_context + self.horizon]
        
        if aggregate_time:
            gt = gt.reshape(-1)
            mask = mask.reshape(-1)
            output = output.reshape(-1)

        loss = utils.MSE(output, gt, mask)
        
        if self.training:
            regularizer = self._regularizer()
            loss = loss + self.config.get('regularizer_weight', 1.0) * regularizer
            eval_dict = {f'{prefix}/loss': loss,
                         f'{prefix}/regularizer': regularizer}
        else:
            eval_dict = {f'{prefix}/loss': loss}

        if not self.training:
            self._add_eval_metrics(eval_dict, self.to_raw(gt), self.to_raw(output),
                                   mask, prefix=f'{prefix}/raw')
            self._add_eval_metrics(eval_dict, self.to_log(gt), self.to_log(output),
                                   mask, prefix=f'{prefix}/log')
            # print(f'negative predictions = {(output < 0).sum()}')
            # print(f'val mask entries = {mask.sum()}')

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
        #return eval_dict




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get('lr', 0.01))

        scheduler = StepLR(optimizer,
                           step_size=self.config.get('lr_decay', 1000),
                           gamma=self.config.get('lr_gamma', 1))

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


    def initialize(self, data):

        # TODO: use separate topographical / location embedding model whose output is fed to all other model components?

        if self.encoder is not None:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
        else:
            h_t = [torch.zeros(data.num_nodes, self.decoder.n_hidden, device=data.edge_index.device) for
                   _ in range(self.decoder.n_lstm_layers)]
            c_t = [torch.zeros(data.num_nodes, self.decoder.n_hidden, device=data.edge_index.device) for
                   _ in range(self.decoder.n_lstm_layers)]

        self.decoder.initialize(h_t, c_t)

        self.deltas = []

        # setup model components
        if self.boundary_model is not None:
            self.boundary_model.initialize(data)

        # relevant info for later
        if not self.training:
            self.edge_fluxes = torch.zeros((data.edge_index.size(1), 1, self.horizon), device=data.edge_index.device)
            self.node_flux = torch.zeros((data.num_nodes, 1, self.horizon), device=data.edge_index.device)
            self.node_sink = torch.zeros((data.num_nodes, 1, self.horizon), device=data.edge_index.device)
            self.node_source = torch.zeros((data.num_nodes, 1, self.horizon), device=data.edge_index.device)

        # initial system state
        if self.initial_model is not None:
            x = self.initial_model(h_t[-1], data, self.t_context - 1)
        else:
            x = data.x[..., self.t_context - 1].view(-1, 1)

        model_states = {'x': x,
                        'hidden': h_t[-1],
                        'boundary_nodes': data.boundary.view(-1, 1),
                        'inner_nodes': torch.logical_not(data.boundary).view(-1, 1)}

        # if self.ground_model is not None:
        #     model_states['ground_states'] = self.ground_model(h_t)

        return model_states

    def forecast_step(self, model_states, graph_data, t):

        x = model_states['x']
        hidden = model_states['hidden']
        tidx = t - self.t_context

        if self.boundary_model is not None:
            x_boundary = self.boundary_model(x)
            h_boundary = self.boundary_model(hidden)
            x = x * model_states['inner_nodes'] + x_boundary * model_states['boundary_nodes']
            hidden = hidden * model_states['inner_nodes'] + h_boundary * model_states['boundary_nodes']


        # update hidden states
        hidden = self.decoder(x, hidden, graph_data, t)

        # predict movements
        if self.flux_model is not None:

            # predict fluxes between neighboring cells
            net_flux = self.flux_model(x, hidden, graph_data, t)
            x = x + net_flux

            if not self.training:
                # save model component outputs
                self.edge_fluxes[..., tidx] = self.flux_model.edge_fluxes
                self.node_flux[..., tidx] = self.flux_model.node_flux

        if self.source_sink_model is not None:

            # predict local source/sink terms
            delta = self.source_sink_model(x, hidden, graph_data, t,
                                           ground_states=model_states.get('ground_states', None))
            x = x + delta

            if not self.training:
                # save model component outputs
                self.node_source[..., tidx] = self.source_sink_model.node_source
                self.node_sink[..., tidx] = self.source_sink_model.node_sink
            else:
                self.deltas.append(delta)

        if self.config.get('force_zeros', False):
            x = x * graph_data.local_night[..., t]
            x = x + self.zero_value * torch.logical_not(graph_data.local_night[..., t])
        
        model_states['x'] = x
        model_states['hidden'] = hidden

        return model_states

    def _regularizer(self):

        if len(self.deltas) > 0:
            deltas = torch.cat(self.deltas, dim=0)
            penalty = deltas.pow(2).mean()
        else:
            penalty = 0

        return penalty


    def predict_step(self, batch, batch_idx):

        # make predictions
        output = self.to_raw(self.forecast(batch))
        gt = self.to_raw(batch.y) if hasattr(batch, 'y') else None

        # get fluxes along edges
        #adj = to_dense_adj(batch.edge_index, edge_attr=self.edge_fluxes)
        #edge_fluxes = adj.view(batch.num_nodes, batch.num_nodes, -1)

        # get net fluxes per node
        #influxes = edge_fluxes.sum(1)
        #outfluxes = edge_fluxes.permute(1, 0, 2).sum(1)

        #if hasattr(batch, 'fluxes'):
        #    # compute approximate fluxes from radar data
        #    radar_fluxes = to_dense_adj(batch.edge_index, edge_attr=batch.fluxes).view(
        #        batch.num_nodes, batch.num_nodes, -1)
        #else:
        #    radar_fluxes = None

        result = {
            'y_hat': output,
            'y': gt,
            #'influx': influxes,
            #'outflux': outfluxes,
            #'source': self.node_source,
            #'sink': self.node_sink,
            #'edge_fluxes': edge_fluxes,
            #'radar_fluxes': radar_fluxes,
            'local_night': batch.local_night,
            'missing': batch.missing,
            'tidx': batch.tidx
        }
        return result


class LocalMLPForecast(ForecastModel):
    """
    Forecast model using a local MLP with parameters shared across time and space.
    """

    def __init__(self, **kwargs):
        """
        Initialize LocalMLPForecast and all its components.
        """

        super(LocalMLPForecast, self).__init__(**kwargs)

        # setup model
        self.use_acc = kwargs.get('use_acc_vars', False)
        n_in = kwargs.get('n_env', 0) + kwargs.get('coord_dim', 2) + self.use_acc * 2
        self.mlp = NodeMLP(n_in, **kwargs)

        self.use_log_transform = kwargs.get('use_log_transform', False)


    def forecast_step(self, model_states, data, t):

        if self.use_acc:
            inputs = torch.cat([data.coords, data.env[..., t], data.acc[..., t]], dim=1)
        else:
            inputs = torch.cat([data.coords, data.env[..., t]], dim=1)

        x = self.mlp(inputs)

        if not self.use_log_transform:
            x = torch.pow(x, 2)

        if self.config.get('force_zeros', False):
            x = x * data.local_night[..., t]
            x = x + self.zero_value * torch.logical_not(data.local_night[..., t])

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

    def forecast_step(self, model_states, data, t):
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

        self.scale = kwargs.get('scale', 1.0)
        self.log_offset = kwargs.get('log_offset', 1e-8)

    def to_raw(self, values):

        if self.use_log_transform:
            log = values / self.scale
            raw = torch.exp(log) - self.log_offset
        else:
            raw = values / self.scale

        return raw

    def to_log(self, values):

        if self.use_log_transform:
            log = values / self.scale
        else:
            raw = values / self.scale
            log = torch.log(raw + self.log_offset)

        return log


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
        dynamic_features_t0 = torch.cat([graph_data.get(feature)[..., t].reshape(x.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)
        dynamic_features_t1 = torch.cat([graph_data.get(feature)[..., t - 1].reshape(x.size(0), -1) for
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

        if self.use_log_transform:
            raw_net_flux = self.to_raw(x) * net_flux
        else:
            raw_net_flux = self.to_raw(net_flux)

        if not self.training:
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
            raw_out_flux = out_flux * self.to_raw(x_i) * areas_i.view(-1, 1)
            raw_net_flux = raw_out_flux[reverse_edges] - raw_out_flux
        else:
            in_flux = flux * x_j * areas_j.view(-1, 1)
            out_flux = in_flux[reverse_edges]
            net_flux = (in_flux - out_flux) / areas_i.view(-1, 1)  # net influx into cell i per km2
            raw_net_flux = self.to_raw(in_flux - out_flux)

        if not self.training:
            self.edge_fluxes = raw_net_flux

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

        self.scale = kwargs.get('scale', 1.0)
        self.log_offset = kwargs.get('log_offset', 1e-8)

    def to_raw(self, values):

        if self.use_log_transform:
            log = values / self.scale
            raw = torch.exp(log) - self.log_offset
        else:
            raw = values / self.scale

        return raw

    def to_log(self, values):

        if self.use_log_transform:
            log = values / self.scale
        else:
            raw = values / self.scale
            log = torch.log(raw + self.log_offset)

        return log


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

        # dynamic features for current and previous time step
        dynamic_features_t0 = torch.cat([graph_data.get(feature)[..., t].reshape(x.size(0), -1) for
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
            raw_x = self.to_raw(x)
            # TODO: make sure this conversion is correct
            raw_source = raw_x * source
            raw_sink = raw_x * frac_sink
        else:
            # source is the total density while sink is a fraction
            sink = frac_sink * x
            if ground_states is not None:
                source = frac_source * ground_states
                ground_states = ground_states - source + sink
            delta = source - sink
            raw_source = self.to_raw(source)
            raw_sink = self.to_raw(sink)

        if not self.training:
            self.node_source = raw_source  # birds/km2 taking-off in cell i
            self.node_sink = raw_sink  # birds/km2 landing in cell i

        return delta


class InitialState(torch.nn.Module):
    """
    Predict initial bird densities for all cells based on encoder hidden states.
    """

    def __init__(self, node_features, dynamic_features, **kwargs):
        """
        Initialize InitialState module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(InitialState, self).__init__()

        self.node_features = node_features
        self.dynamic_features = dynamic_features

        n_node_in = sum(self.node_features.values()) + \
                    sum(self.dynamic_features.values()) + \
                    kwargs.get('n_hidden')

        self.mlp = MLP(n_node_in, 1, **kwargs)

        self.use_log_transform = kwargs.get('use_log_transform', False)
        self.scale = kwargs.get('scale', 1.0)
        self.log_offset = kwargs.get('log_offset', 1e-8)

        if self.use_log_transform:
            self.zero_value = np.log(self.log_offset) * self.scale
        else:
            self.zero_value = 0


    def forward(self, hidden, graph_data, t):
        """
        Predict initial bird densities.

        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index of first forecasting step
        """

        # static graph features
        node_features = torch.cat([graph_data.get(feature).reshape(graph_data.num_nodes, -1) for
                                          feature in self.node_features], dim=1)

        # dynamic features for current and previous time step
        dynamic_features_t0 = torch.cat([graph_data.get(feature)[..., t].reshape(graph_data.num_nodes, -1) for
                                         feature in self.dynamic_features], dim=1)

        inputs = torch.cat([node_features, dynamic_features_t0, hidden], dim=1)

        x0 = self.mlp(inputs)

        if self.use_log_transform:
            return torch.clamp(x0, min=self.zero_value)
        else:
            return x0.pow(2)



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
        dynamic_features_t0 = torch.cat([graph_data.get(feature)[..., t].reshape(x.size(0), -1) for
                                         feature in self.dynamic_features], dim=1)

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

    def initialize(self, graph_data):

        self.edge_index = graph_data.edge_index[:, torch.logical_not(graph_data.boundary2boundary_edges)]

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

    def forward(self, data):
        """Run encoder until the given number of context time steps has been reached."""

        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for _ in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for _ in range(self.n_lstm_layers)]

        node_features = torch.cat([data.get(feature).reshape(data.x.size(0), -1) for
                                   feature in self.node_features], dim=1)

        for t in range(self.t_context):

            # dynamic features for current time step
            dynamic_features = torch.cat([data.get(feature)[..., t].reshape(data.x.size(0), -1) for
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
