import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch_geometric.nn import MessagePassing, inits
from torch_geometric.nn.models import GAT
from torch_geometric.utils import to_dense_adj, degree, scatter
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.pool import knn
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU, DCRNN
import pytorch_lightning as pl
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from fluxrgnn import utils
from fluxrgnn.transforms import Transforms


class Forecast:

    def __init__(self):

        self.reset()

    def update(self, states: dict):

        for key, value in states.items():
            if key in self.states:
                self.states[key].append(value)
            else:
                self.states[key] = [value]

    def get_state(self, key):

        return self.states[key]

    def finalize(self):

        states = {}
        for key, values in self.states.items():
            states[key] = torch.stack(values, dim=-1)

        return states

    def reset(self):

        self.states = {}



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
        self.min_horizon = max(1, kwargs.get('min_horizon', 1))
        self.t_context = max(1, kwargs.get('context', kwargs.get('test_context', 0)))
        self.tf_start = min(1.0, kwargs.get('teacher_forcing', 0.0))

        # observation model mapping cell densities to radar observations
        self.observation_model = observation_model

        self.use_log_transform = kwargs.get('use_log_transform', False)
        self.scale = kwargs.get('scale', 1.0)
        # self.transforms = [t for t in kwargs.get('transforms', []) if t.feature == 'x']
        self.transforms = Transforms(kwargs.get('transforms', []))

        # self.zero_value = self.apply_forward_transforms(torch.tensor(0))
        # print(f'zero value = {self.zero_value}')

        self.training_coefs = kwargs.get('training_coefs', {'x': 1.0})
        self.test_vars = kwargs.get('test_vars', ['x'])
        self.predict_vars = kwargs.get('predict_vars', ['x'])


    def forecast(self, data, horizon, t0=0, teacher_forcing=0):
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

        # initialize forecast (including prediction for first time step)
        states = self.initialize(data, t0=t0)
        x = states['x']

        forecast = Forecast()
        forecast.update(states)

        # cell_data = data.node_type_subgraph(['cell']).to_homogeneous()
        # radar_data = data['radar']

        # predict until the max forecasting horizon is reached
        forecast_horizon = range(self.t_context + 1, self.t_context + horizon + 1)
        for tidx in forecast_horizon:

            t = t0 + tidx
            x = states['x']

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
                x = tidx_select(interpolated, t - 1).view(-1, 1)
            #else:
            #    print('no ground truth data available to use for teacher forcing')
            
            
            # make prediction for next time step
            states = self.forecast_step(x, data, t, teacher_forcing)

            if self.config.get('force_zeros', False):
                local_night = tidx_select(data['cell'].local_night, t)
                states['x'] = states['x'] * local_night
                states['x'] = states['x'] + self.transforms.zero_value['x'].to(x.device) * \
                                    torch.logical_not(local_night)


            # forecast.append(x)
            forecast.update(states)


        # forecast = torch.cat(forecast, dim=-1)
        # return forecast

        return forecast.finalize()


    def initialize(self, data, t0=0):

        # make prediction for first time step
        # cell_data = data.node_type_subgraph(['cell']).to_homogeneous()
        states = self.forecast_step(None, data, t0 + self.t_context)

        return states


    def forecast_step(self, x, data, t, teacher_forcing=0):

        raise NotImplementedError


    def training_step(self, batch, batch_idx):

        # get teacher forcing probability for current epoch
        tf = self.tf_start * pow(self.config.get('teacher_forcing_gamma', 1), 
                                 self.current_epoch)
        self.log('teacher_forcing', tf)
        
        h_rate = self.config.get('increase_horizon_rate', 0)
        if h_rate > 0:
            epoch = max(0, self.current_epoch - self.config.get('increase_horizon_start', 0))
            horizon = min(self.horizon, int(epoch * h_rate) + self.min_horizon)
        else:
            horizon = self.horizon

        self.log('horizon', horizon)
        
        t0 = torch.randint(0, self.config.get('max_t0', 1), (batch.num_graphs,),
                           device=self.device, requires_grad=False)
        
        # make predictions for all cells
        forecast = self.forecast(batch, horizon, t0=t0, teacher_forcing=tf)

        #compute loss
        loss = 0
        for var, coef in self.training_coefs.items():
            if var in forecast:
                loss_var, eval_dict = self._eval_step(batch, forecast, radar_mask=batch['radar'].train_mask,
                                                      prefix='train', t0=t0, var=var)
                self.log_dict(eval_dict, batch_size=batch.num_graphs)

                loss += coef * loss_var

        regularizer = self.config.get('regularizer_weight', 0.0) * self._regularizer()
        self.log('train/regularizer', regularizer.detach(), batch_size=batch.num_graphs)

        loss += regularizer

        self.log('train/loss', loss.detach(), batch_size=batch.num_graphs, prog_bar=True)
        
        torch.cuda.empty_cache()

        return loss


    def validation_step(self, batch, batch_idx):

        t0 = torch.randint(0, self.config.get('max_t0', 1), (batch.num_graphs,), device=self.device)
        
        # make predictions for all cells
        forecast = self.forecast(batch, self.horizon, t0=t0)

        # evaluate forecast
        for var in (list(self.training_coefs.keys()) + self.test_vars):
            if var in forecast:
                # loss_var, eval_dict = self._eval_step(batch, forecast,
                #                                       radar_mask=batch['radar'].train_mask,
                #                                       prefix='val', t0=t0, var=var)
                # self.log_dict(eval_dict, batch_size=batch.num_graphs)

                # compute evaluation metrics for radars used during training
                _, eval_dict = self._eval_step(batch, forecast,
                                               radar_mask=batch['radar'].train_mask,
                                               prefix='val/observed', t0=t0, var=var)
                self.log_dict(eval_dict, batch_size=batch.num_graphs)

                # compute evaluation metrics for held-out radars
                _, eval_dict = self._eval_step(batch, forecast,
                                               radar_mask=batch['radar'].test_mask,
                                               prefix='val/unobserved', t0=t0, var=var)
                self.log_dict(eval_dict, batch_size=batch.num_graphs)

        # # evaluate forecast
        # _, eval_dict = self._eval_step(batch['radar'], forecast, self.horizon,
        #                             radar_mask=batch['radar'].train_mask, prefix='val', t0=t0)
        # self.log_dict(eval_dict, batch_size=batch.num_graphs)



    def on_test_epoch_start(self):

        self.test_results = {
                'test/mask': [], 
                'test/train_mask': [],
                'test/test_mask': [],
                # 'test/measurements': [],
                # 'test/predictions': [],
                }

        for var in self.test_vars:
            self.test_results[f'test/measurements/{var}'] = []
            self.test_results[f'test/predictions/{var}'] = []


    def test_step(self, batch, batch_idx):

        for t0 in range(self.config.get('max_t0', 1)):

            # make predictions for all cells
            forecast = self.forecast(batch, self.horizon, t0=t0)

            for var in self.test_vars:
                if var in forecast:

                    # compute evaluation metrics for radars used during training
                    _, eval_dict = self._eval_step(batch, forecast,
                                                   radar_mask=batch['radar'].train_mask,
                                                   prefix='test/observed', t0=t0, var=var)
                    self.log_dict(eval_dict, batch_size=batch.num_graphs)

                    # compute evaluation metrics for held-out radars
                    _, eval_dict = self._eval_step(batch, forecast,
                                                   radar_mask=batch['radar'].test_mask,
                                                   prefix='test/unobserved', t0=t0, var=var)
                    self.log_dict(eval_dict, batch_size=batch.num_graphs)


                    if self.observation_model is not None:
                        forecast[var] = self.observation_model(forecast[var],
                                                               batch['cell', 'radar'],
                                                               batch['radar'].num_nodes)
                    
                    predicted = self.transforms.transformed2raw(forecast[var], var)
                    t_steps = predicted.size(-1)

                    measured = self.transforms.transformed2raw(batch['radar'][var], var)
                    measured = measured[..., t0 + self.t_context: t0 + self.t_context + t_steps]
                    
                    self.test_results[f'test/predictions/{var}'].append(predicted.detach())
                    self.test_results[f'test/measurements/{var}'].append(measured.detach())
            
            # # compute evaluation metrics as a function of the forecasting horizon
            # eval_dict_per_t = self._eval_step(radar_data, prediction, self.horizon, prefix='test', aggregate_time=False, t0=t0)
            #
            # for m, values in eval_dict_per_t.items():
            #     if m in self.test_metrics:
            #         self.test_metrics[m].append(values)
            #     else:
            #         self.test_metrics[m] = [values]

            self.test_results['test/mask'].append(
                torch.logical_not(batch['radar'].missing_x)[:, (t0 + self.t_context): (t0 + self.t_context + self.horizon + 1)].detach()
            )
            self.test_results['test/train_mask'].append(batch['radar'].train_mask.detach())
            self.test_results['test/test_mask'].append(batch['radar'].test_mask.detach())




    def on_test_epoch_end(self):

        #for m, value_list in self.test_metrics.items():
        #    self.test_metrics[m] = torch.concat(value_list, dim=0).reshape(-1, self.horizon)

        for m, value_list in self.test_results.items():
            self.test_results[m] = torch.stack(value_list)


    def on_predict_epoch_start(self):

        self.predict_results = {
                #'predict/t_q50': [],
                'predict/tidx': [],
                }

        for var in self.predict_vars:
            self.predict_results[f'predict/predictions/{var}'] = []

    def on_predict_epoch_end(self):

        for m, value_list in self.predict_results.items():
            self.predict_results[m] = torch.stack(value_list)
    
    def predict_step(self, batch, batch_idx):

        for t0 in [0]: #range(self.config.get('max_t0', 1)):
        
            # make predictions for all cells
            forecast = self.forecast(batch, self.horizon, t0=t0)

            for var in self.predict_vars:
                if var in forecast:
                    self.predict_results[f'predict/predictions/{var}'].append(
                        self.transforms.transformed2raw(forecast[var], var).detach()
                    )
            self.predict_results['predict/tidx'].append(batch['cell'].tidx[(t0 + self.t_context): (t0 + self.t_context + self.horizon + 1)].detach())
            #self.predict_results['predict/t_q50'].append(batch['cell'].t_q50[:, (t0 + self.t_context): (t0 + self.t_context + self.horizon + 1)])

            self.add_additional_predict_results()

    
    def add_additional_predict_results(self):

        return 0


    def _regularizer(self):
        return torch.zeros(1, device=self.device)

    def _eval_step(self, graph_data, forecast, radar_mask=None, prefix='', t0=0, var='x'):

        radar_data = graph_data['radar']

        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0, device=self.device)
        t0 = t0.view(-1, 1)

        # map cell predictions to radar locations
        output = self.observation_model(forecast[var], graph_data['cell', 'radar'], radar_data.num_nodes)
        output = output.reshape(radar_data.num_nodes, -1, output.size(-1))
        var_dim = output.size(1)
        horizon = output.size(-1) - 1
        output = output[radar_mask]
        
        missing = radar_data.get(f'missing_{var}', torch.zeros(radar_data.local_night.size(), device=self.device))
        if self.config.get('force_zeros', False):
            local_mask = torch.logical_and(radar_data.local_night, torch.logical_not(missing))
        else:
            local_mask = torch.logical_not(missing)

        gt = tidx_select(radar_data[var], t0 + self.t_context, steps=horizon).view(radar_data.num_nodes, -1)
        gt = gt[radar_mask]
        # gt = gt[radar_mask, self.t_context: self.t_context + horizon].reshape(-1)
        local_mask = tidx_select(local_mask, t0 + self.t_context, steps=horizon).view(radar_data.num_nodes, 1, horizon + 1).repeat(1, var_dim, 1)
        # local_mask = local_mask[radar_mask, self.t_context: self.t_context + horizon].reshape(-1)
        local_mask = local_mask[radar_mask]

        output = output.reshape(-1)
        gt = gt.reshape(-1)
        local_mask = local_mask.reshape(-1)


        if local_mask.sum() == 0:
            # no valid data points
            return 0, {}
        else:
            if self.config.get('root_transformed_loss', False):
                weights = 1 + torch.pow(self.transforms.transformed2raw(gt, var), 0.75)
                loss = utils.MSE(output, gt, local_mask, weights)
            else:
                loss = utils.MSE(output, gt, local_mask)

            # if self.training:
            #     loss = loss + self.config.get('regularizer_weight', 1.0) * self._regularizer()
            #     eval_dict = {f'{prefix}/{var}/loss': loss.detach(),
            #              f'{prefix}/{var}/log-loss': torch.log(loss).detach()}
            #              # f'{prefix}/regularizer': self._regularizer().detach()}
            # else:
            eval_dict = {f'{prefix}/{var}/loss': loss.detach(),
                         f'{prefix}/{var}/log-loss': torch.log(loss).detach()}

            if not self.training:

                raw_gt = self.transforms.transformed2raw(gt, var)
                raw_output = self.transforms.transformed2raw(output, var)
                self._add_eval_metrics(eval_dict, raw_gt, raw_output, local_mask, prefix=f'{prefix}/{var}/raw')

                if var == 'x':
                    log_gt = self.transforms.raw2log(raw_gt)
                    log_output = self.transforms.raw2log(raw_output)
                    self._add_eval_metrics(eval_dict, log_gt, log_output, local_mask, prefix=f'{prefix}/{var}/log')

                    pow_gt = self.transforms.raw2pow(raw_gt)
                    pow_output = self.transforms.raw2pow(raw_output)
                    self._add_eval_metrics(eval_dict, pow_gt, pow_output, local_mask, prefix=f'{prefix}/{var}/pow')

        return loss, eval_dict

    def _add_eval_metrics(self, eval_dict, gt, output, mask, prefix=''):

        # root mean squared error
        rmse = torch.sqrt(utils.MSE(output, gt, mask)).detach()
        eval_dict.update({f'{prefix}/RMSE': rmse})

        # mean absolute error
        mae = utils.MAE(output, gt, mask).detach()
        eval_dict.update({f'{prefix}/MAE': mae})

        # symmetric mean absolute percentage error
        smape = utils.SMAPE(output, gt, mask).detach()
        eval_dict.update({f'{prefix}/SMAPE': smape})

        # mean absolute percentage error
        mape = utils.MAPE(output, gt, mask).detach()
        eval_dict.update({f'{prefix}/MAPE': mape})
        
        # avg residuals
        mean_res = (((output - gt) * mask).sum(0) / mask.sum(0)).detach()
        eval_dict.update({f'{prefix}/mean_residual': mean_res})

        # pseudo R squared (a.k.a "variance explained")
        r2 = utils.R2(output, gt, mask).detach()
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
                 encoder=None, boundary_model=None, initial_model=None,
                 ground_model=None, **kwargs):
        """
        Initialize FluxRGNN and all its components.

        :param dynamics: transition model (e.g. FluxRGNNTransition, LSTMTransition or Persistence)
        :param encoder: encoder model (e.g. RecurrentEncoder)
        :param boundary_model: model handling boundary cells (e.g. Extrapolation)
        :param initial_model: model predicting the initial state
        :param location_encoder: model mapping static cell features to postitional embeddings
        """

        super(FluxRGNN, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.flux_model = flux_model
        self.source_sink_model = source_sink_model
        self.boundary_model = boundary_model
        self.initial_model = initial_model
        # self.location_encoder = location_encoder
        #self.ground_model = ground_model

        self.n_hidden = kwargs.get('n_hidden')
        self.store_fluxes = kwargs.get('store_fluxes', False)


    def initialize(self, graph_data, t0=0):

        # TODO: use separate topographical / location embedding model whose output is fed to all other model components?

        cell_data = graph_data.node_type_subgraph(['cell']).to_homogeneous()

        # if self.location_encoder is not None:
        #     # get cell embeddings
        #     self.cell_embeddings = self.location_encoder(cell_data)
        # else:
        #     self.cell_embeddings = None

        if self.encoder is not None:
            # push context timeseries through encoder to initialize decoder
            rnn_states = self.encoder(graph_data, t0) #, embeddings=self.cell_embeddings)
            self.decoder.initialize(rnn_states)
            #h_t, c_t = self.encoder(graph_data, t0)
            #hidden = h_t[-1]
        else:
            self.decoder.initialize_zeros(cell_data.num_nodes, self.device)
            #if self.decoder is not None:
            #    h_t = [torch.zeros(cell_data.num_nodes, self.n_hidden, device=cell_data.coords.device)
            #       for _ in range(self.decoder.node_rnn.n_layers)]
            #    c_t = [torch.zeros(cell_data.num_nodes, self.n_hidden, device=cell_data.coords.device)
            #       for _ in range(self.decoder.node_rnn.n_layers)]
            #    hidden = h_t[-1]
            #else:
            #    hidden = torch.zeros(cell_data.num_nodes, self.n_hidden, device=cell_data.coords.device)

        
        #if self.decoder is not None:
        #    self.decoder.initialize(h_t, c_t)

        # hidden = h_t[-1] #self.decoder.get_hidden()
        hidden = self.decoder.get_hidden()

        self.regularizers = []

        # setup model components
        if self.boundary_model is not None:
            self.boundary_model.initialize(cell_data)

        # relevant info for later
        if not self.training and self.store_fluxes:
            self.edge_fluxes = []#torch.zeros((cell_edges.edge_index.size(1), 1, self.horizon), device=cell_data.coords.device)
            self.node_velocity = []  # torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)
            self.node_flux = []#torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)
            self.node_sink = []#torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)
            self.node_source = []#torch.zeros((cell_data.num_nodes, 1, self.horizon), device=cell_data.coords.device)

        # predict initial system state
        if self.initial_model is not None:
            x = self.initial_model(graph_data, self.t_context + t0, hidden) #, embeddings=self.cell_embeddings)
            #x = tidx_select(cell_data.x, self.t_context + t0).view(-1, 1)
        #elif hasattr(cell_data, 'x'):
        #    x = cell_data.x[..., self.t_context - 1].view(-1, 1)
        else:
            x = torch.zeros_like(cell_data.coords[:, 0]).unsqueeze(-1)
        #print(f'initial x size = {x.size()}')

        #model_states = {'x': x,
        #                'hidden': hidden,
        #                'hidden_enc': hidden if self.encoder is not None else None,
        #                'boundary_nodes': cell_data.boundary.view(-1, 1),
        #                'inner_nodes': torch.logical_not(cell_data.boundary).view(-1, 1)}

        #if self.ground_model is not None:
        #     #model_states['ground_states'] = self.ground_model(graph_data, self.t_context + t0, h_t[-1])
        #     model_states['ground_states'] = self.apply_forward_transforms(torch.ones_like(x) * 500)
        #     self.regularizers.append(model_states['ground_states'])
        #else:
        #    model_states['ground_states'] = None

        #return model_states

        return {'x': x}

    def forecast_step(self, x, data, t, teacher_forcing=0):

        output = {}

        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()

        boundary_nodes = cell_data.boundary.view(-1, 1)
        inner_nodes = torch.logical_not(boundary_nodes)
        
        # use gt data instead of model output with probability 'teacher_forcing'
        #r = torch.rand(1)
        #if cell_data.x is not None and (r < teacher_forcing):
            # TODO: map measurements in node_storage 'radar' to cells in node_storage 'cell',
            #  or use smaller horizon instead of teacher forcing?
            #x = cell_data.x[torch.arange(cell_data.num_nodes, device=cell_data.x.device), t - 1].view(-1, 1)
        #else:
        #x = model_states['x']


        # update hidden states
        #if self.decoder is not None:
        hidden = self.decoder(x, cell_data, t) #, embeddings=self.cell_embeddings)

        if self.boundary_model is not None:
            h_boundary = self.boundary_model(hidden)
            hidden = hidden * inner_nodes + h_boundary * boundary_nodes

        # predict movements
        if self.flux_model is not None:

            # predict fluxes between neighboring cells
            net_flux = self.flux_model(x, hidden, data, t) #, embeddings=self.cell_embeddings)

            #print(f'avg net flux = {net_flux.mean()}')

            if hasattr(self.flux_model, 'node_velocity'):

                output['bird_uv'] = self.flux_model.node_velocity
                output['fluxes'] = output['bird_uv'] * x
            #if self.training and hasattr(self.flux_model, 'node_velocity'):
            #    #print('add velocity regularizer')
            #    uv_cells = self.flux_model.node_velocity
            #    # output['bird_uv'] = uv_cells


            #    uv_gt = tidx_select(data['radar'].bird_uv, t)
            #    uv_hat = self.observation_model(uv_cells, data['cell', 'radar'], data['radar'].num_nodes)
            #    # uv_hat = uv_hat[:data['radar'].num_nodes]
                
            #    mask = tidx_select(torch.logical_not(data['radar'].missing_bird_uv), t)
            #    mask = mask.reshape(-1) * data['radar'].train_mask.reshape(-1)

            #    uv_error = uv_hat.view(data['radar'].num_nodes, -1) - uv_gt.view(data['radar'].num_nodes, -1)
            #    uv_error = (uv_error * mask.view(-1, 1)).sum(0) / mask.sum(0)
                #uv_error = uv_error[mask]
                
            #    #self.regularizers.append(uv_hat.view(data['radar'].num_nodes, -1) - uv_gt.view(data['radar'].num_nodes, -1))
            #    #print(self.regularizers[-1].size())
            #    self.regularizers.append(uv_error)


            if not self.training and self.store_fluxes:
                # save model component outputs
                self.edge_fluxes.append(self.flux_model.edge_fluxes.detach())
                self.node_flux.append(self.flux_model.node_flux.detach())
                if hasattr(self.flux_model, 'node_velocity'):
                    self.node_velocity.append(self.flux_model.node_velocity.detach())
                    #print(self.flux_model.node_velocity.detach())
        else:
            net_flux = 0

        if self.source_sink_model is not None:

            # predict local source/sink terms
            delta, _ = self.source_sink_model(x, hidden, cell_data, t,
                                              ground_states=None) #,
                                              #embeddings=self.cell_embeddings)

            output['source_sink'] = delta

            if not self.training and self.store_fluxes:
                # save model component outputs
                if hasattr(self.source_sink_model, 'node_source') and hasattr(self.source_sink_model, 'node_sink'):
                    self.node_source.append(self.source_sink_model.node_source.detach())
                    self.node_sink.append(self.source_sink_model.node_sink.detach())
                else:
                    self.node_source.append(delta.detach())
                    self.node_sink.append(-delta.detach())
            #elif ground_states is None:
            if self.training: # and not hasattr(self.flux_model, 'node_velocity'):
                #if hasattr(self.source_sink_model, 'node_source') and hasattr(self.source_sink_model, 'node_sink'):
                #    #reg = self.source_sink_model.node_source * torch.logical_not(torch.logical_and(tidx_select(cell_data.local_night, t), torch.logical_not(tidx_select(cell_data.local_night, t-1))))
                #    #self.regularizers.append(reg)
                #    #print('add source/sink regularizer') 
                #    self.regularizers.append(self.source_sink_model.node_source +
                #                              self.source_sink_model.node_sink)

            
                #    #mask = torch.logical_not(torch.logical_or(tidx_select(cell_data.dusk, t),
                #    #                                          tidx_select(cell_data.dawn, t)))
            
                #    #self.regularizers.append(mask * (self.source_sink_model.node_source +
                #    #                                 self.source_sink_model.node_sink))
                #else:
                #    self.regularizers.append(delta)
                if 'water' in cell_data:
                    self.regularizers.append(delta * cell_data.water.view(-1, 1))
        else:
            delta = 0

        x = x + net_flux + delta
        
        if self.boundary_model is not None:
            # extrapolate densities to boundary cells
            x_boundary = self.boundary_model(x)
            x = x * inner_nodes + x_boundary * boundary_nodes

        output['x'] = x

        return output

    def _regularizer(self):

        if len(self.regularizers) > 0:
            #print(self.regularizers)
            regularizers = torch.cat(self.regularizers, dim=0)
            penalty = regularizers.pow(2).mean()
        else:
            penalty = torch.zeros(1, device=self.device)

        return penalty

    def add_additional_predict_results(self):

        #print('store fluxes?')
        if self.store_fluxes:

            if len(self.node_source) > 0:
                if 'node_source' not in self.predict_results:
                    self.predict_results['node_source'] = []
                self.predict_results['node_source'].append(torch.cat(self.node_source, dim=-1))

            if len(self.node_sink) > 0:
                if 'node_sink' not in self.predict_results:
                    self.predict_results['node_sink'] = []
                self.predict_results['node_sink'].append(torch.cat(self.node_sink, dim=-1))

            if len(self.node_flux) > 0:
                if 'node_flux' not in self.predict_results:
                    self.predict_results['node_flux'] = []
                self.predict_results['node_flux'].append(torch.cat(self.node_flux, dim=-1))

            if len(self.node_velocity) > 0:
                if 'node_velocity' not in self.predict_results:
                    self.predict_results['node_velocity'] = []
                self.predict_results['node_velocity'].append(torch.stack(self.node_velocity, dim=-1))

            #if len(self.edge_fluxes) > 0:
            #    if 'edge_flux' not in self.predict_results:
            #        self.predict_results['edge_flux'] = []
            #    self.predict_results['edge_flux'].append(torch.cat(self.edge_fluxes, dim=-1))


    #def on_predict_start(self):

    #    self.predict_results = {
    #            'prediction': [],
    #            'node_source': [],
    #            'node_sink': [],
    #            'node_flux': [],
    #            'edge_flux': []
    #            }

    #def on_predict_end(self):

    #    for m, value_list in self.predict_results.items():
    #        self.predict_results[m] = torch.stack(value_list)


    #def predict_step(self, batch, batch_idx):

    #    # extract relevant data from batch
    #    cell_data = batch['cell']
    #    radar_data = batch['radar']
        
    #    # make predictions for all cells
    #    prediction = self.forecast(batch, self.horizon)

        # apply observation model to forecast
        #if self.observation_model is not None:
        #    cells_to_radars = batch['cell', 'radar']
        #    prediction = self.observation_model(prediction, cells_to_radars)
        #    prediction = prediction[:radar_data.num_nodes]

    #    self.predict_results['prediction'].append(self.transformed2raw(prediction))
    #    self.predict_results['node_source'].append(self.node_source)
    #    self.predict_results['node_sink'].append(self.node_sink)
    #    self.predict_results['node_flux'].append(self.node_flux)
    #    self.predict_results['edge_flux'].append(self.edge_fluxes)

        #result = {
        #    'predictions': self.to_raw(prediction),
        #    'measurements': self.to_raw(radar_data.x),
        #    'local_night': cell_data.local_night,
        #    'missing': radar_data.missing,
        #    'tidx': cell_data.tidx
        #}
        
    #    return prediction

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

    def __init__(self, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        """
        Initialize LocalMLPForecast and all its components.
        """

        super(LocalMLPForecast, self).__init__(**kwargs)

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        # setup model
        n_in = sum(self.static_cell_features.values()) + sum(self.dynamic_cell_features.values())
        self.mlp = NodeMLP(n_in, **kwargs)


    def forecast_step(self, x, data, t, *args, **kwargs):

        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()

        # static features
        node_features = [cell_data.get(feature).reshape(cell_data.num_nodes, -1) for
                                   feature in self.static_cell_features]

        # dynamic features for current time step t
        dynamic_features = [tidx_select(cell_data.get(feature), t).reshape(cell_data.num_nodes, -1) for
                                      feature in self.dynamic_cell_features]

        # combined features
        inputs = torch.cat(node_features + dynamic_features, dim=1)

        x = self.mlp(inputs)

        if self.config.get('square_output', False):
            x = torch.pow(x, 2)

        if self.config.get('force_zeros', False):
            x = x * tidx_select(cell_data.local_night, t)
            x = x + self.transforms.zero_value['x'].to(x.device) * \
                torch.logical_not(tidx_select(cell_data.local_night, t))

        #model_states['x'] = x

        #return model_states
        return {'x': x}


class RadarToCellForecast(ForecastModel):


    def __init__(self, **kwargs):

        super(RadarToCellForecast, self).__init__(**kwargs)

        if kwargs.get('k', None) is not None:
            self.radar2cell = RadarToCellKNNInterpolation(**kwargs)
        else:
            self.radar2cell = RadarToCellInterpolation(**kwargs)


    def forecast_step(self, x, data, t, *args, **kwargs):

        x = self.radar2cell(data, t)

        #model_states['x'] = x

        #return model_states
        return {'x': x}




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

    def forecast_step(self, x, data, t, *args, **kwargs):
        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()
        #print(data.ridx.device, self.seasonal_patterns.device)
        # get typical density for each radars at the given time point
        # print(data.tidx.min(), data.tidx.max(), self.seasonal_patterns.size())
        x = self.seasonal_patterns[cell_data.cidx, cell_data.tidx[t]].view(-1, 1)

        return {'x': x}

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

    def __init__(self, xgboost, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        """
        Initialize XGBoostForecast model.
        """

        super(XGBoostForecast, self).__init__(**kwargs)

        self.automatic_optimization = False

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        self.xgboost = xgboost

    def fit_xgboost(self, X, y, **kwargs):

        self.xgboost.fit(X, y, **kwargs)

    def forecast_step(self, x, data, t, *args, **kwargs):

        cell_data = data['cell']

        # static graph features
        node_features = [cell_data[feature].reshape(cell_data.num_nodes, -1) for
                                   feature in self.static_cell_features]

        # dynamic features for current and previous time step
        dynamic_features = [tidx_select(cell_data[feature], t).reshape(cell_data.num_nodes, -1) for
                                         feature in self.dynamic_cell_features]

        # combined features
        inputs = torch.cat(node_features + dynamic_features, dim=1).detach().numpy()

        # apply XGBoost
        x = torch.tensor(self.xgboost.predict(inputs)).view(-1, 1)

        return {'x': x}

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

    def __init__(self, edge_features, dynamic_cell_features, **kwargs):
        """
        Initialize Fluxes.

        :param node_features: tensor containing all static node features
        :param edge_features: tensor containing all static edge features
        :param dynamic_features: tensor containing all dynamic node features
        :param n_graph_layers: number of graph NN layers to use for hidden representations
        """

        super(Fluxes, self).__init__(aggr='add', node_dim=0)

        self.edge_features = edge_features
        self.dynamic_cell_features = dynamic_cell_features

        n_edge_in = sum(edge_features.values()) + 2 * sum(dynamic_cell_features.values())

        # setup model components
        self.edge_mlp = EdgeFluxMLP(n_edge_in, **kwargs)
        #self.input2hidden = torch.nn.Linear(n_edge_in, kwargs.get('n_hidden'), bias=False)
        #self.edge_mlp = EdgeFluxMLP(kwargs.get('n_hidden'), **kwargs)
        n_graph_layers = kwargs.get('n_graph_layers', 0)
        self.graph_layers = nn.ModuleList([GraphLayer(**kwargs) for l in range(n_graph_layers)])

        self.use_log_transform = kwargs.get('use_log_transform', False)

        # self.scale = kwargs.get('scale', 1.0)
        # self.log_offset = kwargs.get('log_offset', 1e-8)

        self.transforms = Transforms(kwargs.get('transforms', []))


    def forward(self, x, hidden, graph_data, t):
        """
        Predict fluxes for one time step.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        """

        cell_data = graph_data.node_type_subgraph(['cell']).to_homogeneous()
        
        # propagate hidden states through graph to combine spatial information
        hidden_sp = hidden
        for layer in self.graph_layers:
            hidden_sp = layer([cell_data.edge_index, hidden_sp])

        # static graph features
        edge_features = torch.cat([cell_data.get(feature).reshape(cell_data.edge_index.size(1), -1) for
                                   feature in self.edge_features], dim=1)

        # dynamic features for current and previous time step
        dynamic_features_t0 = torch.cat([tidx_select(cell_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_cell_features], dim=1)
        dynamic_features_t1 = torch.cat([tidx_select(cell_data.get(feature), t-1).reshape(x.size(0), -1) for
                                         feature in self.dynamic_cell_features], dim=1)

        # message passing through graph
        net_flux = self.propagate(cell_data.edge_index,
                                          reverse_edges=cell_data.reverse_edges,
                                          x=x,
                                          hidden=hidden,
                                          hidden_sp=hidden_sp,
                                          edge_features=edge_features,
                                          dynamic_features_t0=dynamic_features_t0,
                                          dynamic_features_t1=dynamic_features_t1,
                                          areas=cell_data.areas)

        if not self.training:
            if self.use_log_transform:
                raw_net_flux = self.transforms.transformed2raw(x) * net_flux
            else:
                raw_net_flux = self.transforms.transformed2raw(net_flux)
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
            #print(f'min net flux: {net_flux.min()}, max net flux: {net_flux.max()}')
            #print(f'min x: {x_j.min()}, max x: {x_j.max()}')
        if not self.training:
            # convert to raw quantities
            if self.use_log_transform:
                raw_out_flux = out_flux * self.transforms.transformed2raw(x_i) * areas_i.view(-1, 1)
                self.edge_fluxes = raw_out_flux[reverse_edges] - raw_out_flux
            else:
                self.edge_fluxes = self.transforms.transformed2raw(in_flux - out_flux)
        
        return net_flux.view(-1, 1)


class NumericalRadarFluxes(MessagePassing):
    """
    Predicts velocities given previous predictions and hidden states,
    and computes corresponding numerical fluxes for time step t -> t+1.
    """

    def __init__(self, radar2cell_model, **kwargs):
        """
        Initialize NumericalRadarFluxes.
        """

        super(NumericalRadarFluxes, self).__init__(aggr='add', node_dim=0)

        self.radar2cell_model = radar2cell_model
        
        self.length_scale = kwargs.get('length_scale', 1.0)
        self.use_log_transform = kwargs.get('use_log_transform', False)

        self.transforms = Transforms(kwargs.get('transforms', []))


    def forward(self, x, hidden, graph_data, t):
        """
        Predict velocities and compute fluxes for one time step.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        """

        assert not self.use_log_transform

        cell_data = graph_data.node_type_subgraph(['cell']).to_homogeneous()
        
        #velocities = tidx_select(radar_data.get('bird_uv'), t).reshape(x.size(0), -1)
        velocities = self.radar2cell_model(graph_data, t)

        # message passing through graph
        net_flux = self.propagate(cell_data.edge_index,
                                          reverse_edges=cell_data.reverse_edges,
                                          x=x,
                                          velocities=velocities,
                                          areas=cell_data.areas,
                                          face_length=cell_data.edge_face_lengths,
                                          edge_normals=cell_data.edge_normals)

        if not self.training:
            raw_net_flux = self.transforms.transformed2raw(net_flux)
            self.node_flux = raw_net_flux  # birds/km2 flying in/out of cell i
        
        self.node_velocity = velocities #* cell_data.length_scale # bird velocity [km/h] if t_unit is 1H
        #print('node velocity = ', self.node_velocity)
        
        return net_flux


    def message(self, x_j, velocities_i, velocities_j,
                edge_normals, reverse_edges, face_length, areas_i):
        """
        Construct message from node j to node i (for all edges in parallel)
        """

        # compute upwind fluxes from cell j to cell i
        edge_velocities = (velocities_i + velocities_j) / 2
        flow = (edge_normals * edge_velocities).sum(1) # velocity in direction of edge (j, i)
        flow = torch.clamp(flow, min=0) # only consider upwind flow
        in_flux = flow.view(-1, 1) * x_j.view(-1, 1) # influx from cell j to cell i [per km]
        out_flux = in_flux[reverse_edges] # outflux from cell i to cell j [per km]
        net_flux = (in_flux - out_flux) * (face_length.view(-1, 1) / (areas_i.view(-1, 1) * self.length_scale))# net flux from j to i
        if not self.training:
            # convert to raw quantities
            self.edge_fluxes = self.transforms.transformed2raw(in_flux - out_flux)

        return net_flux.view(-1, 1)


class NumericalFluxes(MessagePassing):
    """
    Predicts velocities given previous predictions and hidden states,
    and computes corresponding numerical fluxes for time step t -> t+1.
    """

    def __init__(self, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        """
        Initialize NumericalFluxes.

        :param static_cell_features: tensor containing all static node features
        :param edge_features: tensor containing all static edge features
        :param dynamic_cell_features: tensor containing all dynamic node features
        :param n_graph_layers: number of graph NN layers to use for hidden representations
        """

        super(NumericalFluxes, self).__init__(aggr='add', node_dim=0)

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        n_node_in = sum(self.static_cell_features.values()) + sum(self.dynamic_cell_features.values()) #+ kwargs.get('n_hidden')
        self.use_hidden = kwargs.get('use_hidden', True)
        self.length_scale = kwargs.get('length_scale', 1.0)

        # setup model components
        self.input2hidden = torch.nn.Linear(n_node_in, kwargs.get('n_hidden'), bias=False)
        
        if self.use_hidden:
            mlp_in = 2 * kwargs.get('n_hidden')
        else:
            mlp_in = kwargs.get('n_hidden')
        self.velocity_mlp = MLP(mlp_in, 2, **kwargs)
        #self.velocity_mlp = MLP(n_node_in, 2, **kwargs)


        n_graph_layers = kwargs.get('n_graph_layers', 0)
        self.graph_layers = nn.ModuleList([GraphLayer(**kwargs) for l in range(n_graph_layers)])

        self.use_log_transform = kwargs.get('use_log_transform', False)
        self.use_wind = kwargs.get('use_wind', False)

        self.transforms = Transforms(kwargs.get('transforms', []))

        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.input2hidden)

    def forward(self, x, hidden, graph_data, t):
        """
        Predict velocities and compute fluxes for one time step.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        """

        assert not self.use_log_transform

        cell_data = graph_data.node_type_subgraph(['cell']).to_homogeneous()
        
        # propagate hidden states through graph to combine spatial information
        hidden_sp = hidden
        for layer in self.graph_layers:
            hidden_sp = layer([cell_data.edge_index, hidden_sp])

        # static graph features
        node_features = [cell_data.get(feature).reshape(x.size(0), -1) for
                                   feature in self.static_cell_features]
        # dynamic features for current and previous time step
        dynamic_features_t0 = [tidx_select(cell_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_cell_features]

        inputs = node_features + dynamic_features_t0 #+ [hidden_sp]
        inputs = torch.cat(inputs, dim=1)

        inputs = self.input2hidden(inputs)

        if self.use_hidden:
            inputs = torch.cat([inputs, hidden_sp], dim=1)

        velocities = self.velocity_mlp(inputs)

        if self.use_wind:
            wind = tidx_select(cell_data.get('wind'), t).reshape(x.size(0), -1)
            velocities = velocities + wind

        # message passing through graph
        net_flux = self.propagate(cell_data.edge_index,
                                          reverse_edges=cell_data.reverse_edges,
                                          x=x,
                                          hidden=hidden,
                                          hidden_sp=hidden_sp,
                                          node_features=node_features,
                                          dynamic_features_t0=dynamic_features_t0,
                                          velocities=velocities,
                                          areas=cell_data.areas,
                                          face_length=cell_data.edge_face_lengths,
                                          edge_normals=cell_data.edge_normals)

        if not self.training:
            raw_net_flux = self.transforms.transformed2raw(net_flux)
            self.node_flux = raw_net_flux  # birds/km2 flying in/out of cell i
        
        self.node_velocity = velocities #* cell_data.length_scale # bird velocity [km/h] if t_unit is 1H
        
        return net_flux


    def message(self, x_j, velocities_i, velocities_j,
                edge_normals, reverse_edges, face_length, areas_i):
        """
        Construct message from node j to node i (for all edges in parallel)
        """

        # compute upwind fluxes from cell j to cell i
        #print(f'min velocity: {velocities_j.min()}, max velocity: {velocities_j.max()}')
        edge_velocities = (velocities_i + velocities_j) / 2
        flow = (edge_normals * edge_velocities).sum(1) # velocity in direction of edge (j, i)
        flow = torch.clamp(flow, min=0) # only consider upwind flow
        in_flux = flow.view(-1, 1) * x_j.view(-1, 1) # influx from cell j to cell i [per km]
        out_flux = in_flux[reverse_edges] # outflux from cell i to cell j [per km]
        net_flux = (in_flux - out_flux) * (face_length.view(-1, 1) / (areas_i.view(-1, 1) * self.length_scale))# net flux from j to i
        #print(f'min net flux: {net_flux.min()}, max net flux: {net_flux.max()}')
        if not self.training:
            # convert to raw quantities
            self.edge_fluxes = self.transforms.transformed2raw(in_flux - out_flux)

        return net_flux.view(-1, 1)



class SourceSink(torch.nn.Module):
    """
    Predict source and sink terms for time step t -> t+1, given previous predictions and hidden states.
    """

    def __init__(self, model_inputs=None, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        """
        Initialize SourceSink module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(SourceSink, self).__init__()

        self.model_inputs = {} if model_inputs is None else model_inputs
        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        self.use_hidden = kwargs.get('use_hidden', True)

        n_node_in = sum(self.static_cell_features.values()) + \
                    sum(self.dynamic_cell_features.values()) + \
                    sum(self.model_inputs.values())


        # setup model components
        # self.node_lstm = NodeLSTM(n_node_in, **kwargs)
        #self.source_sink_mlp = SourceSinkMLP(n_node_in, **kwargs)
        #self.input_embedding = MLP(n_node_in, kwargs.get('n_hidden'), **kwargs)
        self.input_embedding = torch.nn.Linear(n_node_in, kwargs.get('n_hidden'), bias=False)
        #self.source_sink_mlp = MLP(n_node_in, 2, **kwargs)

        self.use_hidden = kwargs.get('use_hidden', True)
        
        if self.use_hidden:
            mlp_in = 2 * kwargs.get('n_hidden')
        else:
            mlp_in = kwargs.get('n_hidden')
        self.source_sink_mlp = MLP(mlp_in, 2, **kwargs)

        # self.source_sink_mlp = MLP(2 * kwargs.get('n_hidden'), 2, **kwargs)
        # self.source_sink_mlp = MLP(kwargs.get('n_hidden'), 2, **kwargs)

        self.use_log_transform = kwargs.get('use_log_transform', False)

        self.transforms = Transforms(kwargs.get('transforms', []))

        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.input_embedding)


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
        node_features = [graph_data.get(feature).reshape(x.size(0), -1) for
                                          feature in self.static_cell_features]

        # dynamic features for current time step
        dynamic_features_t0 = [tidx_select(graph_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_cell_features]

        inputs = node_features + dynamic_features_t0

        if 'x' in self.model_inputs:
            inputs.append(x.view(-1, 1))
        if 'ground_states' in self.model_inputs:
            inputs.append(ground_states.view(-1, 1))

        inputs = torch.cat(inputs, dim=1)
        inputs = self.input_embedding(inputs)

        if self.use_hidden:
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
            #frac_source = torch.sigmoid(source_sink[:, 1].view(-1, 1))

        # fraction of birds landing (between 0 and 1, with initial random outputs close to 0)
        frac_sink = torch.tanh(source_sink[:, 1].view(-1, 1)).pow(2)
        #frac_sink = torch.sigmoid(source_sink[:, 1].view(-1, 1))

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

            #print(f'avg source = {source.mean()}')
            #print(f'avg sink = {sink.mean()}')


        if not self.training:
            # convert to raw quantities
            if self.use_log_transform:
                raw_x = self.transforms.transformed2raw(x)
                # TODO: make sure this conversion is correct
                self.node_source = raw_x * source # birds/km2 taking-off in cell i
                self.node_sink = raw_x * frac_sink # birds/km2 landing in cell i
            else:
                self.node_source = self.transforms.transformed2raw(source) # birds/km2 taking-off in cell i
                self.node_sink = self.transforms.transformed2raw(sink) # birds/km2 landing in cell i
        else:
            self.node_source = source
            self.node_sink = frac_sink if self.use_log_transform else sink
    
        return delta, ground_states



class DeltaMLP(torch.nn.Module):
    """
    Predict delta for time step t -> t+1, given previous predictions and hidden states.
    """

    def __init__(self, static_cell_features, dynamic_cell_features, **kwargs):
        """
        Initialize DeltaMLP module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(DeltaMLP, self).__init__()

        self.static_cell_features = static_cell_features
        self.dynamic_cell_features = dynamic_cell_features

        n_node_in = sum(self.static_cell_features.values()) + \
                    sum(self.dynamic_cell_features.values()) + \
                    1 + kwargs.get('n_hidden')

        # setup model components
        self.delta_mlp = MLP(n_node_in, 1, **kwargs)


    def forward(self, x, hidden, graph_data, t, ground_states=None):
        """
        Predict delta for one time step.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        :param ground_states: estimates of birds on the ground
        """

        # static graph features
        node_features = torch.cat([graph_data.get(feature).reshape(x.size(0), -1) for
                                   feature in self.static_cell_features], dim=1)

        # dynamic features for current time step
        dynamic_features_t0 = torch.cat([tidx_select(graph_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_cell_features], dim=1)

        inputs = torch.cat([x.view(-1, 1), node_features, dynamic_features_t0], dim=1)
        inputs = torch.cat([hidden, inputs], dim=1)

        # inputs = self.input_embedding(inputs)
        delta = self.delta_mlp(inputs)

        return delta, None


class LocationEncoder(torch.nn.Module):
    """
    Encodes geographical locations and other static cell features such as land cover into hidden representations,
    taking the spatial structure of the tessellation into account.
    """

    def __init__(self, static_cell_features, out_channels, **kwargs):
        """
        Initialize LocationEncoder module.

        :param static_cell_features: mapping from names of static cell features to their dimensionality
        """

        super(LocationEncoder, self).__init__()

        self.static_cell_features = static_cell_features
        self.embedding_dim = out_channels

        n_node_in = sum(self.static_cell_features.values())
        # n_hidden = location_encoder.n_hidden #kwargs.get('n_hidden')

        # setup model components
        # self.input_embedding = torch.nn.Linear(n_node_in, n_hidden, bias=False)

        # self.location_encoder = location_encoder
        self.location_encoder = GAT(n_node_in, out_channels,
                                    num_layers=kwargs.get('n_layers', 1),
                                    dropout=kwargs.get('dropout_p', 0.0))

    def forward(self, cell_data):
        """
        Encode static cell features with GNN.

        :return hidden: hidden representation for all cells
        :param graph_data: SensorData instance containing information on static and dynamic features
        """

        # collect static graph features
        inputs = torch.cat([cell_data.get(feature).reshape(cell_data.num_nodes, -1) for
                                          feature in self.static_cell_features], dim=1)

        # input = self.input_embedding(input)

        embeddings = self.location_encoder(inputs, edge_index=cell_data.edge_index, edge_attr=cell_data.edge_attr) # [n_cells, n_hidden]

        return embeddings


class ObservationModel(MessagePassing):

    def __init__(self):
        
        super(ObservationModel, self).__init__(aggr='add', node_dim=0)
        
    def forward(self, cell_states, cells_to_radars, n_radars):

        weighted_degree = scatter(cells_to_radars.edge_weight, cells_to_radars.edge_index[1],
                                  dim_size=cell_states.size(0), reduce='sum')
        
        predictions = self.propagate(cells_to_radars.edge_index, x=cell_states,
                                     weighted_degree=weighted_degree,
                                     edge_weight=cells_to_radars.edge_weight)

        return predictions[:n_radars]

    def message(self, x_j, weighted_degree_i, edge_weight):
        # from cell j to radar i
        w_ij = (edge_weight / weighted_degree_i)
        return torch.einsum('i,i...->i...', w_ij, x_j)
        # return x_j * edge_weight.view(-1, 1) / weighted_degree_i.view(-1, 1)


#class RadarToCellGNN(MessagePassing):

#    def __init__(self, static_radar_features=None, dynamic_radar_features=None, **kwargs):
#        super(RadarToCellGNN, self).__init__(**kwargs)
#
#        self.static_radar_features = {} if static_radar_features is None else static_radar_features
#        self.dynamic_radar_features = {} if dynamic_radar_features is None else dynamic_radar_features

        # n_node_in = sum(self.dynamic_features.values()) + 1
        #
        # self.mlp = MLP(n_node_in, kwargs.get('n_hidden'), **kwargs)

#    def forward(self, graph_data, t, *args, **kwargs):
#
#        n_radars = graph_data['radar'].num_nodes
#        n_cells = graph_data['cell'].num_nodes
#
#        radars_to_cells = graph_data['radar', 'cell']
#
#        static_features = [graph_data['radar'].get(feature).reshape(n_radars, -1) for
#                           feature in self.static_radar_features]
#
#        dynamic_features = [tidx_select(graph_data['radar'].get(feature), t).reshape(n_radars, -1) for
#                            feature in self.dynamic_radar_features]
#
#        all_features = torch.cat(static_features + dynamic_features, dim=1)
#        dummy_features = torch.zeros((n_cells - n_radars, all_features.size(1)), device=all_features.device)
#        embedded_features = torch.cat([all_features, dummy_features], dim=0)
#
#        weighted_degree = scatter('add', radars_to_cells.edge_weight, radars_to_cells.edge_index[1])
#
#        cell_states = self.propagate(radars_to_cells.edge_index,
#                                     x=embedded_features,
#                                     weighted_degree=weighted_degree,
#                                     edge_weight=radars_to_cells.edge_weight)
#
#        return cell_states

    # def message(self, x_j, edge_weight):
    #     # from radar j to cell i

    #     # TODO: use info about environment (sun elevation, etc) at cell i AND radar j to compute message
    #
    #     inputs = torch.cat([x_j, edge_weight.view(-1, 1)], dim=1)
    #
    #     m_ij = self.mlp(inputs)
    #
    #     return m_ij

#    def message(self, x_j, weighted_degree_i, edge_weight):
#        # from radar j to cell i
#        # compute weighted average of closeby radars
#        return x_j * edge_weight.view(-1, 1) / weighted_degree_i.view(-1, 1)


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

        self.transforms = Transforms(kwargs.get('transforms', []))


    def forward(self, graph_data, t, *args, **kwargs):
        """
        Determine initial bird densities.

        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index of first forecasting step
        :return x0: initial bird density estimates
        """

        x0 = self.initial_state(graph_data, t, *args, **kwargs)
        x0 = torch.clamp(x0, min=self.transforms.zero_value['x'].to(x0.device))

        return x0

    def initial_state(self, graph_data, t, *args, **kwargs):

        raise NotImplementedError
    


class RadarToCellInterpolation(MessagePassing):

    def __init__(self, radar_variables, **kwargs):

        super(RadarToCellInterpolation, self).__init__(aggr='sum', node_dim=0)

        self.radar_variables = radar_variables

        self.n_features = sum(self.radar_variables.values())
        
    def forward(self, graph_data, t, *args, **kwargs):

        radars_to_cells = graph_data['radar', 'cell']
        n_radars = graph_data['radar'].num_nodes
        n_cells = graph_data['cell'].num_nodes

        # dynamic features for current time step
        variables = torch.cat([tidx_select(graph_data['radar'].get(var), t).reshape(n_radars, -1)
                               for var in self.radar_variables], dim=1)

        dummy_variables = torch.zeros((n_cells-n_radars, variables.size(1)), device=variables.device)
        embedded_variables = torch.cat([variables, dummy_variables], dim=0)

        weighted_degree = scatter(radars_to_cells.edge_weight, radars_to_cells.edge_index[1],
                                  dim_size=n_cells, reduce='sum')

        cell_states = self.propagate(radars_to_cells.edge_index,
                                     x=embedded_variables,
                                     weighted_degree=weighted_degree,
                                     edge_weight=radars_to_cells.edge_weight)

        return cell_states

    def message(self, x_j, weighted_degree_i, edge_weight):
        # from radar j to cell i
        
        w_ij = (edge_weight / weighted_degree_i)
        return torch.einsum('i,i...->i...', w_ij, x_j)
        #return x_j * edge_weight.view(-1, 1) / weighted_degree_i.view(-1, 1)


class RadarToCellKNNInterpolation(MessagePassing):

    def __init__(self, radar_variables, k, **kwargs):
        super(RadarToCellKNNInterpolation, self).__init__(aggr='sum', node_dim=0)

        self.radar_variables = radar_variables
        self.k = k

        self.n_features = sum(self.radar_variables.values())

    def forward(self, graph_data, t, *args, **kwargs):

        n_radars = graph_data['radar'].num_nodes
        n_cells = graph_data['cell'].num_nodes
        
        # dynamic features for current time step
        variables = torch.cat([tidx_select(graph_data['radar'].get(var), t).reshape(n_radars, -1)
                               for var in self.radar_variables], dim=1)
        
        # use only valid training radars for interpolation
        missing = tidx_select(graph_data['radar'].missing_x, t).reshape(n_radars)
        radar_mask = torch.logical_and(graph_data['radar'].train_mask, torch.logical_not(missing)) # size [n_radars, time_steps]
        n_radars = radar_mask.sum()

        variables = variables[radar_mask]
        radar_pos = graph_data['radar'].pos[radar_mask]
        cell_pos = graph_data['cell'].pos

        if hasattr(graph_data, 'batch'):
            radar_batch = graph_data['radar'].batch[radar_mask]
            cell_batch = graph_data['cell'].batch
        else:
            radar_batch = None
            cell_batch = None

        # cell_states = knn_interpolate(variables[radar_mask], radar_pos[radar_mask], cell_pos, graph_data['radar'].batch[radar_mask],
        #                               graph_data['cell'].batch, k=self.k)

        edge_index = knn(radar_pos, cell_pos, k=self.k, batch_x=radar_batch, batch_y=cell_batch)
        edge_index = torch.flip(edge_index, dims=[0]) # edges go from radars to cells
        
        dummy_variables = torch.zeros((n_cells - n_radars, variables.size(1)), device=variables.device)
        embedded_variables = torch.cat([variables, dummy_variables], dim=0)

        distance = F.pairwise_distance(radar_pos[edge_index[0]], cell_pos[edge_index[1]], p=2)
        
        #print(f'max distance: {distance.max()}')
        #print(f'min distance: {distance.min()}')
        edge_weight = 1. / (distance + 1e-6)

        weighted_degree = scatter(edge_weight, edge_index[1], dim_size=n_cells, reduce='sum')

        cell_states = self.propagate(edge_index, x=embedded_variables,
                                     weighted_degree=weighted_degree,
                                     edge_weight=edge_weight)

        return cell_states

    def message(self, x_j, weighted_degree_i, edge_weight):
        # from radar j to cell i

        w_ij = (edge_weight / weighted_degree_i)
        return torch.einsum('i,i...->i...', w_ij, x_j)
        # return x_j * edge_weight.view(-1, 1) / weighted_degree_i.view(-1, 1)


class CorrectedRadarToCellInterpolation(MessagePassing):

    def __init__(self, radar_variables, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        super(CorrectedRadarToCellInterpolation, self).__init__(aggr='sum', node_dim=0)

        self.radar_variables = radar_variables

        self.n_features = sum(self.radar_variables.values())

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        n_in = sum(self.static_cell_features.values()) + \
               sum(self.dynamic_cell_features.values()) + \
               kwargs.get('n_hidden') + \
               self.n_features

        self.mlp = MLP(n_in, self.n_features, **kwargs)

    def forward(self, graph_data, t, hidden, *args, **kwargs):
        radars_to_cells = graph_data['radar', 'cell']
        n_radars = graph_data['radar'].num_nodes
        n_cells = graph_data['cell'].num_nodes

        # radar measurements for current time step
        variables = torch.cat([tidx_select(graph_data['radar'].get(var), t).reshape(n_radars, -1)
                               for var in self.radar_variables], dim=1)

        dummy_variables = torch.zeros((n_cells - n_radars, variables.size(1)), device=variables.device)
        embedded_variables = torch.cat([variables, dummy_variables], dim=0)

        weighted_degree = scatter(radars_to_cells.edge_weight, radars_to_cells.edge_index[1],
                                  dim_size=n_cells, reduce='sum')

        cell_states = self.propagate(radars_to_cells.edge_index,
                                     x=embedded_variables,
                                     weighted_degree=weighted_degree,
                                     edge_weight=radars_to_cells.edge_weight)

        # static cell features
        static_cell_features = [graph_data['cell'].get(feature).reshape(n_cells, -1)
                                 for feature in self.static_cell_features]

        # dynamic features for current time step
        dynamic_cell_features = [tidx_select(graph_data['cell'].get(feature), t).reshape(n_cells, -1)
                                  for feature in self.dynamic_cell_features]

        # predict correction term
        cell_inputs = torch.cat([hidden, cell_states] + static_cell_features + dynamic_cell_features, dim=1)
        cell_states = cell_states + self.mlp(cell_inputs)

        return cell_states

    def message(self, x_j, weighted_degree_i, edge_weight):
        # from radar j to cell i

        w_ij = (edge_weight / weighted_degree_i)
        return torch.einsum('i,i...->i...', w_ij, x_j)
        #return x_j * edge_weight.view(-1, 1) / weighted_degree_i.view(-1, 1)


class RadarToCellGNN(MessagePassing):

    def __init__(self, static_radar_features=None, dynamic_radar_features=None,
                 static_cell_features=None, dynamic_cell_features=None, location_encoder=None, **kwargs):

        super(RadarToCellGNN, self).__init__(aggr='sum', node_dim=0)

        self.n_features = kwargs.get('n_hidden')
        self.k = kwargs.get('k', 10)

        self.static_radar_features = {} if static_radar_features is None else static_radar_features
        self.dynamic_radar_features = {} if dynamic_radar_features is None else dynamic_radar_features

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        self.location_encoder = location_encoder

        n_edge_in = sum(self.static_radar_features.values()) + \
                    sum(self.dynamic_radar_features.values()) + \
                    sum(self.static_cell_features.values()) + \
                    sum(self.dynamic_cell_features.values()) + 2 #\
                    #kwargs.get('radar2cell_edge_attr')

        if self.location_encoder is not None:
            n_edge_in += self.location_encoder.embedding_dim

        self.edge_mlp = MLP(n_edge_in, kwargs.get('n_hidden'), **kwargs)
        self.node_mlp = MLP(kwargs.get('n_hidden'), kwargs.get('n_hidden'), **kwargs)

    def forward(self, graph_data, t, hidden, *args, **kwargs):
        # radars_to_cells = graph_data['radar', 'cell']
        n_radars = graph_data['radar'].num_nodes
        n_cells = graph_data['cell'].num_nodes

        # static radar features
        static_radar_features = [graph_data['radar'].get(feature).reshape(n_radars, -1)
                           for feature in self.static_radar_features]

        # dynamic radar features for current time step
        dynamic_radar_features = [tidx_select(graph_data['radar'].get(feature), t).reshape(n_radars, -1)
                               for feature in self.dynamic_radar_features]

        # static cell features
        static_cell_features = [graph_data['cell'].get(feature).reshape(n_cells, -1)
                                 for feature in self.static_cell_features]

        cell_data = graph_data.node_type_subgraph(['cell']).to_homogeneous()

        if self.location_encoder is not None:
            static_cell_features.append(self.location_encoder(cell_data))

        # dynamic cell features for current time step
        dynamic_cell_features = [tidx_select(graph_data['cell'].get(feature), t).reshape(n_cells, -1)
                                  for feature in self.dynamic_cell_features]

        all_cell_features = torch.cat(static_cell_features + dynamic_cell_features, dim=1)

        # use only valid training radars for interpolation
        missing = tidx_select(graph_data['radar'].missing_x, t).reshape(n_radars)
        radar_mask = torch.logical_and(graph_data['radar'].train_mask,
                                       torch.logical_not(missing))  # size [n_radars, time_steps]
        n_radars = radar_mask.sum()
        
        all_radar_features = torch.cat(static_radar_features + dynamic_radar_features, dim=1)
        all_radar_features = all_radar_features[radar_mask]
        dummy_features = torch.zeros((n_cells - n_radars, all_radar_features.size(1)), device=all_radar_features.device)
        embedded_radar_features = torch.cat([all_radar_features, dummy_features], dim=0)

        radar_pos = graph_data['radar'].pos[radar_mask]
        cell_pos = graph_data['cell'].pos

        if hasattr(graph_data, 'batch'):
            radar_batch = graph_data['radar'].batch[radar_mask]
            cell_batch = graph_data['cell'].batch
        else:
            radar_batch = None
            cell_batch = None

        edge_index = knn(radar_pos, cell_pos, k=self.k, batch_x=radar_batch, batch_y=cell_batch)
        edge_index = torch.flip(edge_index, dims=[0])  # edges go from radars to cells

        distance = F.pairwise_distance(radar_pos[edge_index[0]], cell_pos[edge_index[1]], p=2)
        edge_weight = 1. / (distance + 1e-6)

        weighted_degree = scatter(edge_weight, edge_index[1], dim_size=n_cells, reduce='sum')

        cell_states = self.propagate(edge_index,
                                     x_radar=embedded_radar_features,
                                     x_cell=all_cell_features,
                                     weighted_degree=weighted_degree,
                                     edge_weight=edge_weight)

        cell_states = self.node_mlp(cell_states)

        return cell_states

    def message(self, x_radar_j, x_cell_i, edge_weight, weighted_degree_i):
        # message from radar j to cell i

        edge_inputs = torch.cat([x_radar_j, x_cell_i, edge_weight.unsqueeze(1), weighted_degree_i.unsqueeze(1)], dim=1)
        msg = self.edge_mlp(edge_inputs)

        # w_ij = (edge_weight / weighted_degree_i)
        # return torch.einsum('i,i...->i...', w_ij, msg)

        return msg


class InitialStateMLP(InitialState):
    """
    Predict initial bird densities for all cells based on encoder hidden states.
    """

    def __init__(self, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        """
        Initialize InitialState module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(InitialStateMLP, self).__init__(**kwargs)

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        n_node_in = sum(self.static_cell_features.values()) + \
                    sum(self.dynamic_cell_features.values()) + \
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
        node_features = [cell_data.get(feature).reshape(num_nodes, -1) for
                                          feature in self.static_cell_features]

        # dynamic features for current and previous time step
        dynamic_features_t0 = [tidx_select(cell_data.get(feature), t).reshape(num_nodes, -1) for
                                         feature in self.dynamic_cell_features]

        inputs = torch.cat(node_features + dynamic_features_t0 + [hidden], dim=1)

        x0 = self.mlp(inputs)

        if not self.use_log_transform:
            x0 = torch.pow(x0, 2)

        return x0


class ObservationCopy(torch.nn.Module):
    """
    Copies observations to cells.
    """

    def __init__(self, radar_variables, **kwargs):
        """
        Initialize ObservationCopy.
        """

        super(ObservationCopy, self).__init__(**kwargs)

        self.radar_variables = radar_variables

        self.n_features = sum(self.radar_variables.values())

    def forward(self, graph_data, t, *args, **kwargs):

        n_radars = graph_data['radar'].num_nodes
        assert (graph_data['cell'].num_nodes == n_radars)

        # radar measurements for current time step
        radar_data = torch.cat([tidx_select(graph_data['radar'].get(var), t).reshape(n_radars, -1)
                               for var in self.radar_variables], dim=1)

        return radar_data

    #
    # def initial_state(self, graph_data, t, *args, **kwargs):
    #
    #     assert (graph_data['cell'].num_nodes == graph_data['radar'].num_nodes)
    #
    #     return tidx_select(graph_data['radar'].x, t).view(-1, 1)


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

        observation_mask = tidx_select(cell_data.missing_x, t)
        validity = tidx_select(cell_data.missing_x, t)

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

    def __init__(self, node_rnn, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        """
        Initialize RecurrentDecoder module.

        :param node_features: tensor containing all static node features
        :param dynamic_features: tensor containing all dynamic node features
        """

        super(RecurrentDecoder, self).__init__()

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        n_inputs = sum(self.static_cell_features.values()) + sum(self.dynamic_cell_features.values()) + 1

        # if kwargs.get('rnn_type', 'LSTM'):
        #     self.node_rnn = NodeLSTM(n_node_in, **kwargs)
        # else:
        #     self.node_rnn = NodeGRU(n_node_in, **kwargs)

        self.node_rnn = node_rnn(n_inputs, **kwargs)

    def initialize(self, states):
    
        self.node_rnn.setup_states(states)
        self.hidden_enc = self.node_rnn.get_hidden()


    def initialize_zeros(self, batch_size, device):

        self.node_rnn.setup_zero_states(batch_size, device)
        self.hidden_enc = self.node_rnn.get_hidden()

    def get_hidden(self):

        return self.node_rnn.get_hidden()


    def forward(self, x, graph_data, t):
        """
        Predict next hidden state.

        :return x: predicted migration intensities for all cells and time points
        :return hidden: updated hidden states for all cells and time points
        :param graph_data: SensorData instance containing information on static and dynamic features
        :param t: time index
        """

        # static graph features
        node_features = [graph_data.get(feature).reshape(x.size(0), -1) for
                                          feature in self.static_cell_features]

        # dynamic features for current and previous time step
        dynamic_features_t0 = [tidx_select(graph_data.get(feature), t).reshape(x.size(0), -1) for
                                         feature in self.dynamic_cell_features]

        #print(x.size(), node_features.size(), dynamic_features_t0.size())
        inputs = torch.cat([x.view(-1, 1)] + node_features + dynamic_features_t0, dim=1)

        rnn_states = self.node_rnn(inputs, edge_index=graph_data.edge_index, edge_weight=None, hidden=self.hidden_enc)
        hidden = self.node_rnn.get_hidden()

        #h_t, c_t = self.node_rnn(inputs, hidden_enc)

        #hidden = h_t[-1]

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

        self.n_hidden = kwargs.get('n_hidden')
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.activation = kwargs.get('activation', torch.nn.ReLU())

        self.use_hidden = kwargs.get('use_hidden', True)
        #if self.use_hidden:
        #    n_in += self.n_hidden
      
        self.input2hidden = torch.nn.Linear(n_in, self.n_hidden, bias=False)
        
        #self.hidden2output = torch.nn.Linear(self.n_hidden, 1)

        #self.edge_mlp = MLP(self.n_hidden * 2, 1, **kwargs)
        if self.use_hidden:
            mlp_in = 2 * self.n_hidden
        else:
            mlp_in = self.n_hidden
        self.edge_mlp = MLP(mlp_in, 1, **kwargs)
        #self.edge_mlp = MLP(n_in, 1, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.input2hidden)
        #init_weights(self.fc_edge_in)
        #self.fc_edge_hidden.apply(init_weights)
        #init_weights(self.hidden2output)
        #self.edge_mlp.apply(init_weights)

    def forward(self, inputs, hidden):
        inputs = self.input2hidden(inputs)
        if self.use_hidden:
            inputs = torch.cat([inputs, hidden], dim=1)

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

#
# class DeltaMLP(torch.nn.Module):
#     """MLP predicting local delta terms"""
#
#     def __init__(self, n_in, **kwargs):
#         super(DeltaMLP, self).__init__()
#
#         self.n_hidden = kwargs.get('n_hidden', 64)
#         self.dropout_p = kwargs.get('dropout_p', 0)
#
#         self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden + n_in, self.n_hidden),
#                                                      torch.nn.Dropout(p=self.dropout_p),
#                                                      torch.nn.LeakyReLU(),
#                                                      torch.nn.Linear(self.n_hidden, 1))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.hidden2delta.apply(init_weights)
#
#     def forward(self, hidden, inputs):
#         inputs = torch.cat([hidden, inputs], dim=1)
#
#         delta = self.hidden2delta(inputs).view(-1, 1)
#
#         return delta


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
        # TODO: test effect of doing dropout after activation
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

    def __init__(self, n_inputs, **kwargs):
        super(NodeLSTM, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_layers = kwargs.get('n_rnn_layers', 1)
        self.use_encoder = kwargs.get('use_encoder', False)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # node embedding
        self.input2hidden = torch.nn.Linear(n_inputs, self.n_hidden, bias=False)

        n_in_hidden = self.n_hidden * 2 if self.use_encoder else self.n_hidden
        self.lstm_in = torch.nn.LSTMCell(n_in_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_layers - 1)])

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        init_weights(self.lstm_in)
        self.lstm_layers.apply(init_weights)

    def setup_states(self, states):

        #self.h, self.c = states

        self.h = [states[0][l] for l in range(self.n_layers)]
        self.c = [states[1][l] for l in range(self.n_layers)]

    def setup_zero_states(self, batch_size, device):
        self.h = [torch.zeros(batch_size, self.n_hidden, device=device, requires_grad=False) for _ in range(self.n_layers)]
        self.c = [torch.zeros(batch_size, self.n_hidden, device=device, requires_grad=False) for _ in range(self.n_layers)]

    def get_hidden(self):
        return self.h[-1]

    def forward(self, inputs, hidden=None, **kwargs):

        inputs = self.input2hidden(inputs)

        if hidden is not None and self.use_encoder:
            inputs = torch.cat([inputs, hidden], dim=1)

        # lstm layers
        self.h[0], self.c[0] = self.lstm_in(inputs, (self.h[0], self.c[0]))
        for l in range(self.n_layers - 1):
            self.h[l] = F.dropout(self.h[l], p=self.dropout_p, training=self.training, inplace=False)
            self.c[l] = F.dropout(self.c[l], p=self.dropout_p, training=self.training, inplace=False)
            self.h[l + 1], self.c[l + 1] = self.lstm_layers[l](self.h[l], (self.h[l + 1], self.c[l + 1]))

        return self.h, self.c

class NodeGRU(torch.nn.Module):
    """Decoder GRU combining hidden states with additional inputs."""

    def __init__(self, n_inputs, **kwargs):
        super(NodeGRU, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_layers = kwargs.get('n_rnn_layers', 1)
        self.use_encoder = kwargs.get('use_encoder', False)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # node embedding
        self.input2hidden = torch.nn.Linear(n_inputs, self.n_hidden, bias=False)

        n_in_hidden = self.n_hidden * 2 if self.use_encoder else self.n_hidden
        self.gru_in = torch.nn.GRUCell(n_in_hidden, self.n_hidden)
        self.gru_layers = nn.ModuleList([torch.nn.GRUCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_layers - 1)])

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        init_weights(self.gru_in)
        self.gru_layers.apply(init_weights)

    def setup_states(self, h):
        self.h = h

    def setup_zero_states(self, batch_size, device):
        self.h = [torch.zeros(batch_size, self.n_hidden, device=device, requires_grad=False) for _ in range(self.n_layers)]

    def get_hidden(self):
        return self.h[-1]

    def forward(self, inputs, hidden=None, **kwargs):

        inputs = self.input2hidden(inputs)

        if hidden is not None and self.use_encoder:
            inputs = torch.cat([inputs, hidden], dim=1)

        # gru layers
        self.h[0] = self.gru_in(inputs, self.h[0])
        for l in range(self.n_layers - 1):
            self.h[l] = F.dropout(self.h[l], p=self.dropout_p, training=self.training, inplace=False)
            self.h[l + 1] = self.gru_layers[l](self.h[l], self.h[l + 1])

        return self.h


class NodeGConvGRU(torch.nn.Module):
    """Decoder GConvGRU combining hidden states with additional inputs."""

    def __init__(self, n_inputs, **kwargs):
        super(NodeGConvGRU, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.K = kwargs.get('K', 2)
        self.n_layers = kwargs.get('n_rnn_layers', 1)
        self.use_encoder = kwargs.get('use_encoder', False)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # node embedding
        self.input2hidden = torch.nn.Linear(n_inputs, self.n_hidden, bias=False)

        n_in_hidden = self.n_hidden * 2 if self.use_encoder else self.n_hidden
        self.gru_in = GConvGRU(n_in_hidden, self.n_hidden, self.K)
        self.gru_layers = nn.ModuleList([GConvGRU(self.n_hidden, self.n_hidden, self.K)
                                          for _ in range(self.n_layers - 1)])

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        init_weights(self.gru_in)
        self.gru_layers.apply(init_weights)

    def setup_states(self, h):
        self.h = h

    def setup_zero_states(self, batch_size, device):
        self.h = [torch.zeros(batch_size, self.n_hidden, device=device) for _ in range(self.n_layers)]

    def get_hidden(self):
        return self.h[-1]

    def forward(self, inputs, edge_index, edge_weight=None, hidden=None, **kwargs):

        inputs = self.input2hidden(inputs)

        if hidden is not None and self.use_encoder:
            inputs = torch.cat([inputs, hidden], dim=1)

        # gru layers
        self.h[0] = self.gru_in(inputs, edge_index=edge_index, edge_weight=edge_weight, H=self.h[0])
        for l in range(self.n_layers - 1):
            self.h[l] = F.dropout(self.h[l], p=self.dropout_p, training=self.training, inplace=False)
            self.h[l + 1] = self.gru_layers[l](self.h[l], edge_index=edge_index, edge_weight=edge_weight, H=self.h[l + 1])

        return self.h

class NodeGConvGRU_broken(torch.nn.Module):
    """Decoder GConvGRU combining hidden states with additional inputs."""

    def __init__(self, n_inputs, **kwargs):
        super(NodeGConvGRU_broken, self).__init__()

        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_layers = kwargs.get('n_rnn_layers', 1)
        self.use_encoder = kwargs.get('use_encoder', False)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # node embedding
        self.input2hidden = torch.nn.Linear(n_inputs, self.n_hidden, bias=False)

        n_in_hidden = self.n_hidden * 2 if self.use_encoder else self.n_hidden
        #self.layer_in = GConvGRU(n_in_hidden, self.n_hidden, **kwargs)
        #self.hidden_layers = nn.ModuleList([GConvGRU(self.n_hidden, self.n_hidden, **kwargs)
        #                                  for _ in range(self.n_layers - 1)])

        self.layer_in = torch.nn.GRUCell(n_in_hidden, self.n_hidden, **kwargs)
        self.hidden_layers = nn.ModuleList([torch.nn.GRUCell(self.n_hidden, self.n_hidden, **kwargs)
            for _ in range(self.n_layers - 1)])
        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        init_weights(self.layer_in)
        self.hidden_layers.apply(init_weights)

    def setup_states(self, states):

        self.h = states

    def setup_zero_states(self, batch_size, device):
        self.h = [torch.zeros(batch_size, self.n_hidden, device=device) for _ in range(self.n_layers)]

    def get_hidden(self):
        return self.h[-1]

    def forward(self, inputs, edge_index, edge_weight=None, hidden=None):

        inputs = self.input2hidden(inputs)

        if hidden is not None and self.use_encoder:
            inputs = torch.cat([inputs, hidden], dim=1)

        # lstm layers
        self.h[0] = self.layer_in(inputs, self.h[0]) #edge_index=edge_index, edge_weight=edge_weight, H=self.h[0])
        for l in range(self.n_layers - 1):
            self.h[l] = F.dropout(self.h[l], p=self.dropout_p, training=self.training, inplace=False)
            self.h[l + 1] = self.hidden_layers[l](self.h[l], self.h[l+1]) #edge_index=edge_index,
                                                  #edge_weight=edge_weight, H=self.h[l + 1])

        return self.h




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

    def initialize(self, cell_data):
        self.edge_index = cell_data.edge_index[:, torch.logical_not(cell_data.boundary2boundary_edges)]

    def forward(self, var):
        var = self.propagate(self.edge_index, var=var)
        return var

    def message(self, var_j):
        return var_j



class RecurrentEncoder(torch.nn.Module):
    """Encoder LSTM extracting relevant information from sequences of past environmental conditions and system states"""

    def __init__(self, node_rnn, radar2cell_model, location_encoder=None,
                 static_cell_features=None, dynamic_cell_features=None, **kwargs):
        super(RecurrentEncoder, self).__init__()

        self.t_context = kwargs.get('context', 24)
        self.n_hidden = kwargs.get('n_hidden', 64)
        # self.n_lstm_layers = kwargs.get('n_rnn_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.radar2cell_model = radar2cell_model

        self.location_encoder = location_encoder

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        #print(static_cell_features)
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        n_inputs = sum(self.static_cell_features.values()) + \
                    sum(self.dynamic_cell_features.values()) + \
                    self.radar2cell_model.n_features

        if self.location_encoder is not None:
            n_inputs += self.location_encoder.embedding_dim

        # self.input2hidden = torch.nn.Linear(n_node_in, self.n_hidden, bias=False)
        # self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden)
        #                                   for _ in range(self.n_lstm_layers)])

        # if kwargs.get('rnn_type', 'LSTM'):
        #     self.node_rnn = NodeLSTM(n_node_in, **kwargs)
        # else:
        #     self.node_rnn = NodeGRU(n_node_in, **kwargs)

        self.node_rnn = node_rnn(n_inputs, **kwargs)
        #self.node_rnn = NodeGRU(n_inputs, **kwargs)
        # self.reset_parameters()

    # def reset_parameters(self):
    #
    #     self.lstm_layers.apply(init_weights)
    #     init_weights(self.input2hidden)

    def forward(self, data, t0=0):
        """Run encoder until the given number of context time steps has been reached."""

        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()

        self.node_rnn.setup_zero_states(cell_data.num_nodes, device=cell_data.coords.device)

        static_cell_features = [cell_data.get(feature).reshape(cell_data.num_nodes, -1)
                                   for feature in self.static_cell_features]

        if self.location_encoder is not None:
            static_cell_features.append(self.location_encoder(cell_data))

        # process all context time steps and the first forecasting time step
        for tidx in range(self.t_context + 1):
            
            t = tidx + t0
            
            # dynamic features for current time step
            dynamic_cell_features = [tidx_select(cell_data.get(feature), t).reshape(cell_data.num_nodes, -1)
                                          for feature in self.dynamic_cell_features]
            # get radar features and map them to cells
            radar_features = self.radar2cell_model(data, t, self.node_rnn.get_hidden())
            
            inputs = torch.cat(static_cell_features + dynamic_cell_features + [radar_features], dim=1)

            rnn_states = self.node_rnn(inputs, edge_index=cell_data.edge_index, edge_weight=None) # TODO use weights

        return rnn_states


class LSTMEncoder(torch.nn.Module):
    """Encoder LSTM extracting relevant information from sequences of past environmental conditions and system states"""

    def __init__(self, radar2cell_model, static_cell_features=None, dynamic_cell_features=None, **kwargs):
        super(LSTMEncoder, self).__init__()

        self.t_context = kwargs.get('context', 24)
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.radar2cell_model = radar2cell_model

        self.static_cell_features = {} if static_cell_features is None else static_cell_features
        self.dynamic_cell_features = {} if dynamic_cell_features is None else dynamic_cell_features

        n_inputs = sum(self.static_cell_features.values()) + \
                    sum(self.dynamic_cell_features.values()) + \
                    self.radar2cell_model.n_features

        self.node_rnn = torch.nn.LSTM(n_inputs, self.n_hidden, dropout=self.dropout_p, batch_first=True)
        

    def forward(self, data, t0=0):
        """Run encoder until the given number of context time steps has been reached."""

        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()

        static_cell_features = [cell_data.get(feature).reshape(cell_data.num_nodes, -1, 1).repeat(1, 1, self.t_context + 1)
                                   for feature in self.static_cell_features]

        dynamic_cell_features = [tidx_select(cell_data.get(feature), t0, steps=self.t_context).reshape(cell_data.num_nodes, -1, self.t_context + 1) for feature in self.dynamic_cell_features] # [cells, features, time_steps]

        radar_features = torch.cat([self.radar2cell_model(data, t0 + tidx) for tidx in range(self.t_context + 1)], dim=-1).reshape(cell_data.num_nodes, -1, self.t_context + 1)

        
        inputs = torch.cat(static_cell_features + dynamic_cell_features + [radar_features], dim=1)

        inputs = inputs.permute(0, 2, 1) # [cells, time_steps, features]
        
        output, (h_t, c_t) = self.node_rnn(inputs)

        return (h_t, c_t)


class GRUEncoder(torch.nn.Module):
    """Encoder LSTM extracting relevant information from sequences of past environmental conditions and system states"""

    def __init__(self, radar2cell_model, static_cell_features, dynamic_cell_features, **kwargs):
        super(GRUEncoder, self).__init__()

        self.t_context = kwargs.get('context', 24)
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_layers = kwargs.get('n_rnn_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.radar2cell_model = radar2cell_model
        self.static_cell_features = static_cell_features
        self.dynamic_cell_features = dynamic_cell_features

        n_node_in = sum(static_cell_features.values()) + \
                    sum(dynamic_cell_features.values()) + \
                    self.radar2cell_model.n_features

        self.input2hidden = torch.nn.Linear(n_node_in, self.n_hidden, bias=False)

        self.gru_layers = nn.ModuleList([nn.GRUCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_layers)])

        self.reset_parameters()

    def reset_parameters(self):

        self.gru_layers.apply(init_weights)
        init_weights(self.input2hidden)

    def forward(self, data, t0=0):
        """Run encoder until the given number of context time steps has been reached."""

        cell_data = data.node_type_subgraph(['cell']).to_homogeneous()

        # initialize lstm variables
        h_t = [torch.zeros(data.num_nodes, self.n_hidden, device=data.coords.device) for _ in range(self.n_layers)]

        static_cell_features = torch.cat([cell_data.get(feature).reshape(cell_data.num_nodes, -1)
                                          for feature in self.static_cell_features], dim=1)
        # TODO: push node_features through GNN or MLP to extract spatial representations?

        # process all context time steps and the first forecasting time step
        for tidx in range(self.t_context + 1):
            t = tidx + t0

            # dynamic features for current time step
            dynamic_cell_features = torch.cat([tidx_select(cell_data.get(feature), t).reshape(cell_data.num_nodes, -1)
                                               for feature in self.dynamic_cell_features], dim=1)
            # get radar features and map them to cells
            radar_features = self.radar2cell_model(data, t)

            inputs = torch.cat([static_cell_features, dynamic_cell_features, radar_features], dim=1)

            h_t = self.update(inputs, h_t)


            del radar_features
            del dynamic_cell_features
            del inputs

        del cell_data
        del static_cell_features

        return h_t

    def update(self, inputs, h_t):
        """Include information on the current time step into the hidden state."""

        inputs = self.input2hidden(inputs)

        h_t[0] = self.gru_layers[0](inputs, h_t[0])
        for l in range(1, self.n_layers):
            h_t[l - 1] = F.dropout(h_t[l - 1], p=self.dropout_p, training=self.training, inplace=False)
            h_t[l] = self.gru_layers[l](h_t[l - 1], h_t[l])

        del inputs

        return h_t


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

    mask = torch.logical_and(tidx >= full_indices, tidx <= full_indices + steps)

    if len(shape) > 2:
        #print(f'mask: {mask.size()}, features: {features.size()}')
        dim1 = torch.prod(shape[1:-1])
        mask = mask.view(shape[0], 1, shape[-1])
        #print(f'{features.view(shape[0], -1, shape[-1])[mask]}')
        f = features.view(shape[0], -1, shape[-1])[mask.repeat(1, dim1, 1)]
        #f = f.view(*shape[:-1], -1)
        #print(f'features after: {f.size()}')
    else:
        #print(f'mask: {mask.size()}, features: {features.size()}, mask_sum: {mask.sum()}')
        #print(f'steps: {steps}')
        #print(steps * shape[0])
        f = features[mask]

        #print(f'mask per node: {mask.sum(-1)}')


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
            mask = torch.logical_and(data.local_night, torch.logical_not(data.missing_x))
        else:
            mask = torch.logical_not(data.missing_x)

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
            mask = data.local_night & ~data.missing_x
        else:
            mask = ~data.missing_x

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
