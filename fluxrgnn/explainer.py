from fluxrgnn import models, dataloader
import torch
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from scipy.special import binom
import shap
import numpy as np
from math import prod
import copy
import itertools
import warnings
import sklearn



class ForecastExplainer():
    """
    Wrapper around shap.KernelExplainer for spatiotemporal forecast models using SensorHeteroData as input
    """

    def __init__(self, forecast_model, background, feature_names, **kwargs):

        self.forecast_model = forecast_model
        self.forecast_model.eval()

        if len(background.shape) == 3:
            self.background = np.expand_dims(background, 0) # [1, n_features, n_cells, T]
        elif len(background.shape) == 4:
            self.background = background # [n_samples, n_features, n_cells, T]
        else:
            raise Exception("background must have 3 or 4 dimensions")

        self.n_samples, self.n_features, self.n_cells, self.T = self.background.shape
        self.N = self.n_cells * self.T

        self.feature_names = feature_names

        self.t0 = kwargs.get('t0', 0)
        self.horizon = self.forecast_model.horizon
        self.context = self.forecast_model.t_context
        self.node_store = kwargs.get('node_store', 'cell')

        self.explain_processes = kwargs.get('explain_processes', True)
        if self.explain_processes:
            self.output_shape = (self.n_cells, 5, self.horizon) 
            self.output_names = ['x', 'source', 'sink', 'mtr', 'direction']
        else:
            self.output_shape = (self.n_cells, 1, self.horizon)
            self.output_names = ['x']
        self.output_dim = prod(self.output_shape)



    def predict(self, input_graph):
        """
        Generates forecast for given features
        :param input_graph: SensorHeteroData object
        :return: prediction: np.ndarray of shape [n_samples, N]
        """

        forecast = self.forecast_model.forecast(input_graph, self.horizon, t0=self.t0) # [cells, horizon]

        prediction = forecast['x'][..., :self.horizon]

        if self.explain_processes:
            #source = torch.stack(self.forecast_model.node_sink, dim=-1)
            #sink = torch.stack(self.forecast_model.node_source, dim=-1) # [cells, 1, horizon]

            #velocities = torch.stack(self.forecast_model.node_velocity, dim=-1) # [cells, 2, horizon]
            source = torch.clamp(forecast['source_sink'], 0)
            sink = torch.clamp(-forecast['source_sink'], 0)
            
            velocities = forecast['bird_uv']
            speed = torch.linalg.vector_norm(velocities, dim=-2).unsqueeze(1)
            
            direction = (torch.rad2deg(torch.arctan2(velocities[:, 0], velocities[:, 1])) + 360) % 360
            direction = direction.unsqueeze(1)

            #fluxes = prediction * velocities
            mtr = prediction * speed

            prediction = torch.cat([prediction, source, sink, mtr, direction], dim=1) # [n_samples * n_cells, 5, horizon]

        return prediction #.reshape(-1)


    def mask_features(self, binary_mask, input_batch, sample_idx=0):
        """
        Replace masked features with corresponding background samples
        :param binary_mask: np.ndarray of shape [n_features] containing 0's for features that should be replaced
        :param input_graph: SensorHeteroData object containing original features
        :return: updated SensorHeteroData object
        """

        # apply changes to copies of original input data
        masked_input_batch = copy.copy(input_batch)
        indices = np.where(np.logical_not(binary_mask))[0]

        for fidx in indices:
            name = self.feature_names[fidx]
            fbg = self.background[:, fidx].reshape(self.n_samples * self.n_cells, -1, self.T)

            masked_input_batch[self.node_store][name] = torch.tensor(fbg, dtype=input_batch[self.node_store][name].dtype,
                                                       device=input_batch[self.node_store][name].device)


        return masked_input_batch


    def explain(self, input_graph, n_samples=1000):
        """
        Compute SHAP values for given instance

        :param input_graph: SensorHeteroData
        :return: SHAP values
        """

        assert isinstance(input_graph, dataloader.SensorHeteroData)

        # make sure model and data are on the same device
        input_graph = input_graph.to(self.forecast_model.device)

        input_batch = Batch.from_data_list([copy.copy(input_graph) for _ in range(self.n_samples)])
        
        def f(binary_masks):
            # maps feature masks to model outputs and averages over background samples
            n_masks = binary_masks.shape[0]
            out = torch.zeros((n_masks, self.output_dim), device=self.forecast_model.device)
            for i in range(n_masks):
                print(f'retain {binary_masks[i].sum()} features: {binary_masks[i]}')
                masked_input = self.mask_features(binary_masks[i], input_batch)
                pred = self.predict(masked_input) # [n_samples * n_cells, n_outputs, horizon]
                pred = pred.reshape(self.n_samples, -1, pred.size(-2), pred.size(-1))

                if 'direction' in self.output_names:
                    # find minimum angle across background samples
                    directions = pred[:, :, -1, :]
                    min_dir = directions.min(0) # [n_cells, horizon]

                    # find angles that need to wrap around
                    wrap_idx = (directions - min_dir.unsqueeze(0)) > 180
                    directions[wrap_idx] -= 360

                    # write directions back to pred
                    pred[:, :, -1, :] = directions

                    # average over background
                    pred = pred.mean(0)

                    # make sure final directions are between 0 and 360 degrees
                    pred[:, -1, :] = (pred[:, -1, :] + 360) % 360
                else:
                    pred = pred.mean(0)
                
                out[i] = pred.reshape(-1)
                # out[i] = unbatch(pred, masked_input[self.node_store].batch, dim=0) # [n_samples, n_cells * horizon]
                # for j in range(self.n_samples):
                #     masked_input = self.mask_features(binary_masks[i], input_graph, sample_idx=j)
                #     pred = self.predict(masked_input)
                #     out[i, j] = pred
                # TODO: get batch for each node and reshape to size [samples, cells * time] (use ptg.utils.unbatch)

            #out /= self.n_samples
            # print(f'std over predictions = {out.std(1)}')
            # out = out.mean(1) # take sample mean

            #return out # shape [n_masks, N]
            return out.cpu().numpy()


        # setup explainer (compute expectation over background)
        mask_all = np.zeros((1, self.n_features)) # mask for overall background
        explainer = shap.KernelExplainer(f, mask_all)

        #explainer = KernelExplainer(f, self.n_features, self.forecast_model.device)

        # compute SHAP values
        mask_none = np.ones((1, self.n_features)) # mask for actual prediction
        shap_values = explainer.shap_values(mask_none, nsamples=n_samples, silent=True)
        shap_values = np.stack(shap_values, axis=0)

        result = {'shap_values': shap_values.reshape(*self.output_shape, self.n_features),
                 'expected_values': explainer.expected_value.reshape(self.output_shape),
                 'actual_values': explainer.fx.reshape(self.output_shape),
                 'permuted_values': explainer.ey.reshape(-1, *self.output_shape),
                 'feature_names': self.feature_names,
                 'output_names': self.output_names
                }
        
        #for name in self.feature_names:
        #    result[name] = input_graph[self.node_store][name][:, self.t0 + self.context: self.t0 + self.context + self.horizon]

        return result

    # def explain(self, features, background):
    #     """
    #     Compute SHAP values for given features
    #
    #     :param features: np.ndarray of shape [n_samples, n_features, T * n_cells]
    #     :param background: np.ndarray of shape [n_samples, n_features, T * n_cells]
    #     :return: SHAP values
    #     """
    #
    #     # setup explainer (compute expectation over background)
    #     explainer = shap.KernelExplainer(self.predict, background)
    #
    #     # compute SHAP values
    #     shap_values = explainer.shap_values(features, nsamples=1000)
    #


def construct_background(dataset, feature_names, reduction='sampling', n_samples=100, **kwargs):

    seed = kwargs.get('seed', 1234)
    node_store = kwargs.get('node_store', 'cell')

    if reduction == 'sampling':
        # use a limited number of samples as background
        subset = random_split(dataset, (n_samples, len(dataset) - n_samples),
                              generator=torch.Generator().manual_seed(seed))[0]
    else:
        # use entire dataset as background
        subset = dataset

    background = []
    for graph_data in subset:
        features = []
        for name in feature_names:
            features.append(graph_data[node_store][name])
        features = torch.stack(features, dim=0)
        background.append(features)
    background = torch.stack(background, dim=0) # shape [subset_size, n_features, n_cells, T]

    if reduction == 'mean':
        background = background.mean(0).unsqueeze(0)

    return background #.cpu().numpy()



class KernelExplainer():

    # for now, use link='identity', later implement 'sum' to compress high-dimensional states
    def __init__(self, model_f, n_features, device):

        self.model_f = model_f
        self.n_features = n_features
        self.device = device

        self.reset()

        # average model prediction
        self.pred_null = self.model_f(torch.zeros(1, n_features, device=device)).squeeze() # shape [n_cells * horizon]
        self.pred_dim = self.pred_null.numel()

    def reset(self):

        self.n_samples_added = 0
        self.n_samples_run = 0
        self.ey = []
        self.n_fixed = []
        self.masks = []

    def shap_values(self, **kwargs):
        # background_mask is boolean tensor, with 1's for features that should be fixed,
        # and 0's for features that should be randomized
        use_regularization = kwargs.get('use_regularization', False)
        
        fidx = torch.arange(self.n_features, device=self.device)

        # get actual prediction
        feature_mask = torch.ones(self.n_features, device=self.device)
        self.pred_orig = self.model_f(feature_mask.unsqueeze(0)).squeeze()

        self.reset()

        if self.n_features == 0:
            # all features are fixed, so no feature has any effect
            phi = torch.zeros(self.n_features, self.pred_dim, device=self.device)
        elif self.n_features == 1:
            # all but one feature is fixed, so it has all the effect
            phi = torch.zeros(self.n_features, self.pred_dim, device=self.device)
            diff = self.pred_orig - self.pred_null
            phi[fidx[0], :] = diff
        else:
            # check all possible feature coalitions
            n_samples = kwargs.get('nsamples', 2 * self.n_features + 2**11)
            if self.n_features <= 30:
                # enumerate all subsets
                self.max_samples = 2**self.n_features - 2
                self.n_samples = min(self.max_samples, n_samples)
            else:
                self.max_samples = 2**30
                self.n_samples = n_samples

            max_subset_size = int(np.ceil((self.n_features - 1) / 2.0))
            n_paired_subset_sizes = int(np.floor((self.n_features - 1) / 2.0))

            self.kernel_weights = torch.zeros(self.n_samples, device=self.device)
            print(f'n_samples = {self.n_samples}')
            print(f'max_subset_size = {max_subset_size}')

            weights = torch.tensor([(self.n_features - 1) / (i * (self.n_features - i))
                                    for i in range(1, max_subset_size + 1)], device=self.device)
            weights[:n_paired_subset_sizes] *= 2
            weights /= weights.sum()

            remaining_weights = copy.copy(weights)

            n_full_subset_sizes = 0

            # fill out all subset sizes that can be enumerated completely given the available samples
            n_samples_left = self.n_samples
            for subset_size in range(1, max_subset_size + 1):
                print(f'processing subsets of size {subset_size}')
                coef = binom(self.n_features, subset_size)
                if subset_size <= n_paired_subset_sizes:
                    n_subsets = coef * 2
                else:
                    n_subsets = coef

                if n_samples_left * remaining_weights[subset_size - 1] / n_subsets >= 1.0 - 1e-8:
                    # enough samples left to enumerate all subsets of this size
                    n_full_subset_sizes += 1
                    n_samples_left -= n_subsets

                    if remaining_weights[subset_size - 1] < 1.0:
                        # normalize remaining weights
                        remaining_weights /= (1 - remaining_weights[subset_size - 1])

                    # add all samples of the current subset size
                    w = weights[subset_size - 1] / coef
                    if subset_size <= n_paired_subset_sizes:
                        w /= 2.0
                    for indices in torch.combinations(fidx, subset_size):
                        feature_mask[:] = 1
                        feature_mask[indices] = 0

                        self.add_sample(feature_mask, w)

                        if subset_size <= n_paired_subset_sizes:
                            # also sample complement
                            feature_mask = torch.logical_not(feature_mask)
                            self.add_sample(feature_mask, w)

                else:
                    # not enough samples left for subset size
                    break

            print(f'added {self.n_samples_added}. Start sampling random subsets')
            # add random samples from remaining subset space
            n_fixed_samples = self.n_samples_added
            n_samples_left = self.n_samples - self.n_samples_added
            print(f'full subsets = {n_full_subset_sizes}, max subset size = {max_subset_size}')
            if n_full_subset_sizes != max_subset_size and n_samples_left > 0:
                remaining_weights = copy.copy(weights)
                remaining_weights[:n_paired_subset_sizes] /= 2
                remaining_weights = remaining_weights[n_full_subset_sizes:]
                remaining_weights /= remaining_weights.sum()

                print(remaining_weights.sum(), remaining_weights)

                #random_sets = np.random.choice(len(remaining_weights), 4 * n_samples_left, p=remaining_weights)
                random_sets = torch.multinomial(remaining_weights, num_samples=4*n_samples_left, replacement=True)
                
                set_idx = 0
                used_masks = {}

                while n_samples_left > 0 and set_idx < len(random_sets):
                    feature_mask[:] = 0
                    #indices = random_sets[set_idx]
                    set_idx += 1
                    subset_size = random_sets[set_idx] + n_full_subset_sizes + 1
                    feature_mask[torch.randperm(self.n_features)[:subset_size]] = 1

                    # only add sample if we haven't seed it before, otherwise increase the weight of that sample
                    mask_tuple = tuple(feature_mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.n_samples_added
                        n_samples_left -= 1
                        self.add_sample(feature_mask, 1.0)
                    else:
                        self.kernel_weights[used_masks[mask_tuple]] += 1.0

                    # add compliment sample
                    if n_samples_left > 0 and subset_size <= n_paired_subset_sizes:
                        feature_mask[:] = 1
                        feature_mask[random_sets[set_idx]] = 0

                        if new_sample:
                            n_samples_left -= 1
                            self.add_sample(feature_mask, 1.0)
                        else:
                            self.kernel_weights[used_masks[mask_tuple] + 1] += 1.0

            # normalize weights for random samples to equal the weight left after fixed samples
            weight_left = weights[n_full_subset_sizes:].sum()
            self.kernel_weights[n_fixed_samples:] *= weight_left / self.kernel_weights[n_fixed_samples:].sum()


            # use all samples to find Shapley values
            self.ey = torch.stack(self.ey, dim=0) # [n_samples, pred_dim]
            self.masks = torch.stack(self.masks, dim=0) # [n_samples, n_features]

            # get shap value for each feature and pred_dim
            phi = torch.zeros(self.n_features, self.pred_dim, device=self.device)

            diff = self.ey - self.pred_null.unsqueeze(0) # [n_samples, pred_dim]
            n_fixed = torch.stack(self.n_fixed, dim=0)


            frac_sampled = self.n_samples / self.max_samples
            if use_regularization and frac_sampled < 0.2:
                print('Use regularization')
                aug_weights = torch.cat([self.kernel_weights * (self.n_features - n_fixed),
                                         self.kernel_weights * n_fixed], dim=0) # [n_samples * 2]
                sqrt_aug_weights = torch.sqrt(aug_weights)
                aug_diff_elim = torch.cat([diff,
                                           diff - (self.pred_orig - self.pred_null).unsqueeze(0)],
                                          dim=0) # [n_samples * 2, pred_dim]
                aug_diff_elim *= sqrt_aug_weights
                aug_mask = torch.cat([self.masks, self.masks - 1], dim=0) # [n_samples * 2, n_features]
                aug_mask = sqrt_aug_weights.unsqueeze(1) * aug_mask

                # select the top 10 features for each pred_dim
                reg_mask = torch.zeros(self.n_features, self.pred_dim, dtype=torch.bool, device=self.device)
                for dim in range(self.pred_dim):
                    reg_indices = (sklearn.linear_models.lars_path(aug_mask, aug_diff_elim[:, dim], max_iter=10)[1])
                    reg_mask[reg_indices, dim] = 1
                n_selected_features = 10
            else:
                reg_mask = torch.ones(self.n_features, self.pred_dim, dtype=torch.bool, device=self.device)
                n_selected_features = self.n_features
            # eliminate one variable with the constraint that all features sum to the output difference
            
            print(self.masks.size())
            mask = self.masks.unsqueeze(-1).repeat(1, 1, self.pred_dim)[:, reg_mask].reshape(self.n_samples, n_selected_features, self.pred_dim) # [n_samples, n_selected_features, pred_dim]
            print(mask.size())
            mask_elim = mask[:, -1, :] # [n_samples, pred_dim]
            mask_keep = mask[:, :-1, :] # [n_samples, n_selected_features - 1, pred_dim]
            print(diff.size(), mask_elim.size(), self.pred_orig.size(), self.pred_null.size())
            diff_elim = diff - (mask_elim * (self.pred_orig - self.pred_null).unsqueeze(0))  # [n_samples, pred_dim]
            etmp = mask_keep - mask_elim.unsqueeze(1)  # [n_samples, n_selected_features - 1, pred_dim]
            # use pred_dim as batch dimension
            X = etmp.permute(2, 0, 1) # [pred_dim, n_samples, n_selected_features - 1]
            print(diff_elim.size())
            Y = diff_elim.permute(1, 0).unsqueeze(-1) # [pred_dim, n_samples, 1]

            # compute weighted masks
            print(self.kernel_weights.size(), etmp.size(), Y.size())
            WX = self.kernel_weights.view(1, -1, 1) * X # [pred_dim, n_samples, n_selected_features - 1]

            # solve linear system Ax = B for all columns in B (i.e. all pred dims)
            A = X.transpose(1, 2) @ WX # [pred_dim, n_selected_features - 1, n_selected_features - 1]
            B = WX.transpose(1, 2) @ Y # [pred_dim, n_selected_features - 1, 1]

            try:
                w = torch.linalg.solve(A, B) # [pred_dim, n_selected_features - 1, 1]
            except Exception:
                warnings.warn('Linear regression is singular! Use least squares solutions instead.')
                sqrt_weights = torch.sqrt(self.kernel_weights).view(1, -1, 1)
                w = torch.linalg.lstsq(sqrt_weights * X, sqrt_weights * Y)

            w = w.squeeze(-1) # [pred_dim, n_selected_features - 1]
            
            # non-selected feature importance is 0
            w_elim = (self.pred_orig - self.pred_null) - w.sum(1) # [pred_dim, 1]
            w = torch.cat([w, w_elim.unsqueeze(-1)], dim=1) # [pred_dim, n_selected_features]
            phi[reg_mask] = w.T.reshape(-1)


        # clean up rounding errors:
        phi[phi.abs() < 1e-10] = 0


        return phi


    def add_sample(self, feature_mask, w):

        self.kernel_weights[self.n_samples_added] = w
        self.ey.append(self.model_f(feature_mask.unsqueeze(0)).squeeze())
        self.n_fixed.append(feature_mask.sum())
        self.masks.append(feature_mask)
        self.n_samples_added += 1
