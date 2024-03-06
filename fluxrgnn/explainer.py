from fluxrgnn import models, dataloader
import torch
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from scipy.special import binom
import shap
import numpy as np
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
        self.horizon = kwargs.get('horizon', 48)
        self.node_store = kwargs.get('node_store', 'cell')


    def convert_outputs(self, prediction):

        return prediction.detach().cpu().reshape(-1) #.numpy()


    def predict(self, input_graph):
        """
        Generates forecast for given features
        :param input_graph: SensorHeteroData object
        :return: prediction: np.ndarray of shape [n_samples, N]
        """

        prediction = self.forecast_model.forecast(input_graph, self.horizon, t0=self.t0)

        return self.convert_outputs(prediction)


    def mask_features(self, binary_mask, input_graph, sample_idx=0):
        """
        Replace masked features with corresponding background samples
        :param binary_mask: np.ndarray of shape [n_features] containing 0's for features that should be replaced
        :param input_graph: SensorHeteroData object containing original features
        :return: updated SensorHeteroData object
        """

        # background_mask = np.logical_not(binary_mask)
        # out = np.tile(features, (self.n_samples, 1, 1))
        # out[:, background_mask, :] = self.background[:, background_mask, :]

        # apply changes to copies of original input data
        input_batch = Batch.from_data_list([copy.copy(input_graph) for _ in range(self.n_samples)])

        # indices = np.where(np.logical_not(binary_mask))[0]
        indices = torch.logical_not(binary_mask).nonzero()

        for fidx in indices:
            name = self.feature_names[fidx]
            # fbg = self.background[sample_idx, fidx] # [n_cells, T]
            fbg = self.background[:, fidx].reshape(self.n_samples * self.n_cells, self.horizon)

            input_batch[self.node_store][name] = torch.tensor(fbg, dtype=input_batch[self.node_store][name].dtype,
                                                       device=input_batch[self.node_store][name].device)

            # data[self.node_store][name] = torch.tensor(fbg, dtype=data[self.node_store][name].dtype,
            #                                                 device=data[self.node_store][name].device)

            #print(f'original {name}: {input_graph[self.node_store][name]}')
            #print(f'masked {name}: {data[self.node_store][name]}')

            #assert not torch.allclose(data[self.node_store][name], input_graph[self.node_store][name])

        return input_batch


    def explain(self, input_graph, n_samples=1000):
        """
        Compute SHAP values for given instance

        :param input_graph: SensorHeteroData
        :return: SHAP values
        """

        assert isinstance(input_graph, dataloader.SensorHeteroData)

        # make sure model and data are on the same device
        input_graph = input_graph.to(self.forecast_model.device)

        def f(binary_masks):
            # maps feature masks to model outputs and averages over background samples
            n_masks = binary_masks.shape[0]
            # out = np.zeros((n_masks, self.n_cells * self.horizon))
            out = torch.zeros((n_masks, self.n_cells * self.horizon))
            for i in range(n_masks):
                print(f'retain {binary_masks[i].sum()} features')
                # TODO: push all samples through model in parallel
                masked_input = self.mask_features(binary_masks[i], input_graph)
                pred = self.predict(masked_input) # [n_samples * n_cells * horizon]
                out[i] = pred.reshape(self.n_samples, -1).mean()
                # out[i] = unbatch(pred, masked_input[self.node_store].batch, dim=0) # [n_samples, n_cells * horizon]
                # for j in range(self.n_samples):
                #     masked_input = self.mask_features(binary_masks[i], input_graph, sample_idx=j)
                #     pred = self.predict(masked_input)
                #     out[i, j] = pred
                # TODO: get batch for each node and reshape to size [samples, cells * time] (use ptg.utils.unbatch)

            #out /= self.n_samples
            # print(f'std over predictions = {out.std(1)}')
            # out = out.mean(1) # take sample mean

            return out # shape [n_masks, N]


        # setup explainer (compute expectation over background)
        # mask_all = np.zeros((1, self.n_features)) # mask for overall background
        # explainer = shap.KernelExplainer(f, mask_all)

        explainer = KernelExplainer(f, self.n_features, self.forecast_model.device)

        # compute SHAP values
        # mask_none = np.ones((1, self.n_features)) # mask for actual prediction
        # shap_values = explainer.shap_values(mask_none, nsamples=n_samples)
        shap_values = explainer.shap_values(nsamples=n_samples)


        return {'shap_values': shap_values,
                # 'expected_values': explainer.expected_value,
                'expected_values': explainer.pred_null,
                'feature_names': self.feature_names}


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
        self.pred_null = self.model_f(torch.zeros(n_features, device=device)) # shape [n_cells * horizon]
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
        fidx = torch.arange(self.n_features)

        # get actual prediction
        feature_mask = torch.ones(self.n_features, device=self.device)
        self.pred_orig = self.model_f(feature_mask)

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
            n_samples = kwargs.get('n_samples', 2 * self.n_features + 2**11)
            if self.n_features <= 30:
                # enumerate all subsets
                self.max_samples = 2**self.n_features - 2
                self.n_samples = min(self.max_samples, n_samples)
            else:
                self.max_samples = 2**30
                self.n_samples = n_samples

            max_subset_size = int(np.ceil((self.n_features - 1) / 2.0))
            n_paired_subset_sizes = int(np.floor((self.n_features - 1) / 2.0))

            self.kernel_weights = torch.zeros(self.n_samples)

            weights = torch.tensor([(self.n_features - 1) / (i * (self.n_features - i))
                                    for i in range(1, max_subset_size + 1)])
            weights[:n_paired_subset_sizes] *= 2
            weights /= weights.sum()

            remaining_weights = copy.copy(weights)

            n_full_subsets = 0

            # fill out all subset sizes that can be enumerated completely given the available samples
            n_samples_left = self.n_samples
            for subset_size in range(1, max_subset_size + 1):
                coef = binom(self.n_features, subset_size)
                if subset_size <= n_paired_subset_sizes:
                    n_subsets = coef * 2
                else:
                    n_subsets = coef

                if n_samples_left * remaining_weights[subset_size - 1] / n_subsets >= 1.0 - 1e-8:
                    # enough samples left to enumerate all subsets of this size
                    n_full_subsets += 1
                    n_samples_left -= n_subsets

                    if remaining_weights[subset_size - 1] < 1.0:
                        # normalize remaining weights
                        remaining_weights /= (1 - remaining_weights[subset_size - 1])

                    # add all samples of the current subset size
                    w = weights[subset_size - 1] / coef
                    if subset_size <= n_paired_subset_sizes:
                        w /= 2.0
                    for indices in itertools.combinations(fidx, subset_size):
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


            # add random samples from remaining subset space
            n_fixed_samples = self.n_samples_added
            n_samples_left = self.n_samples - self.n_samples_added
            if n_full_subsets != n_subsets:
                remaining_weights = copy.copy(weights)
                remaining_weights[:n_paired_subset_sizes] /= 2
                remaining_weights = remaining_weights[n_full_subsets:]
                remaining_weights /= remaining_weights.sum()

                random_sets = np.random.choice(len(remaining_weights), 4 * n_samples_left, p=remaining_weights)
                set_idx = 0
                used_masks = {}

                while n_samples_left > 0 and set_idx < len(random_sets):
                    feature_mask[:] = 0
                    indices = random_sets[set_idx]
                    set_idx += 1
                    subset_size = indices + n_full_subsets + 1
                    feature_mask[np.random.permutation(self.n_features)[:subset_size]] = 1

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
                        feature_mask[indices] = 0

                        if new_sample:
                            n_samples_left -= 1
                            self.add_sample(feature_mask, 1.0)
                        else:
                            self.kernel_weights[used_masks[mask_tuple] + 1] += 1.0

            # normalize weights for random samples to equal the weight left after fixed samples
            weight_left = weights[n_full_subsets:].sum()
            self.kernel_weights[n_fixed_samples:] *= weight_left / self.kernel_weights[n_fixed_samples:].sum()


            # use all samples to find Shapley values
            self.ey = torch.stack(self.ey, dim=0) # [n_samples, pred_dim]
            self.masks = torch.stack(self.masks, dim=0) # [n_samples, n_features]

            # get shap value for each feature and pred_dim
            phi = torch.zeros(self.n_features, self.pred_dim)

            diff = torch.stack(self.ey, dim=0) - self.pred_null.unsqueeze(0) # [n_samples, pred_dim]
            n_fixed = torch.stack(self.n_fixed, dim=0)


            frac_sampled = self.n_samples / self.max_samples
            if self.use_regularization and frac_sampled < 0.2:
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
            else:
                reg_mask = torch.ones(self.n_features, self.pred_dim, dtype=torch.bool, device=self.device)

            # eliminate one variable with the constraint that all features sum to the output difference
            mask = self.masks.unsqueeze(-1).repeat(self.pred_dim)[:, reg_mask] # [n_samples, n_selected_features, pred_dim]
            mask_elim = mask[:, -1, :] # [n_samples, pred_dim]
            mask_keep = mask[:, :-1, :] # [n_samples, n_selected_features - 1, pred_dim]
            diff_elim = diff - (mask_elim * (self.pred_orig - self.pred_null).unsqueeze(0))  # [n_samples, pred_dim]
            etmp = mask_keep - mask_elim.unsqueeze(1)  # [n_samples, n_selected_features - 1]

            # compute weighted masks
            WX = self.kernel_weights.unsqueeze(1) * etmp # [n_samples, n_selected_features - 1]

            # solve linear system Ax = B for all columns in B (i.e. all pred dims)
            A = etmp.T @ WX # [n_selected_features - 1, n_selected_features - 1]
            B = WX.T @ diff_elim # [n_selected_features - 1, pred_dim]

            try:
                w = torch.linalg.solve(A, B) # [n_selected_features - 1, pred_dim]
            except Exception:
                warnings.warn('Linear regression is singular! Use least squares solutions instead.')
                sqrt_weights = torch.sqrt(self.kernel_weights).unsqueeze(1)
                w = torch.linalg.lstsq(sqrt_weights * etmp, sqrt_weights * diff_elim)

            # non-selected feature importance is 0
            w_elim = (self.pred_orig - self.pred_null - w.sum(0)).unsqueeze(0)
            w = torch.stack([w, w_elim], dim=0) # [n_selected_features, pred_dim
            phi[reg_mask] = w


        # clean up rounding errors:
        phi[phi.abs() < 1e-10] = 0


        return phi


    def add_sample(self, feature_mask, w):

        self.n_samples_added += 1
        self.kernel_weights[self.n_samples_added] = w
        self.ey.append(self.model_f(feature_mask))
        self.n_fixed.append(feature_mask.sum())
        self.masks.append(feature_mask)