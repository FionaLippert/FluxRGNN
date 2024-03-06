from fluxrgnn import models, dataloader
import torch
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import shap
import numpy as np
import copy



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

        return prediction.detach().cpu().numpy().flatten()


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

        print(input_batch[self.node_store].batch.reshape(self.n_samples, self.n_cells))

        indices = np.where(np.logical_not(binary_mask))[0]

        for fidx in indices:
            name = self.feature_names[fidx]
            # fbg = self.background[sample_idx, fidx] # [n_cells, T]
            fbg = self.background[:, fidx].reshape(self.n_samples * self.n_cells, self.T)

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
            # out = np.zeros((n_masks, self.n_samples, self.n_cells * self.horizon))
            out = np.zeros((n_masks, self.n_cells * self.horizon))
            for i in range(n_masks):
                print(f'retain {binary_masks[i].sum()} features')
                # TODO: push all samples through model in parallel
                masked_input = self.mask_features(binary_masks[i], input_graph)
                pred = self.predict(masked_input) # [n_samples * n_cells * horizon]
                out[i] = pred.reshape(self.n_samples, -1).mean(0)
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
        mask_all = np.zeros((1, self.n_features)) # mask for overall background
        explainer = shap.KernelExplainer(f, mask_all)

        # compute SHAP values
        mask_none = np.ones((1, self.n_features)) # mask for actual prediction
        shap_values = explainer.shap_values(mask_none, nsamples=n_samples)
        shap_values = np.stack(shap_values, axis=0)

        return {'shap_values': shap_values, 
                'expected_values': explainer.expected_value,
                'actual_values': explainer.fx,
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

    return background.cpu().numpy()



