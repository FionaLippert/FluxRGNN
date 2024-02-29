from typing import List, Optional, Union
import copy
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('log_transform')
class LogTransform(BaseTransform):
    r"""Applies a log-transformation to each node feature :obs:`feature`.

    Args:
        feature (str, optional): feature to rescale (default: :obs:`x`)
        offset (float, optional): Offset added to avoid undefined behavior. (default: :obj:`1e-5`)
        node_types (str or List[str], optional): The specified node type(s) to
            apply to tranformation to if used on heterogeneous graphs.
            If set to :obj:`None`, the log-transformation will be applied to each node feature
            :obj:`feature` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        feature: str = 'x',
        offset: float = 1e-5,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.feature = feature
        self.offset = offset
        self.node_types = node_types

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        return self.forward(copy.copy(data))

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                # offset = torch.full((num_nodes, 1), self.offset, dtype=torch.float)

                if hasattr(store, self.feature):
                    values = getattr(store, self.feature)
                    setattr(store, self.feature, self.tensor_forward(values))

        return data

    def tensor_forward(self, x: torch.Tensor):

        return torch.log(x + self.offset)

    def tensor_backward(self, x: torch.Tensor):

        return torch.exp(x) - self.offset

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(feature={self.feature}, offset={self.offset})'

@functional_transform('pow_transform')
class PowerTransform(BaseTransform):
    r"""Applies a power-transformation to each node feature :obs:`feature`.

    Args:
        feature (str, optional): feature to rescale (default: :obs:`x`)
        exponent (float, optional): exponent used in power transform. (default: :obj:`1/3`)
        node_types (str or List[str], optional): The specified node type(s) to
            apply to tranformation to if used on heterogeneous graphs.
            If set to :obj:`None`, the log-transformation will be applied to each node feature
            :obj:`feature` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        feature: str = 'x',
        exponent: float = 1/3,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.feature = feature
        self.exponent = exponent
        self.node_types = node_types

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        return self.forward(copy.copy(data))

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:

                if hasattr(store, self.feature):
                    values = getattr(store, self.feature)
                    setattr(store, self.feature, self.tensor_forward(values))

        return data

    def tensor_forward(self, x: torch.Tensor):

        return torch.pow(x, self.exponent)

    def tensor_backward(self, x: torch.Tensor):

        return torch.pow(x, 1/self.exponent)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(feature={self.feature}, exponent={self.exponent})'



@functional_transform('rescale')
class Rescaling(BaseTransform):
    r"""Rescales node feature :obs:`feature` by adding :obs:`offset` and mutliplying with :obs:`factor`.

    Args:
        feature (str, optional): feature to rescale (default: :obs:`x`)
        offset (float, optional): Offset added to node feature (default: :obj:`0.0`)
        factor (float, optional): Factor by which the shifted node features are multiplied (default: :obs:`1.0`)
        node_types (str or List[str], optional): The specified node type(s) to
            apply to tranformation to if used on heterogeneous graphs.
            If set to :obj:`None`, the rescaling will be applied to each node feature
            :obj:`feature` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        feature: str = 'x',
        offset: float = 0.0,
        factor: float = 1.0,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.feature = feature
        self.offset = offset
        self.factor = factor
        self.node_types = node_types

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:

        return self.forward(copy.copy(data))

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                if num_nodes is None:
                    setattr(store, 'num_nodes', store.coords.size(0))

                if hasattr(store, self.feature):
                    values = getattr(store, self.feature)
                    setattr(store, self.feature, self.tensor_forward(values))

        return data

    def tensor_forward(self, x: torch.Tensor):

        return self.factor * (x + self.offset)

    def tensor_backward(self, x: torch.Tensor):

        return x / self.factor - self.offset

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(feature={self.feature}, offset={self.offset}, factor={self.factor})'
