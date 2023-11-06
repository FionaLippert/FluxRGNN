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
                    setattr(store, self.feature, torch.log(values + self.offset))

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(feature={self.feature}, offset={self.offset})'


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
                offset = torch.full((num_nodes, 1), self.offset, dtype=torch.float)

                if hasattr(store, self.feature):
                    values = getattr(store, self.feature)
                    setattr(store, self.feature, self.factor * (values + self.offset))

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(feature={self.feature}, offset={self.offset}, factor={self.factor})'
