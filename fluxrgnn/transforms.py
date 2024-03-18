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


class Transforms:

    def __init__(self, transforms: list, **kwargs):

        self.transforms = {}

        for t in transforms:
            var = t.feature

            if var not in self.transforms:
                self.transforms[var] = []

            self.transforms[var].append(t)

        self.zero_value = {var: self.apply_forward_transforms(torch.tensor(0), var) for var in self.transforms}

        self.log_offset = kwargs.get('log_offset', 1e-8)
        self.pow_exponent = kwargs.get('pow_exponent', 0.3333)

    def apply_forward_transforms(self, values: torch.Tensor, var: str = 'x'):

        out = values
        if var in self.transforms:
            for t in self.transforms[var]:
                out = t.tensor_forward(out)

        return out

    def apply_backward_transforms(self, values: torch.Tensor, var: str = 'x'):

        out = values
        if var in self.transforms:
            for t in reversed(self.transforms[var]):
                out = t.tensor_backward(out)

        return out

    def transformed2raw(self, values: torch.Tensor, var: str = 'x'):

        out = values
        if var in self.transforms:
            out = torch.clamp(out, min=self.zero_value[var].to(values.device))
            out = self.apply_backward_transforms(out, var)

        return out

    # def to_log(self, values):
    def raw2log(self, values: torch.Tensor):

        log = torch.clamp(values, min=0)
        log = torch.log(log + self.log_offset)

        return log

    def raw2pow(self, values):

        pow = torch.pow(values, 1 / 3)

        return pow

