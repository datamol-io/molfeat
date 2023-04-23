from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import copy
from molfeat.utils.commons import _parse_to_evaluable_str
from molfeat.utils import requires
from molfeat.utils.pooler import Pooling
from molfeat.trans.pretrained.base import PretrainedMolTransformer

if requires.check("graphormer_pretrained"):
    from graphormer_pretrained.embeddings import GraphormerEmbeddingsExtractor
    from graphormer_pretrained.data.smiles_dataset import GraphormerInferenceDataset


class GraphormerTransformer(PretrainedMolTransformer):
    r"""
    Graphormer transformer from microsoft, pretrained on PCQM4Mv2 quantum chemistry dataset
    for the prediction of homo-lumo gap.

    Attributes:
        featurizer: Graphormer embedding object
        dtype: Data type. Use call instead
        pooling: Pooling method for Graphormer's embedding layer
    """

    def __init__(
        self,
        kind: str = "pcqm4mv2_graphormer_base",
        dtype: Callable = np.float32,
        pooling: str = "mean",
        max_length: Optional[int] = None,
        concat_layers: Union[List[int], int] = -1,
        ignore_padding: bool = True,
        version=None,
        **params,
    ):
        """
        Pretrained graphormer featurizer.

        !!! note
            The default behaviour of this feature extractor is to return the last hidden state of the encoder,
            averaged across all nodes (including the virtual node connected to all other nodes).

            For a different behaviour, please change the pooling method:
            * `graph` or `virtual`: use the virtual node embedding in the last layer to get the graph representation
            * `mean`, `max`, `sum`, etc or any other supported pooling of `molfeat.utils.pooler.Pooling`
                will take the operation defined by the pooling layer across all nodes of each graph

        Args:
            kind: name of the featurizer as available in the model store
            dtype: Data type to output
            pooling: type of pooling to use. One of ['graph', 'virtual', 'mean', 'max', 'sum']. The value "graph" corresponds to the virtual node representation
            max_length: Maximum length of the input sequence to consider. Please update this for large sequences
            concat_layers: Layer to concat to get the representation. By default the last hidden layer is returned.
            ignore_padding: Whether to ignore padding in the representation (default: True) to avoid effect of batching
            params: any other parameter to pass to PretrainedMolTransformer
        """

        super().__init__(dtype=dtype, pooling=pooling, **params)
        if not requires.check("graphormer_pretrained"):
            raise ValueError("`graphormer` is required to use this featurizer.")

        if concat_layers is None:
            concat_layers = -1
        if not isinstance(concat_layers, list):
            concat_layers = [concat_layers]
        self.concat_layers = concat_layers
        self.preload = True
        self.name = kind
        self._require_mols = False
        self.max_length = max_length
        self.ignore_padding = ignore_padding
        if isinstance(pooling, str):
            if pooling in Pooling.SUPPORTED_POOLING:
                pooling = Pooling(dim=1, name=pooling)
            else:
                pooling = None
        self.pooling = pooling
        self.featurizer = GraphormerEmbeddingsExtractor(
            pretrained_name=self.name, max_nodes=self.max_length, concat_layers=self.concat_layers
        )
        self.featurizer.config.max_nodes = self.max_length
        self.version = version

    def __repr__(self):
        return "{}(name={}, pooling={}, dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self.name),
            _parse_to_evaluable_str(self.pooling.name),
            _parse_to_evaluable_str(self.dtype),
        )

    @staticmethod
    def list_available_models():
        """List available graphormer model to use"""
        return [
            "pcqm4mv1_graphormer_base",  # PCQM4Mv1
            "pcqm4mv2_graphormer_base",  # PCQM4Mv2
            "pcqm4mv1_graphormer_base_for_molhiv",  # ogbg-molhiv
            "oc20is2re_graphormer3d_base",  # Open Catalyst Challenge
        ]

    def _embed(self, inputs: List[str], **kwargs):
        """Internal molecular embedding

        Args:
            smiles: input smiles
        """
        with torch.no_grad():
            layer_reprs, graph_reprs, padding_mask = self.featurizer(inputs)
            if self.pooling is None:
                x = graph_reprs
            else:
                x = self.pooling(layer_reprs, mask=(padding_mask if self.ignore_padding else None))
        return x.numpy()

    def __getstate__(self):
        """Getting state to allow pickling"""
        d = copy.deepcopy(self.__dict__)
        d["precompute_cache"] = None
        d.pop("featurizer", None)
        return d

    def __setstate__(self, d):
        """Setting state during reloading pickling"""
        self.__dict__.update(d)
        self._update_params()

    def compute_max_length(self, inputs: list):
        """Compute maximum node number for the input list of molecules

        Args:
            inputs: input list of molecules
        """
        dataset = GraphormerInferenceDataset(
            inputs,
            multi_hop_max_dist=self.featurizer.config.multi_hop_max_dist,
            spatial_pos_max=self.featurizer.config.spatial_pos_max,
        )
        xs = [item.x.size(0) for item in dataset]
        return max(xs)

    def set_max_length(self, max_length: int):
        """Set the maximum length for this featurizer"""
        self.max_length = max_length
        self._update_params()
        self._preload()

    def _convert(self, inputs: list, **kwargs):
        """Convert molecules to the right format

        Args:
            inputs: inputs to preprocess

        Returns:
            processed: pre-processed input list
        """
        inputs = super()._convert(inputs, **kwargs)
        batch = self.featurizer._convert(inputs)
        return batch

    def _update_params(self):
        super()._update_params()
        self.featurizer = GraphormerEmbeddingsExtractor(
            pretrained_name=self.name, max_nodes=self.max_length
        )
        self.featurizer.config.max_nodes = self.max_length
