from typing import Callable
from typing import List
from typing import Optional

import numpy as np
import torch
import copy
from molfeat.utils.commons import _parse_to_evaluable_str
from molfeat.utils import requires
from molfeat.utils.pooler import Pooling
from molfeat.trans.pretrained.base import PretrainedMolTransformer

if requires.check("graphormer"):
    from graphormer.embeddings import GraphormerEmbeddingsExtractor
    from graphormer.data.smiles_dataset import GraphormerInferenceDataset


class GraphormerTransformer(PretrainedMolTransformer):
    r"""
    Graphormer transformer based on pretrained sequence embedder

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
        version=None,
        **params,
    ):
        super().__init__(dtype=dtype, pooling=pooling, **params)
        if not requires.check("graphormer"):
            raise ValueError("`graphormer` is required to use this featurizer.")

        self.preload = True
        self.name = kind
        self._require_mols = False
        self.max_length = max_length
        self.pooling = pooling
        if isinstance(pooling, str):
            pooling = Pooling(dim=1, name=pooling)
        self.pooling = pooling
        self.featurizer = GraphormerEmbeddingsExtractor(
            pretrained_name=self.name, max_nodes=self.max_length
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
            smiles: input smiless
        """
        with torch.no_grad():
            x = self.featurizer.model(inputs)
            x = self.pooling(x)
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
