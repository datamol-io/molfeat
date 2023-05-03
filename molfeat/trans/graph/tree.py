from typing import Iterable
from typing import Callable
from typing import Union
from typing import List
from typing import Optional

import os
import torch
import numpy as np
import datamol as dm
from loguru import logger
from molfeat.calc.tree import TreeDecomposer
from molfeat.trans.base import MoleculeTransformer
from molfeat.utils import datatype
from molfeat.utils.commons import one_hot_encoding
from molfeat.utils import requires

if requires.check("dgl"):
    import dgl


class MolTreeDecompositionTransformer(MoleculeTransformer):
    r"""
    Transforms a molecule into a tree structure whose nodes correspond to different functional groups.
    """

    def __init__(
        self,
        vocab: Optional[Iterable] = None,
        one_hot: bool = False,
        dtype: Optional[Callable] = None,
        cache: bool = True,
        **params,
    ):
        """MolTree featurizer

        Args:
            vocab: List of the smiles of the functional groups or clusters.
                If None, the transformer should be fiited before any usage.
            one_hot (bool, optional): Whether or not for a tree a 1d array or a 2d array is returned as features
                If 1d array, vocabulary elements are mapped into integers,
                otherwise, vocabulary elements  ar mapped into one-hot vectors
            cache: Whether to cache the tree decomposition to avoid recomputing for seen molecules
            dtype: Output data type. Defaults to None

        Attributes:
            vocab: Mapping from clusters to integers
            vocab_size: The number of clusters + 1
            one_hot: Whether or not for a sequence a 1d array or a 2d array is returned as features
        """

        self._save_input_args()

        super().__init__(
            dtype=dtype,
            one_hot=one_hot,
            cache=cache,
            featurizer=TreeDecomposer(cache=cache),
            **params,
        )
        if vocab is not None:
            self.vocab = vocab
            self._vocab_size = len(self.vocab) + 1
            self._fitted = True
        else:
            self.vocab = None
            self._vocab_size = None
            self._fitted = False

        if not requires.check("dgl"):
            raise ValueError("dgl is required for this featurizer, please install it first")

        if self.dtype is not None and not datatype.is_dtype_tensor(self.dtype):
            raise ValueError("DGL featurizer only supports torch tensors currently")

    @property
    def vocab_size(self):
        """Compute vocab size of this featurizer

        Returns:
            size: vocab size
        """
        return self._vocab_size

    def fit(
        self,
        X: List[Union[dm.Mol, str]],
        y: Optional[list] = None,
        output_file: Optional[os.PathLike] = None,
        **fit_params,
    ):
        """Fit the current transformer on given dataset.

        The goal of fitting is for example to identify nan columns values
        that needs to be removed from the dataset

        Args:
            X: input list of molecules
            y (list, optional): Optional list of molecular properties. Defaults to None.
            output_file: path to a file that will be used to store the generated set of fragments.
            fit_params: key val of additional fit parameters


        Returns:
            self: MolTransformer instance after fitting
        """
        if self.vocab is not None:
            logger.warning("The previous vocabulary of fragments will be erased.")
        self.vocab = self.featurizer.get_vocab(X, output_file=output_file, log=self.verbose)
        self._vocab_size = len(self.vocab) + 1
        self._fitted = True

        # save the vocab in the state
        self._input_args["vocab"] = self.vocab

        return self

    def _transform(self, mol: dm.Mol):
        r"""
        Compute features for a single molecule.
        This method would potentially need to be reimplemented by child classes

        Args:
            mol (dm.Mol): molecule to transform into features

        Returns
            feat: featurized input molecule

        """
        if not self._fitted:
            raise ValueError(
                "Need to call the fit function before any transformation. \
                Or provide the fragments vocabulary at the object construction"
            )

        try:
            _, edges, fragments = self.featurizer(mol)
            n_nodes = len(fragments)
            enc = [self.vocab.index(f) + 1 if f in self.vocab else 0 for f in fragments]
            enc = datatype.cast(enc, (self.dtype or torch.long))
            graph = dgl.graph(([], []))
            graph.add_nodes(n_nodes)
            for edge in edges:
                graph.add_edges(*edge)
                graph.add_edges(*edge[::-1])

            if self.one_hot:
                enc = [one_hot_encoding(f, self.vocab, encode_unknown=True) for f in fragments]
                enc = np.asarray(enc)
                enc = datatype.cast(enc, (self.dtype or torch.float))

            graph.ndata["hv"] = enc
        except Exception as e:
            raise e
            if self.verbose:
                logger.error(e)
            graph = None
        return graph
