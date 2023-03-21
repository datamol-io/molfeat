from typing import List, Union
from typing import Callable
from typing import Optional

import numpy as np
import torch

from molfeat.utils import datatype
from molfeat.utils.pooler import Pooling
from molfeat.trans.base import MoleculeTransformer
from molfeat.utils.commons import _parse_to_evaluable_str
from molfeat.utils import requires

if requires.check("bio_embeddings"):
    from bio_embeddings import embed as bio_embedder


class ProtBioFingerprint(MoleculeTransformer):
    """
    Wrapper for general purpose biological sequence representations, as provided by [`bio_embeddings`](https://github.com/sacdallago/bio_embeddings)

    For a list of available embeddings, see: https://docs.bioembeddings.com/v0.2.2/api/bio_embeddings.embed.html

    !!! note:
        The embeddings proposed here are the general purpose embeddings, meaning that task-specific
        embeddings offered by `bio_embeddings` (e.g PBTucker, DeepBlast) are not included.

        According to the bio_embeddings documentation, `prottrans_bert_bfd` and `seqvec` are the best embeddings.
    """

    SUPPORTED_EMBEDDINGS = [
        "bepler",
        "cpcprot",
        "esm",
        "esm1b",
        "esm1v",
        "fasttext",
        "glove",
        "one_hot_encoding",
        "plus_rnn",
        "prottrans_albert_bfd",
        "prottrans_bert_bfd",
        "prottrans_t5_bfd",
        "prottrans_t5_uniref50",
        "prottrans_t5_xl_u50",
        "prottrans_xlnet_uniref100",
        "seqvec",
        "unirep",
        "word2vec",
    ]

    def __init__(
        self,
        featurizer: Union[str, Callable] = "seqvec",
        pooling: str = "mean",
        dtype: Callable = np.float32,
        device: Optional[Union[torch.device, str]] = None,
        layer_pooling: str = "sum",
        **kwargs,
    ):
        """Constructor for Deep Learning based Protein representation.
        SeqVec featurizer will e

        Args:
            featurizer: Name of callable of the embedding model
            pooling: Pooling method to use for sequence embedding. Defaults to "mean".
                If you set pooling to None, token representation will be returned
            dtype: Representation output datatype. Defaults to None.
            device: Torch device to move the model to. Defaults to None.
            layer_pooling: Layer-wise pooling method to use when > 1 layer exists. Default to 'sum'.
                If None, last layers is taken. This is relevant for `seqvec` mostly
        """
        if not requires.check("bio_embeddings"):
            raise ValueError(
                "Cannot use this featurizer without bio_embeddings (pip install 'bio_embeddings[all]')."
            )

        if isinstance(featurizer, bio_embedder.EmbedderInterface):
            featurizer = featurizer
            self._model_name = self.featurizer.name
        else:
            if (
                not isinstance(featurizer, str)
                or featurizer.lower() not in self.SUPPORTED_EMBEDDINGS
            ):
                raise ValueError("Unknown featurizer: {}".format(featurizer))
            self._model_name = featurizer.lower()
            featurizer = bio_embedder.name_to_embedder[self._model_name](device=device, **kwargs)

        super().__init__(featurizer=featurizer, dtype=dtype, **kwargs)
        self._fitted = True
        self._representation = "seq"
        self.pooling = Pooling(dim=0, name=pooling)
        self.layer_pooling = Pooling(dim=0, name=layer_pooling)
        if pooling is None:
            self._representation = "token"
        self._feat_length = None

    def __len__(self):
        """Get featurizer length"""
        return self.featurizer.embedding_dimension

    @property
    def n_layers(self):
        """Get the number of layers used in this embedding"""
        return self.featurizer.number_of_layers

    def __repr__(self):
        return "{}(model={}, pooling={}, dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self._model_name),
            _parse_to_evaluable_str(self.pooling.name),
            _parse_to_evaluable_str(self.dtype),
        )

    def _pool(self, embedding: list):
        """Perform embedding pooling
        Args:
            embedding: input embedding
        """
        if self.n_layers > 1 and self.layer_pooling.name is not None:
            embedding = self.layer_pooling(embedding)
        if len(embedding.shape) > 2:
            # we forcefully take the last layers
            embedding = embedding[-1]
        return self.pooling(embedding)

    def _transform(
        self,
        protein_seq: str,
        **kwargs,
    ):
        """
        Transform a protein/nucleotide sequence into a feature vector.

        Args:
            protein: protein sequence as amino acid sequences

        Returns:
            Embedding of size (FEAT_DIM, N_LAYERS) for token embeddings
                and (FEAT_DIM, N_LAYERS) for sequence embeddings
        """

        rep = self.featurizer.embed(protein_seq)
        return self._pool(rep)

    def transform(self, seqs: List[str], names: Optional[List[str]] = None, **kwargs):
        """
        Transform a list of protein/nucleotide sequence into a feature vector.

        Args:
            seqs: list of protein/nucleotide sequence as amino acids
            names: names of the macromolecules.  Will be ignored
            kwargs: additional arguments for the featurizer

        Returns:
            Embedding of size (N_SEQS, FEAT_DIM) for token embeddings
                and (FEAT_DIM, N_LAYERS) for sequence embeddings
        """
        if not isinstance(seqs, list):
            seqs = [seqs]
        if isinstance(seqs[0], (list, tuple)) and len(seqs[0]) == 2:
            _, seqs = zip(*seqs)
            seqs = list(seqs)
        res = list(self.featurizer.embed_many(seqs, **kwargs))
        res = [self._pool(x) for x in res]
        return res

    def __call__(
        self,
        seqs: List[str],
        ignore_errors: bool = False,
        enforce_dtype: bool = True,
        **kwargs,
    ):
        r"""
        Compute molecular representation of a protein sequence.
        If ignore_error is True, a list of features and valid ids are returned.

        Args:
            seqs: list of protein or nucleotide sequence as amino acids
            enforce_dtype: whether to enforce the instance dtype in the generated fingerprint
            ignore_errors: Whether to ignore errors during featurization or raise an error.
            kwargs: Named parameters for the transform method

        Returns:
            feats: list of valid embeddings
            ids: all valid positions that did not failed during featurization.
                Only returned when ignore_errors is True.

        """
        features = self.transform(seqs, **kwargs)
        ids = np.arange(len(features))
        if ignore_errors:
            features, ids = self._filter_none(features)
        if self.dtype is not None and enforce_dtype:
            if self._representation.startswith("token"):
                features = [
                    datatype.cast(feat, dtype=self.dtype, columns=self.columns) for feat in features
                ]
            else:
                features = datatype.cast(features, dtype=self.dtype, columns=self.columns)
        if not ignore_errors:
            return features
        return features, ids
