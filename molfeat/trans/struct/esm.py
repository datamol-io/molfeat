from typing import List
from typing import Callable
from typing import Optional

import re
import numpy as np
import torch

from molfeat.utils import datatype
from molfeat.utils.pooler import Pooling
from molfeat.trans.base import MoleculeTransformer
from molfeat.utils.commons import _parse_to_evaluable_str


class ESMProteinFingerprint(MoleculeTransformer):
    """
    ESM (Evolutionary Scale Modeling) protein representation embedding.
    ESM is a transformer protein language model introduced by Facebook FAIR in Rives et al., 2019:
    'Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences'
    """

    def __init__(
        self,
        featurizer: str = "esm1b_t33_650M_UR50S",
        loader_repo_or_dir: str = "facebookresearch/esm:main",
        device: Optional[str] = None,
        layers: List[int] = None,
        pooling: str = "mean",
        dtype: Callable = None,
        contact: bool = False,
        **kwargs,
    ):
        """Constructor for ESM protein representation

        Args:
            featurizer: Name of the ESM model to use. Defaults to "esm1b_t33_650M_UR50S".
            loader_repo_or_dir: Path to local dir containing the model or to a github repo. Default to "facebookresearch/esm:main
            device: Torch device to move the model to. Defaults to None.
            layers: Layers to use to extract information. Defaults to None, which is the last layers.
            pooling: Pooling method to use for sequence embedding. Defaults to "mean".
                If you set pooling to None, token representation will be returned (excluding BOS)
            dtype: Representation output datatype. Defaults to None.
            contact: Whether to return the predictied attention contact instead of the representation. Defaults to False.
        """
        self._model_name = featurizer
        self.device = device
        self.dtype = dtype
        self.featurizer = None
        self.alphabet = None
        self.batch_converter = None
        self._fitted = True
        self.cols_to_keep = None
        self.repr_layers = layers
        self.repo_or_dir = loader_repo_or_dir
        self.contact = contact
        max_layer_pattern = re.compile(".*_t([0-9]+)_.*")
        self._max_layers = int(max_layer_pattern.match(featurizer).group(1))
        if layers is None:
            self.repr_layers = [self._max_layers]
        if any(l > self._max_layers for l in self.repr_layers):
            raise ValueError(
                "You are requesting more layers than available for this pretrained model"
            )
        self._representation = "seq"
        self.pooling = Pooling(dim=0, name=pooling)
        if pooling is None:
            self._representation = "token"
        self._feat_length = None
        self._load_model()

    def _load_model(self):
        """Load model internally"""
        self.featurizer, self.alphabet = torch.hub.load(self.repo_or_dir, self._model_name)  # type: ignore
        self.batch_converter = self.alphabet.get_batch_converter()
        if self.device is not None:
            self.featurizer = self.featurizer.to(self.device)
        self.featurizer.eval()

    def __len__(self):
        """Get featurizer length"""
        if self._feat_length is None and not self.contact:
            embds = self._transform("MMMM")
            self._feat_length = embds.shape[-1]
        return self._feat_length

    @property
    def n_layers(self):
        """Number of layers used in the current embeddings"""
        return len(self.repr_layers)

    @torch.no_grad()
    def _embed(self, prot_seqs: List[str], prot_names: Optional[List[str]] = None, **kwargs):
        r"""
        Compute features for a single molecule.
        This method would potentially need to be reimplemented by child classes

        Args:
           prot_seqs: protein sequences as a sequence of amino acids
           prot_names: protein names

        Returns
            feat: list of N_SEQ representation, each of size (SEQ_LEN, FEAT_DIM * N_LAYERS) for token embeddings
                and (FEAT_DIM * N_LAYERS) for sequence embeddings. Note that SEQ_LEN will include the stop token.

        """
        if isinstance(prot_seqs, str):
            prot_seqs = [prot_seqs]
        if prot_names is None:
            prot_names = ["protein_{i}" for i in range(len(prot_seqs))]
        if isinstance(prot_names, str):
            prot_names = [prot_names]
        if len(prot_seqs) != len(prot_names):
            raise ValueError("Must provide the same number of protein sequence and label")
        data = list(zip(prot_names, prot_seqs))
        *_, batch_tokens = self.batch_converter(data)
        if self.device is not None:
            batch_tokens = batch_tokens.to(self.device)

        results = self.featurizer(
            batch_tokens, repr_layers=self.repr_layers, return_contacts=self.contact
        )
        embeddings = []
        if self.contact:
            for _, (seq, att_concats) in enumerate(zip(prot_seqs, results["contacts"])):
                embeddings.append(att_concats[: len(seq), : len(seq)])
        else:
            representation = torch.stack(
                [results["representations"][x] for x in self.repr_layers], dim=-1
            )
            if self._representation.startswith("seq"):
                for seq, token_rep in zip(prot_seqs, representation):
                    embeddings.append(
                        self.pooling(token_rep[1 : len(seq) + 1]).view(1, -1).squeeze(0)
                    )
            else:
                embeddings = list(
                    representation.view(representation.shape[0], representation.shape[1], -1)
                )
        return embeddings

    def __repr__(self):
        return "{}(model={}, pooling={}, dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self._model_name),
            _parse_to_evaluable_str(self.pooling.name),
            _parse_to_evaluable_str(self.dtype),
        )

    def _transform(self, protein_seq: str, protein_name: str = None):
        """
        Transform a protein sequence into a feature vector.

        Args:
            protein: protein sequence as amino acid sequences
            protein_name: protein name

        Returns:
            Embedding of size (SEQ_LEN, FEAT_DIM, N_LAYERS) for token embeddings
                and (FEAT_DIM * N_LAYERS) for sequence embeddings
        """
        return self._embed(protein_seq, protein_name)[0]

    def transform(self, seqs: List[str], names: Optional[List[str]] = None, **kwargs):
        """
        Transform a list of protein sequence into a feature vector.

        Args:
            seqs: list of protein sequence as amino acids
            names: protein names

        Returns:
            Embedding of size (N_SEQS, SEQ_LEN, FEAT_DIM * N_LAYERS) for token embeddings
                and (N_SEQS, FEAT_DIM * N_LAYERS) for sequence embeddings. Use
        """
        if (
            names is None
            and isinstance(seqs, list)
            and isinstance(seqs[0], list)
            and len(seqs[0]) == 2
        ):
            names, seqs = zip(*seqs)
            seqs = list(seqs)
            names = list(names)
        return self._embed(seqs, names)

    def __call__(
        self,
        seqs: List[str],
        names: Optional[List[str]] = None,
        ignore_errors: bool = False,
        enforce_dtype: bool = True,
        **kwargs,
    ):
        r"""
        Compute molecular representation of a protein sequence.
        If ignore_error is True, a list of features and valid ids are returned.

        Args:
            seqs: list of protein sequence as amino acids
            names: protein names
            enforce_dtype: whether to enforce the instance dtype in the generated fingerprint
            ignore_errors: Whether to ignore errors during featurization or raise an error.
            kwargs: Named parameters for the transform method

        Returns:
            feats: list of valid embeddings
            ids: all valid positions that did not failed during featurization.
                Only returned when ignore_errors is True.

        """
        features = self.transform(seqs, names, ignore_errors=ignore_errors, **kwargs)
        ids = np.arange(len(features))
        if ignore_errors:
            features, ids = self._filter_none(features)
        if self.dtype is not None and enforce_dtype:
            if self.contact or not self._representation.startswith("seq"):
                features = [
                    datatype.cast(feat, dtype=self.dtype, columns=self.columns) for feat in features
                ]
            else:
                features = datatype.cast(features, dtype=self.dtype, columns=self.columns)
        if not ignore_errors:
            return features
        return features, ids
