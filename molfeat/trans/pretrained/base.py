from typing import Optional
from typing import Union
from typing import Callable
from typing import List
from collections.abc import Iterable
from functools import lru_cache

import copy
import datamol as dm

from loguru import logger
from molfeat.utils.commons import _parse_to_evaluable_str
from molfeat.trans.base import MoleculeTransformer
from molfeat.utils.cache import DataCache
from molfeat.store.loader import PretrainedModel


class PretrainedMolTransformer(MoleculeTransformer):
    r"""
    Transformer based on pretrained featurizer

    !!! note
        * When implementing a subclass of this class, you need to define the `_embed` and optionally the `_convert` methods.
        * If your model is an instance of PretrainedModel that handles loading of the model from a store or through a complex mechanism
          then you can decide whether you want to preload the true underlying model. You will be in charge of handling the logic of when you need to call preload, and when you don't.
          Note however that by default preloading is only attempted when the featurizer is still an instance of PretrainedModel.


    Attributes
        featurizer (object): featurizer object
        dtype (type, optional): Data type. Use call instead
        precompute_cache: (bool, optional): Whether to precompute the features into a local cache. Defaults to False.
            Note that due to molecular hashing, some pretrained featurizers might be better off just not using any cache as they can be faster.
            Furthermore, the cache is not saved when pickling the object. If you want to save the cache, you need to save the object separately.
        _require_mols (bool): Whether the embedding takes mols or smiles as input
        preload: whether to preload the pretrained model from the store (if available) during initialization.

    """

    def __init__(
        self,
        dtype: Optional[Callable] = None,
        precompute_cache: Optional[Union[bool, DataCache]] = None,
        preload: bool = False,
        **params,
    ):
        self._save_input_args()

        featurizer = params.pop("featurizer", None)
        super().__init__(dtype=dtype, featurizer="none", **params)
        self.featurizer = featurizer
        self._require_mols = False
        self.preload = preload
        self._feat_length = None
        if precompute_cache is False:
            precompute_cache = None
        if precompute_cache is True:
            name = str(self.__class__.__name__)
            precompute_cache = DataCache(name=name)
        self.precompute_cache = precompute_cache

    def set_cache(self, cache: DataCache):
        """Set the cache for the transformer

        Args:
            cache: cache object
        """
        self.precompute_cache = cache

    def _get_param_names(self):
        """Get parameter names for the estimator"""
        out = self._input_params.keys()
        out = [x for x in out if x != "featurizer"]
        return out

    def _embed(self, smiles: str, **kwargs):
        """Compute molecular embeddings for input list of smiles
        This functiom takes a list of smiles or molecules and return the featurization
        corresponding to the inputs.  In `transform` and `_transform`, this function is
        called after calling `_convert`

        Args:
            smiles: input smiles
        """
        raise NotImplementedError

    def _preload(self):
        """Preload the pretrained model for later queries"""
        if self.featurizer is not None and isinstance(self.featurizer, PretrainedModel):
            self.featurizer = self.featurizer.load()
            self.preload = True

    def __getstate__(self):
        """Getting state to allow pickling"""
        d = copy.deepcopy(self.__dict__)
        d["precompute_cache"] = None
        if isinstance(getattr(self, "featurizer", None), PretrainedModel) or self.preload:
            d.pop("featurizer", None)
        return d

    def __setstate__(self, d):
        """Setting state during reloading pickling"""
        self.__dict__.update(d)
        self._update_params()

    def fit(self, *args, **kwargs):
        return self

    def _convert(self, inputs: list, **kwargs):
        """Convert molecules to the right format

        In `transform` and `_transform`, this function is called before calling `_embed`

        Args:
            inputs: inputs to preprocess

        Returns:
            processed: pre-processed input list
        """
        if not self._require_mols:
            inputs = [dm.to_smiles(m) for m in inputs]
        return inputs

    def preprocess(self, inputs: list, labels: Optional[list] = None):
        """Run preprocessing on the input data
        Args:
            inputs: list of input molecules
            labels: list of labels
        """
        out = super().preprocess(inputs, labels)
        if self.precompute_cache not in [False, None]:
            try:
                self.transform(inputs)
            except:
                pass
        return out

    def _transform(self, mol: dm.Mol, **kwargs):
        r"""
        Compute features for a single molecule.
        This method would potentially need to be reimplemented by any child class

        Args:
            mol (dm.Mol): molecule to transform into features

        Returns
            feat: featurized input molecule

        """
        feat = None
        if self.precompute_cache is not None:
            feat = self.precompute_cache.get(mol)
        if feat is None:
            try:
                mols = [dm.to_mol(mol)]
                mols = self._convert(mols, **kwargs)
                feat = self._embed(mols, **kwargs)
                feat = feat[0]
            except Exception as e:
                if self.verbose:
                    logger.error(e)

            if self.precompute_cache is not None:
                self.precompute_cache[mol] = feat
        return feat

    def transform(self, smiles: List[str], **kwargs):
        """Perform featurization of the input molecules

        The featurization process is as follow:
        1. convert the input molecules into the right format, expected by the pre-trained model using `_convert`
        2. compute embedding of the molecule using `_embed`
        3. perform any model-specific postprocessing and cache update

        The dtype returned is the native datatype of the transformer.
        Use `__call__` to get the dtype in the `dtype` attribute format

        Args:
            mols: a list containing smiles or mol objects

        Returns:
            out: featurized molecules
        """
        if isinstance(smiles, str) or not isinstance(smiles, Iterable):
            smiles = [smiles]

        n_mols = len(smiles)
        ind_to_compute = dict(zip(range(n_mols), range(n_mols)))
        pre_computed = [None] * n_mols

        if self.precompute_cache not in [False, None]:
            ind_to_compute = {}
            pre_computed = self.precompute_cache.fetch(smiles)
            ind = 0
            for i, v in enumerate(pre_computed):
                if v is None:
                    ind_to_compute[i] = ind
                    ind += 1

        parallel_kwargs = getattr(self, "parallel_kwargs", {})
        mols = dm.parallelized(
            dm.to_mol, smiles, n_jobs=getattr(self, "n_jobs", 1), **parallel_kwargs
        )
        mols = [mols[i] for i in ind_to_compute]

        if len(mols) > 0:
            converted_mols = self._convert(mols, **kwargs)
            out = self._embed(converted_mols, **kwargs)

            if not isinstance(out, list):
                out = list(out)

            if self.precompute_cache is not None:
                # cache value now
                self.precompute_cache.update(dict(zip(mols, out)))
        out = [
            out[ind_to_compute[i]] if i in ind_to_compute else pre_computed[i]
            for i in range(n_mols)
        ]
        return out

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return str(self) == str(other)
        return False

    def _update_params(self):
        self._fitted = False

    def __len__(self):
        if self._feat_length is None:
            self._preload()
            tmp_mol = dm.to_mol("CCC")
            embs = self._transform(tmp_mol)
            self._feat_length = len(embs)
        return self._feat_length

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return "{}(dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self.dtype),
        )
