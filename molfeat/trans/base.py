from pathlib import Path
from typing import Mapping
from typing import Union
from typing import List
from typing import Any
from typing import Callable
from typing import Optional
from typing import Dict
from collections.abc import Iterable

import abc
import copy
import joblib
import json

import yaml
import fsspec
import pandas as pd
import datamol as dm
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from loguru import logger

from molfeat._version import __version__ as MOLFEAT_VERSION

from molfeat.calc import get_calculator
from molfeat.calc.base import _CALCULATORS
from molfeat.utils import datatype
from molfeat.utils.cache import _Cache, FileCache, MPDataCache
from molfeat.utils.cache import CacheList
from molfeat.utils.commons import fn_to_hex
from molfeat.utils.commons import hex_to_fn
from molfeat.utils.commons import is_callable
from molfeat.utils.parsing import get_input_args
from molfeat.utils.parsing import import_from_string
from molfeat.utils.state import map_dtype
from molfeat.utils.state import ATOM_FEATURIZER_MAPPING
from molfeat.utils.state import BOND_FEATURIZER_MAPPING
from molfeat.utils.state import ATOM_FEATURIZER_MAPPING_REVERSE
from molfeat.utils.state import BOND_FEATURIZER_MAPPING_REVERSE

_TRANSFORMERS = {}


class _TransformerMeta(abc.ABCMeta):
    """Metaclass to register all transformers automatically"""

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)
        if name in _TRANSFORMERS.keys():
            logger.warning(
                f"The {name!r} interaction has been superseded by a "
                f"new class with id {id(cls):#x}"
            )
        # ignore private classes
        if name not in ["BaseFeaturizer"] and not name.startswith("_"):
            # do not register the base class
            _TRANSFORMERS[name] = cls


class BaseFeaturizer(BaseEstimator):
    """
    Molecule featurizer base class that needs to be implemented by all featurizers.
    This featurizer is compatible with scikit-learn estimators and thus can be plugged into a pipeline
    """

    def __init__(
        self,
        n_jobs: int = 1,
        verbose: bool = True,
        dtype: Optional[Union[str, Callable]] = None,
        parallel_kwargs: Optional[Dict[str, Any]] = None,
        **params,
    ):
        self._n_jobs = n_jobs
        self.dtype = dtype
        self.verbose = verbose
        self.parallel_kwargs = parallel_kwargs or {}
        for k, v in params.items():
            setattr(self, k, v)
        self._input_params = dict(n_jobs=n_jobs, dtype=dtype, verbose=verbose, **params)

    @property
    def n_jobs(self):
        """Get the number of concurrent jobs to run with this featurizer"""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if val >= 1:
            self._n_jobs = val
        elif val == -1:
            self._n_jobs = joblib.cpu_count()

    def _get_param_names(self):
        """Get parameter names for the estimator"""
        return self._input_params.keys()

    def _update_params(self):
        """Update parameters of the current estimator"""
        ...

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Returns:
            self: estimator instance
        """
        super().set_params(**params)
        for k, v in params.items():
            if k in self._input_params:
                self._input_params[k] = v
        self._update_params()
        return self

    def copy(self):
        """Return a copy of this object."""
        copy_obj = self.__class__(**self._input_params)
        for k, v in self.__dict__.items():
            if not hasattr(copy_obj, k):
                setattr(copy_obj, k, copy.deepcopy(v))
        return copy_obj

    def preprocess(self, inputs: list, labels: Optional[list] = None):
        """Preprocess input

        Args:
            inputs: inputs to preprocess
            labels: labels to preprocess (optional)

        Returns:
            processed: pre-processed input list
        """
        return inputs, labels

    def get_collate_fn(self, *args, **kwargs):
        """
        Get collate function of this featurizer. In the implementation of this function
        you should set the relevant attributes or argument of the underlying collate function
        (e.g via functools.partial) and return the function itself

        Returns:
            fn: Collate function for pytorch or None
        """
        return None


class MoleculeTransformer(TransformerMixin, BaseFeaturizer, metaclass=_TransformerMeta):

    """
    Base class for molecular data transformer such as Fingerprinter etc.
    If you create a subclass of this featurizer, you will need to make sure that the
    input argument of the init are kept as is in the object attributes.

    !!! note
        The transformer supports a variety of datatype, they are only enforced when passing the
        `enforce_dtype=True` attributes in `__call__`. For pandas dataframes, use `'pandas'|'df'|'dataframe'|pd.DataFrame`

    ???+ tip "Using a custom Calculator"
        You can use your own calculator for featurization. It's recommended to subclass `molfeat.calc.base.SerializableCalculator`
        If you calculator also implements a `batch_compute` method, it will be used for batch featurization and parallelization options will be passed to it.
    """

    def __init__(
        self,
        featurizer: Union[str, Callable],
        n_jobs: int = 1,
        verbose: bool = False,
        dtype: Optional[Union[str, Callable]] = None,
        parallel_kwargs: Optional[Dict[str, Any]] = None,
        **params,
    ):
        """Mol transformer base class

        Args:
            featurizer: featurizer to use
            n_jobs (int, optional): Number of job to run in parallel. Defaults to 1.
            verbose (bool, optional): Verbosity level. Defaults to True.
            dtype (callable, optional): Output data type. Defaults to None, where numpy arrays are returned.
            parallel_kwargs (dict, optional): Optional kwargs to pass to the dm.parallelized function. Defaults to None.

        """
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype,
            featurizer=featurizer,
            parallel_kwargs=parallel_kwargs,
            **params,
        )
        if callable(featurizer):
            self.featurizer = featurizer
        else:
            self.featurizer = get_calculator(featurizer, **params)

        self.cols_to_keep = None
        self._fitted = False

        self._save_input_args()
        if self.featurizer and not (
            isinstance(self.featurizer, str) or is_callable(self.featurizer)
        ):
            raise AttributeError(f"Featurizer {self.featurizer} must be a callable or a string")

    def _save_input_args(self):
        """Save the input arguments of a transformer to the attribute
        `_input_args` of the object.
        """

        # NOTE(hadim): don't override existing _input_args so
        # it's possible to use MoleculeTransformer as a featurizer
        # instead of simply a base class.
        if not hasattr(self, "_input_args"):
            self._input_args = get_input_args()

    def _update_params(self):
        if not callable(self.featurizer):
            params = copy.deepcopy(self._input_params)
            params.pop("featurizer")
            self.featurizer = get_calculator(self.featurizer, **params)
        self._fitted = False

    def __setstate__(self, state):
        state.pop("callbacks", None)
        self.__dict__.update(state)
        self.__dict__["parallel_kwargs"] = state.get("parallel_kwargs", {})
        self._update_params()

    def fit(self, X: List[Union[dm.Mol, str]], y: Optional[list] = None, **fit_params):
        """Fit the current transformer on given dataset.

        The goal of fitting is for example to identify nan columns values
        that needs to be removed from the dataset

        Args:
            X: input list of molecules
            y (list, optional): Optional list of molecular properties. Defaults to None.

        Returns:
            self: MolTransformer instance after fitting
        """
        feats = self.transform(X, ignore_errors=True)
        lengths = [len(x) for x in feats if not datatype.is_null(x)]
        if lengths:
            # we will ignore all nan
            feats = datatype.to_numpy([f for f in feats if not datatype.is_null(f)])
            self.cols_to_keep = (~np.any(np.isnan(feats), axis=0)).nonzero()[0]
        self._fitted = True
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
        feat = None
        try:
            feat = datatype.to_numpy(self.featurizer(mol))
            if self.cols_to_keep is not None:
                feat = feat[self.cols_to_keep]
        except Exception as e:
            if self.verbose:
                logger.error(e)
        return feat

    def transform(
        self,
        mols: List[Union[dm.Mol, str]],
        ignore_errors: bool = False,
        **kwargs,
    ):
        r"""
        Compute the features for a set of molecules.

        !!! note
            Note that depending on the `ignore_errors` argument, all failed
            featurization (caused whether by invalid smiles or error during
            data transformation) will be substitued by None features for the
            corresponding molecule. This is done, so you can find the positions
            of these molecules and filter them out according to your own logic.

        Args:
            mols: a list containing smiles or mol objects
            ignore_errors (bool, optional): Whether to silently ignore errors


        Returns:
            features: a list of features for each molecule in the input set
        """
        # Convert single mol to iterable format
        if isinstance(mols, (str, dm.Mol)) or not isinstance(mols, Iterable):
            mols = [mols]

        def _to_mol(x):
            return dm.to_mol(x) if x else None

        parallel_kwargs = getattr(self, "parallel_kwargs", {})

        if hasattr(self.featurizer, "batch_compute") and callable(self.featurizer.batch_compute):
            # this calculator can be batched which will be faster
            features = self.featurizer.batch_compute(mols, n_jobs=self.n_jobs, **parallel_kwargs)
        else:
            mols = dm.parallelized(_to_mol, mols, n_jobs=self.n_jobs, **parallel_kwargs)
            if self.n_jobs not in [0, 1]:
                # use a proxy model to run in parallel
                cpy = self.copy()
                features = dm.parallelized(
                    cpy._transform,
                    mols,
                    n_jobs=self.n_jobs,
                    **cpy.parallel_kwargs,
                )
            else:
                features = [self._transform(mol) for mol in mols]
        if not ignore_errors:
            for ind, feat in enumerate(features):
                if feat is None:
                    raise ValueError(
                        f"Cannot transform molecule at index {ind}. Please check logs (set verbose to True) to see errors!"
                    )

        return features

    def __len__(self):
        """Compute featurizer length"""

        # check length and _length attribute
        cols_to_keep = getattr(self, "cols_to_keep", None)
        cur_length = None

        if cols_to_keep is not None:
            cur_length = len(cols_to_keep)
        else:
            cur_length = getattr(self, "length", getattr(self, "_length", None))
            # then check the featurizer length if it's a callable and not a string/None
            if (
                cur_length is None
                and callable(self.featurizer)
                and hasattr(self.featurizer, "__len__")
            ):
                cur_length = len(self.featurizer)

        if cur_length is None:
            raise ValueError(
                f"Cannot auto-determine length of this MolTransformer: {self.__class__.__name__}"
            )

        return cur_length

    def __call__(
        self,
        mols: List[Union[dm.Mol, str]],
        enforce_dtype: bool = True,
        ignore_errors: bool = False,
        **kwargs,
    ):
        r"""
        Calculate features for molecules. Using __call__, instead of transform.
        If ignore_error is True, a list of features and valid ids are returned.
        Note that most Transfomers allow you to specify
        a return datatype.

        Args:
            mols:  Mol or SMILES of the molecules to be transformed
            enforce_dtype: whether to enforce the instance dtype in the generated fingerprint
            ignore_errors: Whether to ignore errors during featurization or raise an error.
            kwargs: Named parameters for the transform method

        Returns:
            feats: list of valid features
            ids: all valid molecule positions that did not failed during featurization.
                Only returned when ignore_errors is True.

        """
        features = self.transform(mols, ignore_errors=ignore_errors, enforce_dtype=False, **kwargs)
        ids = np.arange(len(features))
        if ignore_errors:
            features, ids = self._filter_none(features)
        if self.dtype is not None and enforce_dtype:
            features = datatype.cast(features, dtype=self.dtype, columns=self.columns)
        if not ignore_errors:
            return features
        return features, ids

    @staticmethod
    def _filter_none(features):
        ids_bad = []
        # If the features are a list, filter the None ids
        if isinstance(features, (tuple, list, np.ndarray)):
            for f_id, feat in enumerate(features):
                if datatype.is_null(feat):
                    ids_bad.append(f_id)
            ids_to_keep = [
                this_id for this_id in np.arange(0, len(features)) if this_id not in ids_bad
            ]
            features = [features[ii] for ii in ids_to_keep]

        # If the features are a dict or DataFrame, filter the ids when any key id is None
        elif isinstance(features, (dict, pd.DataFrame)):
            if isinstance(features, dict):
                features = pd.DataFrame(features)
            for feat_col in features.columns:
                for f_id, feat in enumerate(features[feat_col].values.flatten()):
                    if feat is None:
                        ids_bad.append(f_id)
            ids_bad = np.unique(ids_bad).tolist()
            all_ids = np.arange(0, features.shape[0])
            ids_to_keep = [this_id for this_id in all_ids if this_id not in ids_bad]
            features = features.iloc[ids_to_keep, :]

        else:
            ids_to_keep = np.arange(0, features.shape[0])
        return features, list(ids_to_keep)

    @property
    def columns(self):
        """Get the list of columns for this molecular descriptor

        Returns:
            columns (list): Name of the columns of the descriptor
        """
        columns = getattr(self.featurizer, "columns", None)
        cols_to_keep = getattr(self, "cols_to_keep", None)
        if columns is not None and cols_to_keep is not None and len(cols_to_keep) > 0:
            columns = [columns[i] for i in cols_to_keep]
        return columns

    @staticmethod
    def batch_transform(
        transformer: Callable,
        mols: List[Union[dm.Mol, str]],
        batch_size: int = 256,
        n_jobs: Optional[int] = None,
        concatenate: bool = True,
        progress: bool = True,
        leave_progress: bool = False,
        **parallel_kwargs,
    ):
        """
        Batched computation of featurization of a list of molecules

        Args:
            transformer: Fingerprint transformer
            mols: List of molecules to featurize
            batch_size: Batch size
            n_jobs: number of jobs to run in parallel
            concatenate: Whether to concatenate the results or return the list of batched results
            progress: whether to show progress bar
            leave_progress: whether to leave progress bar after completion
            parallel_kwargs: additional arguments to pass to dm.parallelized

        Returns:
            List of batches
        """

        step_size = int(np.ceil(len(mols) / batch_size))
        batched_mols = np.array_split(mols, step_size)

        tqdm_kwargs = parallel_kwargs.setdefault("tqdm_kwargs", {})
        tqdm_kwargs.update(leave=leave_progress, desc="Batch compute:")
        parallel_kwargs["tqdm_kwargs"] = tqdm_kwargs

        # it's recommended to use a precomputed molecule transformer
        # instead of the internal cache for pretrained models
        cache_attr = "cache"
        existing_cache = getattr(transformer, cache_attr, None)
        if existing_cache is None:
            cache_attr = "precompute_cache"
            existing_cache = getattr(transformer, cache_attr, None)

        use_mp_cache = (
            existing_cache is not None
            and not isinstance(existing_cache, MPDataCache)
            and n_jobs not in [None, 0, 1]  # this is based on datamol sequential vs parallel
        )
        if use_mp_cache:
            # we need to change the cache system to one that works with multiprocessing
            # to have a shared memory
            new_cache = MPDataCache()
            new_cache.update(existing_cache)
            setattr(transformer, cache_attr, new_cache)

        transformed = dm.parallelized(
            transformer,
            batched_mols,
            n_jobs=n_jobs,
            progress=progress,
            **parallel_kwargs,
        )
        if use_mp_cache:
            # we set back the original transformation while updating it with
            # all the missing values
            existing_cache.update(getattr(transformer, cache_attr, {}))
            setattr(transformer, cache_attr, existing_cache)

        if concatenate:
            # if we ask for concatenation, then we would need to fix None values ideally
            fixed_transformations = []
            for computed_trans in transformed:
                if computed_trans is None:
                    computed_trans = np.full(len(computed_trans), len(transformer), np.nan)
                else:
                    for i, x in enumerate(computed_trans):
                        if x is None:
                            computed_trans[i] = np.full(len(transformer), np.nan)
                fixed_transformations.append(computed_trans)
            return np.concatenate(fixed_transformations)
        return transformed

    # Featurizer to state methods

    def to_state_dict(self) -> dict:
        """Serialize the featurizer to a state dict."""

        if getattr(self, "_input_args") is None:
            raise ValueError(f"Cannot save state for this transformer '{self.__class__.__name__}'")

        # Process the input arguments before building the state
        args = copy.deepcopy(self._input_args)

        # Deal with dtype
        if "dtype" in args and not isinstance(args["dtype"], str):
            args["dtype"] = map_dtype(args["dtype"])

        ## Deal with graph atom/bond featurizers
        # NOTE(hadim): it's important to highlight that atom/bond featurizers can't be
        # customized with this logic.
        if args.get("atom_featurizer") is not None:
            if hasattr(args.get("atom_featurizer"), "to_state_dict"):
                args["atom_featurizer"] = args["atom_featurizer"].to_state_dict()
                args["_atom_featurizer_is_pickled"] = False
            else:
                logger.warning
                (
                    "You are attempting to pickle an atom featurizer without a `to_state_dict` function into a hex string"
                )
                args["atom_featurizer"] = fn_to_hex(args["atom_featurizer"])
                args["_atom_featurizer_is_pickled"] = True

        # deal with bond featurizer
        if args.get("bond_featurizer") is not None:
            if hasattr(args.get("bond_featurizer"), "to_state_dict"):
                args["bond_featurizer"] = args["bond_featurizer"].to_state_dict()
                args["_bond_featurizer_is_pickled"] = False
            else:
                logger.warning(
                    "You are attempting to pickle a bond featurizer without a `to_state_dict` function into a hex string"
                )
                args["bond_featurizer"] = fn_to_hex(args["bond_featurizer"])
                args["_bond_featurizer_is_pickled"] = True

        ## Deal with custom featurizer
        if "featurizer" in args and isinstance(args["featurizer"], Callable):
            if hasattr(args["featurizer"], "to_state_dict"):
                args["featurizer"] = args["featurizer"].to_state_dict()
                args["_featurizer_is_pickled"] = False
            else:
                logger.warning(
                    "You are attempting to pickle a callable without a `to_state_dict` function into a hex string"
                )
                args["featurizer"] = fn_to_hex(args["featurizer"])
                args["_featurizer_is_pickled"] = True

        # Build the state
        state = {}
        state["name"] = self.__class__.__name__
        state["args"] = args
        state["_molfeat_version"] = MOLFEAT_VERSION
        return state

    def to_state_json(self) -> str:
        return json.dumps(self.to_state_dict())

    def to_state_yaml(self) -> str:
        return yaml.dump(self.to_state_dict(), Dumper=yaml.SafeDumper)

    def to_state_json_file(self, filepath: Union[str, Path]):
        with fsspec.open(filepath, "w") as f:
            f.write(self.to_state_json())  # type: ignore

    def to_state_yaml_file(self, filepath: Union[str, Path]):
        with fsspec.open(filepath, "w") as f:
            f.write(self.to_state_yaml())  # type: ignore

    # State to featurizer methods

    @staticmethod
    def from_state_dict(state: dict, override_args: Optional[dict] = None) -> "MoleculeTransformer":
        """Reload a featurizer from a state dict."""

        # Don't alter the original state dict
        state = copy.deepcopy(state)

        # MoleculeTransformer is a special case that has his own logic
        if state["name"] == "PrecomputedMolTransformer":
            return PrecomputedMolTransformer.from_state_dict(
                state=state,
                override_args=override_args,
            )

        # Get the name
        transformer_class = _TRANSFORMERS.get(state["name"])
        if transformer_class is None:
            raise ValueError(f"The featurizer '{state['name']}' is not supported.")
        if isinstance(transformer_class, str):
            # Get the transformer class from its path
            transformer_class = import_from_string(transformer_class)

        # Process the state as needed
        args = state.get("args", {})

        # Deal with dtype
        if "dtype" in args and isinstance(args["dtype"], str):
            args["dtype"] = map_dtype(args["dtype"])

        ## Deal with graph atom/bond featurizers
        if args.get("atom_featurizer") is not None:
            if not args.get("_atom_featurizer_is_pickled"):
                klass_name = args["atom_featurizer"].get("name")
                args["atom_featurizer"] = ATOM_FEATURIZER_MAPPING_REVERSE[
                    klass_name
                ].from_state_dict(args["atom_featurizer"])
            else:
                # buffer = io.BytesIO(bytes.fromhex(args["atom_featurizer"]))
                # args["atom_featurizer"] = joblib.load(buffer)
                args["atom_featurizer"] = hex_to_fn(args["atom_featurizer"])
            args.pop("_atom_featurizer_is_pickled", None)
        if args.get("bond_featurizer") is not None:
            if not args.get("_bond_featurizer_is_pickled"):
                klass_name = args["bond_featurizer"].get("name")
                args["bond_featurizer"] = BOND_FEATURIZER_MAPPING_REVERSE[
                    klass_name
                ].from_state_dict(args["bond_featurizer"])
            else:
                args["bond_featurizer"] = hex_to_fn(args["bond_featurizer"])
            args.pop("_bond_featurizer_is_pickled", None)
        ## Deal with custom featurizer
        if "featurizer" in args:
            if args.get("_featurizer_is_pickled") is True:
                args["featurizer"] = hex_to_fn(args["featurizer"])
                args.pop("_featurizer_is_pickled")
            elif (
                isinstance(args["featurizer"], Mapping)
                and args["featurizer"].get("name") in _CALCULATORS
            ):
                # we have found a known calculator
                klass_name = args["featurizer"].get("name")
                args["featurizer"] = _CALCULATORS[klass_name].from_state_dict(args["featurizer"])
                args.pop("_featurizer_is_pickled")

        if override_args is not None:
            args.update(override_args)

        # Create the transformer
        featurizer = transformer_class(**args)
        return featurizer

    @staticmethod
    def from_state_json(
        state_json: str,
        override_args: Optional[dict] = None,
    ) -> "MoleculeTransformer":
        state_dict = json.loads(state_json)
        return MoleculeTransformer.from_state_dict(state_dict, override_args=override_args)

    @staticmethod
    def from_state_yaml(
        state_yaml: str,
        override_args: Optional[dict] = None,
    ) -> "MoleculeTransformer":
        state_dict = yaml.load(state_yaml, Loader=yaml.SafeLoader)
        return MoleculeTransformer.from_state_dict(state_dict, override_args=override_args)

    @staticmethod
    def from_state_json_file(
        filepath: Union[str, Path],
        override_args: Optional[dict] = None,
    ) -> "MoleculeTransformer":
        with fsspec.open(filepath, "r") as f:
            featurizer = MoleculeTransformer.from_state_json(f.read(), override_args=override_args)  # type: ignore
        return featurizer

    @staticmethod
    def from_state_yaml_file(
        filepath: Union[str, Path],
        override_args: Optional[dict] = None,
    ) -> "MoleculeTransformer":
        with fsspec.open(filepath, "r") as f:
            featurizer = MoleculeTransformer.from_state_yaml(f.read(), override_args=override_args)  # type: ignore
        return featurizer


class PrecomputedMolTransformer(MoleculeTransformer):
    """Convenience class for storing precomputed features."""

    def __init__(
        self,
        cache: Optional[Union[_Cache, Mapping[Any, Any], CacheList]] = None,
        cache_dict: Optional[Dict[str, Union[_Cache, Mapping[Any, Any], CacheList]]] = None,
        cache_key: Optional[str] = None,
        *args,
        featurizer: Optional[Union[MoleculeTransformer, str]] = None,
        state_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Transformer that supports precomputation of features. You can either pass an empty cache or a pre-populated cache

        Args:
            cache: a datastructure of type mapping that maps each molecule to the precomputed features
            cache_dict: A dictionary of cache objects. This is a convenient structure when use multiple
                datacache for model selection.
            cache_key: The key of cache object to use.
            featurizer: optional featurizer used to compute the features of values not in the cache.
                Either the featurizer object or a string.
            state_path: optional state file path used to initiate the transformer object at the initialization
        """
        if (state_path is not None) and (
            (cache is not None) or (cache_dict is not None and cache_key is not None)
        ):
            raise ValueError(
                "`PrecomputedMolTransformer` can only be initiated by either `state_path` or"
                " the rest of parameters for cache and featurizer. But both are given."
            )

        super().__init__(*args, featurizer="none", **kwargs)

        if state_path is not None:
            self.__dict__ = self.from_state_file(state_path=state_path).__dict__.copy()
        else:
            if cache_dict is not None and cache_key is not None:
                self.cache_key = cache_key
                self.cache = cache_dict[self.cache_key]
            elif cache is not None:
                self.cache = cache
            else:
                raise AttributeError("The cache is not specified.")

            if isinstance(featurizer, str):
                self.base_featurizer = MoleculeTransformer(featurizer, *args, **kwargs)
            else:
                self.base_featurizer = featurizer

        # Set the length of the featurizer
        if len(self.cache) > 0:
            self.length = len(list(self.cache.values())[0])
        elif self.base_featurizer is not None:
            self.length = len(self.base_featurizer)
        else:
            raise AttributeError(
                "The cache is empty and the base featurizer is not specified. It's impossible"
                " to determine the length of the featurizer."
            )

    def _transform(self, mol: dm.Mol):
        r"""
        Return precomputed feature for a single molecule

        Args:
            mol (dm.Mol): molecule to transform into features

        Returns
            feat: featurized input molecule

        """
        feat = self.cache.get(mol)
        # if feat is None and we have an existing featurizer, we can update the cache
        if feat is None and self.base_featurizer is not None:
            feat = self.base_featurizer._transform(mol)
            self.cache[mol] = feat

        try:
            feat = datatype.to_numpy(feat)
            if self.cols_to_keep is not None:
                feat = feat[self.cols_to_keep]
        except Exception as e:
            if self.verbose:
                logger.error(e)
        return feat

    def update(self, feat_dict: Mapping[Any, Any]):
        r"""
        Fill the cache with new set of features for the molecules in mols.

        Args:
            feat_dict: A dictionary of molecules to features.
        """
        self.cache.update(feat_dict)

    def __getstate__(self):
        """Get the state for pickling"""
        state = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in ["cache"]}
        if isinstance(self.cache, FileCache):
            state["file_cache_args"] = dict(
                cache_file=self.cache.cache_file,
                name=self.cache.name,
                mol_hasher=self.cache.mol_hasher,
                n_jobs=self.cache.n_jobs,
                verbose=self.cache.verbose,
                file_type=self.cache.file_type,
                parquet_kwargs=self.cache.parquet_kwargs,
            )
        else:
            # EN: we do not copy the cache
            state["cache"] = self.cache
        return state

    def __setstate__(self, state):
        if "file_cache_args" in state:
            cache = FileCache(**state.pop("file_cache_args"))
            state["cache"] = cache
        return super().__setstate__(state)

    def to_state_dict(self, save_to_file: bool = True) -> dict:
        """Serialize a PrecomputedMolTransformer object to a state dict.

        Notes:
            - The base_featurizer must be set or a ValueError will be raised.
            - The cache must be a FileCache object or a ValueError will be raised.

        Args:
            save_to_file: whether to save the cache to file.
        """

        if self.base_featurizer is None:
            raise ValueError(
                "You can't serialize a PrecomputedMolTransformer that does not contain a"
                " featurizer."
            )

        if not isinstance(self.cache, FileCache):
            raise ValueError("The cache must be a FileCache object.")

        state = {}
        state["name"] = "PrecomputedMolTransformer"
        state["base_featurizer"] = self.base_featurizer.to_state_dict()
        state["cache"] = self.cache.to_state_dict(save_to_file=save_to_file)
        state["_molfeat_version"] = MOLFEAT_VERSION

        return state

    @staticmethod
    def from_state_dict(
        state: dict,
        override_args: Optional[dict] = None,
    ) -> "PrecomputedMolTransformer":
        # Don't alter the original state dict
        state = copy.deepcopy(state)

        args = {}

        # Load the FileCache object
        args["cache"] = FileCache.from_state_dict(state["cache"])

        # Load the base featurizer
        args["featurizer"] = MoleculeTransformer.from_state_dict(state["base_featurizer"])

        if override_args is not None:
            args.update(override_args)

        # Doesn't allow state_path in the initiation args
        args.pop("state_path", None)
        return PrecomputedMolTransformer(**args)

    def from_state_file(
        self,
        state_path: str,
        override_args: Optional[dict] = None,
    ) -> "PrecomputedMolTransformer":
        if state_path.endswith("yaml") or state_path.endswith("yml"):
            return self.from_state_yaml_file(filepath=state_path, override_args=override_args)
        elif state_path.endswith("json"):
            return self.from_state_json_file(filepath=state_path, override_args=override_args)
        else:
            raise ValueError(
                "Only files with 'yaml' or 'json' format are allowed. "
                "The filename must be ending with `yaml`, 'yml' or 'json'."
            )
