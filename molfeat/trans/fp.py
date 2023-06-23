from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import re
import copy
import numpy as np
import datamol as dm

from molfeat.calc import get_calculator, FP_FUNCS
from molfeat.trans.base import MoleculeTransformer
from molfeat.utils import datatype
from molfeat.utils.commons import _parse_to_evaluable_str

_UNSERIALIZABLE_FPS = []


class FPVecTransformer(MoleculeTransformer):
    r"""
    Molecular fingerprinter that computes various fingerprints and descriptors regularly used in QSAR modeling.

    !!! note
        For fingerprints with a radius parameter, you can provide the radius using the notation "fp:radius".
        For example "Morgan Circular 2" can be written as "morgan:2". Under the hood, morgan and ecfp fingerprints
        are equated with the proper radius/diameter adjustment.

        For counting fingerprints, you just need to add the '-count' suffix to the name of the fingerprint. For example:
        "morgan-count:2"
    """

    AVAILABLE_FPS = list(FP_FUNCS.keys()) + [
        "desc3D",
        "desc2D",
        "cats2D",
        "cats3D",
        "pharm2D",
        "pharm3D",
        "scaffoldkeys",
        "skeys",
        "electroshape",
        "usr",
        "usrcat",
    ]

    def __init__(
        self,
        kind: str = "ecfp:4",
        length: int = 2000,
        n_jobs: int = 1,
        verbose: bool = False,
        dtype: Callable = np.float32,
        parallel_kwargs: Optional[dict] = None,
        **params,
    ):
        """Molecular to vector fingerprinter

        Args:
            kind (str, optional): Name of the fingerprints (one supported fingerprints: see self.AVAILABLE_FPS). Defaults to "ecfp4".
            length (int, optional): Length of the fingerprint. Defaults to 2000.
            n_jobs (int, optional): Number of jobs. Defaults to 1.
            verbose (bool, optional): Verbosity level. Defaults to False.
            dtype (Callable, optional): Data type. Defaults to np.float32.
            parallel_kwargs (dict, optional): Optional arguments to pass to dm.parallelized when required. Defaults to None.
            params (dict, optional): Any additional parameters to the fingerprint function
        """
        self._save_input_args()

        # remove any featurizer that was passed as argument
        params.pop("featurizer", None)
        self._feat_params = params
        featurizer = self._prepare_featurizer(kind, length, **params)
        super().__init__(
            featurizer=featurizer,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype,
            parallel_kwargs=parallel_kwargs,
            **params,
        )
        self.kind = kind
        self.length = length
        self._length = None
        # update length for featurizer that have they fixed length
        # EN: setting up a protected _length function helps to bypass
        # the absurd "is" comparison done by sklearn in clone
        # note that the featurizer length would likely be ignored by featurizer
        # that do not support a variable length
        if hasattr(self.featurizer, "__len__"):
            self._length = len(featurizer)
        self._input_params.update(kind=kind, length=length)
        if self.kind.lower() in _UNSERIALIZABLE_FPS:
            self.parallel_kwargs.update(scheduler="threads")

    def __len__(self):
        """Compute featurizer length"""
        if getattr(self, "cols_to_keep", None) is None and self._length is not None:
            return self._length
        return super().__len__()

    def _get_param_names(self):
        """Get parameter names for the estimator"""
        out = self._input_params.keys()
        return [x for x in out if x != "featurizer"]

    @classmethod
    def _prepare_featurizer(cls, kind: str, length: int, **params):
        """Prepare featurizer from its name and parameters

        Args:
            kind: Name of the featurizer
            length: Length of the featurizer
        Returns:
            calculator (Callable): fingerprint calculator
        """
        match = re.search(r":(\d+)$", kind)
        radius = None
        if match is not None:
            radius = match.group(1)
        if radius is not None:
            kind = kind.replace(radius, "").strip(":").lower()
            radius = int(radius)
            if any(x in kind for x in ["ecfp", "fcfp"]):
                radius = max(radius // 2, 1)
            params["radius"] = radius
        if any(x in kind for x in ["morgan", "morgan_circular", "morgan-circular"]):
            kind.replace("_circular", "").replace("-circular", "").replace("morgan", "ecfp")
        if kind not in cls.AVAILABLE_FPS:
            raise ValueError(f"{kind} is not a valid featurizer")
        params["length"] = length

        return get_calculator(kind, **params)

    def _update_params(self):
        params = copy.deepcopy(self._input_params)
        params.pop("featurizer", None)
        params.pop("length", None)
        params.pop("kind", None)
        params.pop("verbose", None)
        params.pop("dtype", None)
        params.pop("n_jobs", None)
        self._fitted = False
        self.featurizer = self._prepare_featurizer(self.kind, self.length, **params)

    def __repr__(self):
        return "{}(kind={}, length={}, dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self.kind),
            _parse_to_evaluable_str(self.length),
            _parse_to_evaluable_str(self.dtype),
        )

    def __str__(self):
        # The output for the print function
        return self.__repr__()

    def __eq__(self, other):
        same_type = type(self) == type(other)
        return same_type and all(
            [getattr(other, k) == v for k, v in self.get_params() if not callable(v)]
        )

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(repr(self))


class FPVecFilteredTransformer(FPVecTransformer):
    r"""
    Fingerprint molecule transformer with columns filters applying to the featurized vector when `fit` is called
    """

    def __init__(
        self,
        kind: str = "ecfp:4",
        length: int = 2000,
        occ_threshold: float = 0,
        del_invariant: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        dtype: Callable = np.float32,
        **params,
    ):
        """Molecular to vector featurization with filtering applied

        Args:
            kind (str, optional): Name of the fingerprints (one supported fingerprints: see self.AVAILABLE_FPS). Defaults to "ecfp4".
            length (int, optional): Length of the fingerprint. Defaults to 2000.
            occ_threshold (float, optional): Minimum proportion a columns need to be non null to be kept.
            del_invariant (bool, optional): Whether to delete columns that are invariant.
            n_jobs (int, optional): Number of jobs. Defaults to 1.
            verbose (bool, optional): Verbosity level. Defaults to False.
            dtype (Callable, optional): Data type. Defaults to np.float32.
            params (dict, optional): Any additional parameters to the fingerprint function
        """

        super().__init__(
            kind=kind,
            length=length,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype,
            **params,
        )
        self.occ_threshold = occ_threshold
        self.del_invariant = del_invariant
        self._input_params.update(occ_threshold=occ_threshold, del_invariant=del_invariant)

    def _update_params(self):
        params = copy.deepcopy(self._input_params)
        params.pop("featurizer", None)
        params.pop("length", None)
        params.pop("kind", None)
        params.pop("verbose", None)
        params.pop("dtype", None)
        params.pop("n_jobs", None)
        params.pop("occ_threshold", None)
        params.pop("del_invariant", None)
        self.featurizer = self._prepare_featurizer(self.kind, self.length, **params)

    def __repr__(self):
        return "{} (kind={}, length={}, occ_threshold={}, del_invariant={}, dtype={})".format(
            self.__class__.__name__,
            _parse_to_evaluable_str(self.kind),
            _parse_to_evaluable_str(self.length),
            _parse_to_evaluable_str(self.occ_threshold),
            _parse_to_evaluable_str(self.del_invariant),
            _parse_to_evaluable_str(self.dtype),
        )

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
            # all nan columns
            unwanted_columns = []
            # let's adjsut occ to float
            occ_threshold = self.occ_threshold
            if occ_threshold > 1:
                occ_threshold = occ_threshold / feats.shape[0]
            # not nan
            unwanted_columns.append(~np.any(np.isnan(feats), axis=0))
            # not enough set bits
            unwanted_columns.append(
                (np.count_nonzero(feats, axis=0) / feats.shape[0]) > occ_threshold
            )
            if self.del_invariant:
                unwanted_columns.append(~np.all(feats == feats[0, :], axis=0))
            self.cols_to_keep = (np.logical_and.reduce(unwanted_columns)).nonzero()[0]
        self._fitted = True
        return self
