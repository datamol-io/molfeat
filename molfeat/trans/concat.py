from typing import Union
from typing import Iterator
from typing import Iterable
from typing import Callable
from typing import Optional
from typing import Dict
from typing import List
from typing import Any

import numpy as np
import pandas as pd
import datamol as dm

from sklearn.base import BaseEstimator
from molfeat.trans.fp import FPVecTransformer
from molfeat.utils import datatype


class FeatConcat(list, BaseEstimator):
    r"""
    Concatenation container for `FPVecTransformer`. This class allows
    merging multiple fingerprints into a single one.
    It gives the ability to call the following methods
        - `fit`
        - `transform`
        - `fit_transform`
    on a list of transformers and concatenate the results.

    !!! note
        The featurization length of this featurizer is accessible via the `length` property.
        `len()` will return the number of base featurizer.
    """

    _STR_DELIMITER = "||"

    def __init__(
        self,
        iterable: Optional[Union[Iterable, str]] = None,
        dtype: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """Featurizer concatenator

        Args:
            iterable: List of featurizer to concatenate.
            dtype: Datatype of the computed fingerprint
            params: Optional dictionary of parameters for the featurizers when there is a need for initializing them
            collate_fn: optional function to provide for custom collating.
                By default the collate function will be None, which will use the torch default
        """
        super().__init__()
        self.params = params or {}
        if isinstance(iterable, str):
            iterable = [x.strip() for x in iterable.split(self._STR_DELIMITER)]
        if iterable is not None:
            for item in iterable:
                if isinstance(item, str):
                    item = FPVecTransformer(kind=item, **self.params.get(item, {}))
                self.append(item)
        self.dtype = dtype
        self._length = None
        self.collate_fn = collate_fn

    def append(self, item):
        r"""Override the ``append`` to accept only ``FPVecTransformer``"""
        self._check_supported(item)
        super().append(item)

    def insert(self, index, item):
        r"""Override the ``insert`` to accept only ``BaseFeaturizer``"""
        self._check_suported(item)
        super().insert(index, item)

    def __add__(self, item):
        """Override the `__add__` method"""
        self._check_supported(item)
        super().__add__(item)

    def __setitem__(self, index, value):
        """Override the `__setitem__`  method"""
        self._check_supported(value)
        super().__setitem__(index, value)

    @property
    def length(self):
        """
        Length property for Feat concatenation.  This is the sum of the length of each transformer.
        Note that __len__ returns the number of base featurizer here instead.
        """
        if self._length is None:
            full_length = 0
            for feat in self:
                if isinstance(feat, FeatConcat):
                    full_length += feat.length
                else:
                    full_length += len(feat)
            self._length = full_length
        return self._length

    def _check_supported(self, item):
        r"""Check if the item is the right type"""
        if not isinstance(item, FPVecTransformer):
            raise ValueError("FPVecTransformer allowed only, provided {}".format(type(item)))

    def get_collate_fn(self, *args, **kwargs):
        """
        Get collate function of this featurizer. The FeatConcat featurizer use the default
        collate function which does not do anything.

        Returns:
            fn: Collate function for pytorch or None
        """
        return getattr(self, "collate_fn", None)

    def iter_index(self, indexes: Union[int, Iterator[int]]):
        r"""
        Allow the `FeatConcat` to be indexed using a list, or any other iterable.

        Args:
            indexes: The indexes to index the ``FeatConcat``.

        Returns
            indexed_fpconcat: A new FeatConcat object with the indexed element
        """
        if not isinstance(indexes, (list, tuple)):
            try:
                indexes = list(indexes)
            except TypeError:
                indexes = [indexes]
        return FeatConcat([self[ii] for ii in indexes])

    @property
    def columns(self):
        """Get the list of columns for the concatenated molecules

        Returns:
            columns (list): Name of the columns of the descriptor
        """
        tmp_mol = dm.to_mol("CC(C)O")
        columns = []
        for fp in self:
            fp_columns = getattr(fp, "columns", None)
            fp_name = str(fp)
            if fp_columns is None:
                fp_out, _ = fp([tmp_mol])
                fp_out = np.asarray(fp_out)
                fp_columns = [f"{fp_name}:{ind}" for ind in range(fp_out.shape[-1])]
            columns.extend(fp_columns)
        return columns

    def transform(self, mols: List[Union[dm.Mol, str]], **kwargs):
        r"""
        Calls the ``FPVecTransformer.transform`` for each transformer in
        the current list, and concatenates the resulting fingerprints.

        Args:
            mols: List of SMILES or molecules
            kwargs: named parameters for transform (see below)

        Returns:
            fps: Computed fingerprints of size NxD, where D is the
                sum of the length of each transformer and N is the number of input
                molecules that have been successfully featurized.
        """

        fps = []
        for _, fp_trans in enumerate(self):
            out = fp_trans.transform(mols, enforce_dtype=False, **kwargs)
            out = datatype.cast(out, dtype="pandas")
            fps.append(out)
        fps = pd.concat(fps, axis=1)
        fps.columns = self.columns
        return fps.values

    def __call__(
        self,
        mols: List[Union[dm.Mol, str]],
        enforce_dtype: bool = False,
        ignore_errors: bool = False,
        **kwargs,
    ):
        r"""
        Calls each of the internal transformer and concatenate results only on valid indices.

        Args:
            mols:  Mol or SMILES of the molecules to be transformed
            enforce_dtype: whether to enforce the instance dtype in the generated fingerprint
            ignore_errors: Whether to ignore errors during featurization or raise an error.
            kwargs: Named parameters for the transform method

        Returns:

            fp: array
                computed fingerprints of size NxD, where D is the
                sum of the length of each transformer and N is the number of input
                molecules that have been successfully featurized.
            idx: array
                Indices of successful featurization given the original molecule input.
        """

        fps = []
        valid_idx = np.zeros(len(mols))
        for _, transf in enumerate(self):
            fp = transf.transform(mols, ignore_errors=ignore_errors, **kwargs)
            fp, idx = transf._filter_none(fp)
            fps.append(fp)
            valid_idx[idx] += 1  # increase counter of valids
        valid_idx = np.nonzero(valid_idx == len(self))[0]
        fps = np.concatenate(fps, axis=1)
        if self.dtype is not None and enforce_dtype:
            fps = datatype.cast(fps, dtype=self.dtype, columns=self.columns)
        if not ignore_errors:
            return fps
        return fps, list(valid_idx)

    def fit_transform(
        self,
        mols: List[Union[str, dm.Mol]],
        y: Optional[Iterable] = None,
        fit_kwargs: Dict = None,
        trans_kwargs: Dict = None,
    ):
        r"""
        Calls the ``self.fit`` followed by the ``fit.transform`` for each transfomer in
        the current list, and concatenates the resulting fingerprints.

        Args:
            mols: List of SMILES or molecules
            y: target for the fitting. Usually ignored for FPVecTransformer
            fit_kwargs:  named parameters for fit
            fit_kwargs:named parameters for transform

        Returns:

            fp: computed fingerprints of size NxD, where D is the
                sum of the length of each transformer and N is the number of input
                molecules that have been successfully featurized.
        """
        fit_kwargs = {} if fit_kwargs is None else fit_kwargs
        trans_kwargs = {} if trans_kwargs is None else trans_kwargs
        self.fit(mols, y=y, **fit_kwargs)
        return self.transform(mols, **trans_kwargs)

    def fit(self, X: List[Union[dm.Mol, str]], y=None, **kwargs):
        r"""
        Calls the ``FPVecTransformer.fit`` for each transformer in the current list.

        Args:
            X: input list of molecules
            y (list, optional): Optional list of molecular properties. Defaults to None.

        Returns:
            self: FeatConcat instance after fitting
        """

        for _, fp_trans in enumerate(self):
            fp_trans.fit(X, y=y, **kwargs)
        return self
