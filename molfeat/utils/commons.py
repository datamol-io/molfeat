"""Common utility functions"""

from typing import Type
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import List
from typing import Union

import types
import os
import inspect
import hashlib
import pickle
import functools
import warnings
import torch
import numpy as np
import datamol as dm
import fsspec

from joblib import wrap_non_picklable_objects

from rdkit.Chem import SaltRemover
from molfeat.utils import datatype

FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)


def is_callable(func):
    r"""
    Check if func is a function or a callable
    """
    return func and (isinstance(func, FUNCTYPES) or callable(func))


def sha256sum(filepath: Union[str, os.PathLike]):
    """Return the sha256 sum hash of a file or a directory

    Args:
        filepath: The path to the file to compute the MD5 hash on.
    """
    if dm.fs.is_dir(filepath):
        files = list(dm.fs.glob(os.path.join(filepath, "**", "*")))
    else:
        files = [filepath]
    file_hash = hashlib.sha256()
    for filepath in sorted(files):
        with fsspec.open(filepath) as f:
            file_hash.update(f.read())  # type: ignore
    file_hash = file_hash.hexdigest()
    return file_hash


def get_class_name(cls: Type):
    """Get class full name

    Args:
        cls: name of the class
    """
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


def _clean_mol_for_descriptors(
    mol, disconnect_metals: bool = False, remove_salt: bool = False, **kwargs
):
    """Clean molecule for descriptors computation
    Args:
        disconnect_metals: whether to disconnect or keep metal ions
        remove_salt: whether to remove salt or
    """
    mol = dm.to_mol(mol)
    mol = dm.standardize_mol(mol, disconnect_metals=disconnect_metals, **kwargs)
    if remove_salt:
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
    return mol


def ensure_picklable(fn: Callable):
    """Ensure a function is picklable

    Args:
        fn: function to be pickled
    """
    if inspect.isfunction(fn) and fn.__name__ == "<lambda>":
        return wrap_non_picklable_objects(fn)
    return fn


def fn_to_hex(fn):
    """Pickle an object and return its hex representation

    Args:
        fn: object to pickle

    Returns:
        str: hex representation of object
    """
    bytes_str = pickle.dumps(ensure_picklable(fn))
    return bytes_str.hex()


def hex_to_fn(hex: str):
    """Load a hex string as a callable. Raise error on fail

    Args:
        hex: hex string to load as a callable

    Returns:
        callable: callable loaded from the hex string
    """
    # EN: pickling with pickle is probably faster
    fn = pickle.loads(bytes.fromhex(hex))
    return fn


def one_hot_encoding(
    val: int,
    allowable_set: Iterable,
    encode_unknown: bool = False,
    dtype: Callable = int,
):
    r"""Converts a single value to a one-hot vector.

    Args:
        val: class to be converted into a one hot vector
        allowable_set: a list or 1D array of allowed choices for val to take
        dtype: data type of the the return. Default = int.
        encode_unknown: whether to map inputs not in allowable set to an additional last element.

    Returns:
        A numpy 1D array of length len(allowable_set) + 1
    """

    encoding = np.zeros(len(allowable_set) + int(encode_unknown), dtype=dtype)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(allowable_set):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0 and encode_unknown:  # aka not found
        encoding[-1] = 1
    return encoding


def _parse_to_evaluable_str(val: Callable):
    r"""
    Generates a string from an object, such that it can be evaluated.
    It only works with basic classes, or classes that implement a
    specific `__repr__()` method.

    Args:
        val: input object or function to repr

    Returns:
        val_str (str): String representation of the function
    """
    if inspect.isclass(val):
        if datatype.is_dtype_numpy(val):
            val_str = "np." + val.__name__
        elif datatype.is_dtype_tensor(val):
            val_str = str(val)
        else:
            val_str = val.__name__
    elif isinstance(val, str):
        val_str = '"{}"'.format(val)
    else:
        val_str = val.__repr__()
    return val_str


def filter_arguments(fn: Callable, params: dict):
    """Filter the argument of a function to only retain the valid ones

    Args:
        fn: Function for which arguments will be checked
        params: key-val dictionary of arguments to pass to the input function

    Returns:
        params_filtered (dict): dict of filtered arguments for the function
    """
    accepted_dict = inspect.signature(fn).parameters
    accepted_list = []
    for key in accepted_dict.keys():
        param = str(accepted_dict[key])
        if param[0] != "*":
            accepted_list.append(param)
    params_filtered = {key: params[key] for key in list(set(accepted_list) & set(params.keys()))}
    return params_filtered


def fold_count_fp(fp: Iterable, dim: int = 2**10, binary: bool = False):
    """Fold an RDKit fingerprint with Datamol's canonical implementation.

    Args:
        fp: iterable fingerprint
        dim: dimension of the folded array if not provided. Defaults to 2**10.
        binary: whether to fold into a binary array or take use a count vector

    Returns:
        folded: returns folded array to the provided dimension
    """
    warnings.warn(
        "molfeat.utils.commons.fold_count_fp is deprecated; use datamol.fold_count_fp instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Keep Molfeat's historical floating-point output while centralizing the
    # molecular primitive in Datamol.
    return dm.fold_count_fp(fp, dim=dim, binary=binary).astype(float, copy=False)


def requires_conformer(calculator: Callable):
    """Decorator for any descriptor calculator that requires conformers"""

    # this is a method or __call__
    if inspect.getfullargspec(calculator).args[0] == "self":

        @functools.wraps(calculator)
        def calculator_wrapper(ref, mol, *args, **kwargs):
            mol = dm.to_mol(mol)
            if mol.GetNumConformers() < 1:
                raise ValueError("Expected a molecule with conformers information.")
            return calculator(ref, mol, *args, **kwargs)

    else:

        @functools.wraps(calculator)
        def calculator_wrapper(mol, *args, **kwargs):
            mol = dm.to_mol(mol)
            if mol.GetNumConformers() < 1:
                raise ValueError("Expected a molecule with conformers information.")
            return calculator(mol, *args, **kwargs)

    return calculator_wrapper


def requires_standardization(
    calculator: Callable = None,
    *,
    disconnect_metals: bool = True,
    remove_salt: bool = True,
    **standardize_kwargs,
):
    """Decorator for any descriptor calculator that required standardization of the molecules
    Args:
        calculator: calculator to wrap
        disconnect_metals: whether to force metal disconnection
        remove_salt: whether to remove salt from the molecule
    """

    def _standardize_mol(calculator):
        @functools.wraps(calculator)
        def wrapped_function(mol, *args, **kwargs):
            mol = _clean_mol_for_descriptors(
                mol,
                disconnect_metals=disconnect_metals,
                remove_salt=remove_salt,
                **standardize_kwargs,
            )
            return calculator(mol, *args, **kwargs)

        @functools.wraps(calculator)
        def class_wrapped_function(ref, mol, *args, **kwargs):
            if not getattr(ref, "do_not_standardize", False):
                mol = _clean_mol_for_descriptors(
                    mol,
                    disconnect_metals=disconnect_metals,
                    remove_salt=remove_salt,
                    **standardize_kwargs,
                )
            return calculator(ref, mol, *args, **kwargs)

        if inspect.getfullargspec(calculator).args[0] == "self":
            return class_wrapped_function
        return wrapped_function

    if calculator is not None:
        return _standardize_mol(calculator)
    return _standardize_mol


def concat_dict(prop_dict: dict, new_name: str, order: Optional[Iterable[str]] = None):
    """Concat properties in dict into a single key dict

    Args:
        prop_dict (dict): Input dict of property names and their computed values
        new_name (str): new name under which the concatenated property dict will be returned
        order: Optional list of key that specifies the order in which concatenation should be done. Sorting list by default

    Returns:
        dict: dictionary of concatenated output values with a single key corresponding to new_name
    """
    if not order:
        order = list(sorted(prop_dict.keys()))

    if len(order) > 0:
        concatenated_val = np.concatenate([prop_dict[x] for x in order], axis=1)
        output_dict = {new_name: concatenated_val}
    return output_dict


def pack_graph(
    batch_G: List[torch.FloatTensor],
    batch_x: List[torch.FloatTensor],
):
    r"""
    Pack a batch of graph and atom features into a single graph

    Args:
        batch_G: List of adjacency graph, each of size (n_i, n_i).
        batch_x: List of atom feature matrices, each of size (n_i, F), F being the number of features

    Returns:
        new_batch_G, new_batch_x: torch.LongTensor 2D, torch.Tensor 2D
            This tuple represents a new arbitrary graph that contains the whole batch,
            and the corresponding atom feature matrix. new_batch_G has a size (N, N), with :math:`N = \sum_i n_i`,
            while new_batch_x has size (N,D)
    """

    new_batch_x = torch.cat(tuple(batch_x), dim=0)
    n_neigb = new_batch_x.shape[0]
    # should be on the same device
    new_batch_G = batch_G[0].new_zeros((n_neigb, n_neigb))
    cur_ind = 0
    for g in batch_G:
        g_size = g.shape[0] + cur_ind
        new_batch_G[cur_ind:g_size, cur_ind:g_size] = g
        cur_ind = g_size
    return new_batch_G, new_batch_x


def pack_bits(obj, protocol=4):
    """Pack an object into a bits representation

    Args:
        obj: object to pack

    Returns:
        bytes: byte-packed version of object
    """
    return pickle.dumps(obj, protocol=protocol)


def unpack_bits(bvalues):
    """Pack an object into a bits representation

    Args:
        bvalues: bytes to be unpacked

    Returns:
        obj: object that was packed
    """
    return pickle.loads(bvalues)


def align_conformers(
    mols: List[dm.Mol],
    ref_id: int = 0,
    copy: bool = True,
    conformer_id: int = -1,
):
    """Align molecules with Datamol's canonical Crippen O3A implementation.

    Args:
        mols: List of molecules to align. All the molecules must have a conformer.
        ref_id: Index of the reference molecule. By default, the first molecule in the list
            will be used as reference.
        copy: Whether to copy the molecules before performing the alignement.
        conformer_id: Conformer id to use.

    Returns:
        mols: The aligned molecules.
        scores: The score of the alignement.
    """

    warnings.warn(
        "molfeat.utils.commons.align_conformers is deprecated; "
        "use datamol.conformers.align_conformers instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dm.conformers.align_conformers(
        mols,
        ref_id=ref_id,
        copy=copy,
        conformer_id=conformer_id,
        backend="crippenO3A",
    )
