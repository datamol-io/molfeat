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
import torch
import numpy as np
import datamol as dm
import fsspec

from joblib import wrap_non_picklable_objects
from scipy.sparse import coo_matrix

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolAlign
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
    for filepath in files:
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
    """Fast folding of a count fingerprint to the specified dimension

    Args:
        fp: iterable fingerprint
        dim: dimension of the folded array if not provided. Defaults to 2**10.
        binary: whether to fold into a binary array or take use a count vector

    Returns:
        folded: returns folded array to the provided dimension
    """
    if hasattr(fp, "GetNonzeroElements"):
        tmp = fp.GetNonzeroElements()
    elif hasattr(fp, "GetOnBits"):
        # try to get the dict of onbit
        on_bits = fp.GetOnBits()
        tmp = dict(zip(on_bits, np.ones(len(on_bits))))
    else:
        raise ValueError(f"Format {type(fp)} is not supported")
    out = (
        coo_matrix(
            (
                list(tmp.values()),
                (np.repeat(0, len(tmp)), [i % dim for i in tmp.keys()]),
            ),
            shape=(1, dim),
        )
        .toarray()
        .flatten()
    )
    if binary:
        out = np.clip(out, a_min=0, a_max=1)
    return out


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
    """Align a list of molecules to a reference molecule.

    Note: consider adding me to `datamol`.

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

    # Check all input molecules has a conformer
    if not all([mol.GetNumConformers() >= 1 for mol in mols]):
        raise ValueError("One or more input molecules is missing a conformer.")

    # Make a copy of the molecules since they are going to be modified
    if copy:
        mols = [dm.copy_mol(mol) for mol in mols]

    # Compute Crippen contributions for every atoms and molecules
    crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in mols]

    # Split reference and probe molecules
    crippen_contrib_ref = crippen_contribs[ref_id]
    crippen_contrib_probes = crippen_contribs
    mol_ref = mols[ref_id]
    mol_probes = mols

    # Loop and align
    scores = []
    for i, mol in enumerate(mol_probes):
        crippenO3A = rdMolAlign.GetCrippenO3A(
            prbMol=mol,
            refMol=mol_ref,
            prbCrippenContribs=crippen_contrib_probes[i],
            refCrippenContribs=crippen_contrib_ref,
            prbCid=conformer_id,
            refCid=conformer_id,
            maxIters=50,
        )
        crippenO3A.Align()

        scores.append(crippenO3A.Score())

    scores = np.array(scores)

    return mols, scores
