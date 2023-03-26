from typing import Callable
from typing import Union
from typing import Optional
from typing import Iterable

import torch
import numpy as np
import datamol as dm
import pandas as pd

from rdkit.DataStructs.cDataStructs import ConvertToExplicit
from rdkit.DataStructs.cDataStructs import SparseBitVect
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.DataStructs.cDataStructs import CreateFromBitString


def ensure_explicit(x: Union[SparseBitVect, ExplicitBitVect]):
    """Ensure that the input vector is not a sparse bit vector

    Args:
        x: input vector

    Returns:
        converted: ExplicitBitVect if input is SparseBitVec, else input as is
    """
    if isinstance(x, SparseBitVect):
        x = ConvertToExplicit(x)
    return x


def to_sparse(x, dtype=None):
    r"""
    Converts dense tensor x to sparse format

    Args:
        x (torch.Tensor): tensor to convert
        dtype (torch.dtype, optional): Enforces new data type for the output.
            If None, it keeps the same datatype as x (Default: None)
    Returns:
        new torch.sparse Tensor
    """

    if dtype is not None:
        x = x.type(dtype)

    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def to_tensor(x, gpu=False, dtype=None):
    r"""
    Convert a numpy array to tensor. The tensor type will be
    the same as the original array, unless specify otherwise

    Args:
        x (numpy.ndarray): Numpy array to convert to tensor type
        gpu (bool optional): Whether to move tensor to gpu. Default False
        dtype (torch.dtype, optional): Enforces new data type for the output

    Returns:
        New torch.Tensor
    """
    if not torch.is_tensor(x):
        try:
            if torch.is_tensor(x[0]):
                x = torch.stack(x)
        except:
            pass
        x = torch.as_tensor(x)
    if dtype is not None:
        x = x.to(dtype=dtype)
    if gpu and torch.cuda.is_available():
        x = x.cuda()
    return x


def to_numpy(x, copy=False, dtype=None):
    r"""
    Convert a tensor to numpy array.

    Args:
        x (Object): The Python object to convert.
        copy (bool, optional): Whether to copy the memory.
            By default, if a tensor is already on CPU, the
            Numpy array will be a view of the tensor.
        dtype (callable, optional): Optional type to cast the values to

    Returns:
        A new Python object with the same structure as `x` but where the tensors are now Numpy
        arrays. Not supported type are left as reference in the new object.
    """
    if isinstance(x, (list, tuple, np.ndarray)) and torch.is_tensor(x[0]):
        x = [to_numpy(xi, copy=copy, dtype=dtype) for xi in x]
    if isinstance(x, np.ndarray):
        pass
    elif torch.is_tensor(x):
        x = x.cpu().detach().numpy()
        x = x.copy()
    elif isinstance(x, SparseBitVect):
        tmp = np.zeros(x.GetNumBits(), dtype=int)
        for n_bit in list(x.GetOnBits()):
            tmp[n_bit] = 1
        x = tmp
    elif isinstance(x, ExplicitBitVect):
        x = dm.fp_to_array(x)
    elif hasattr(x, "GetNonzeroElements"):
        # one of the other rdkit type
        tmp = np.zeros(x.GetLength())
        bit_idx, values = np.array(list(x.GetNonzeroElements().items())).T
        tmp[bit_idx] = values
        x = tmp
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def to_fp(arr: np.ndarray, bitvect: bool = True, sparse: bool = False):
    """Convert numpy array to fingerprint

    Args:
        arr: Numpy array to convert to bitvec
        bitvect: whether to assume the data is a bitvect or intvect
        sparse: whether to convert to sparse bit vect

    Returns:
        fp: RDKit bit vector
    """
    if not isinstance(arr, list) and arr.ndim > 1:
        raise ValueError("Expect a 1D array as input !")
    if not bitvect:
        fp = UIntSparseIntVect(len(arr))
        for ix, value in enumerate(arr):
            fp[ix] = int(value)
    elif sparse:
        onbits = np.where(arr == 1)[0].tolist()
        fp = SparseBitVect(arr.shape[0])
        fp.SetBitsFromList(onbits)
    else:
        arr = np.asarray(arr)
        bitstring = "".join(arr.astype(str))
        fp = CreateFromBitString(bitstring)
    return fp


def is_dtype_tensor(dtype):
    r"""
    Verify if the dtype is a torch dtype

    Args:
        dtype (callable): The dtype of a value. E.g. np.int32, str, torch.float

    Returns:
        A boolean saying if the dtype is a torch dtype
    """
    return isinstance(dtype, torch.dtype) or (dtype == torch.Tensor)


def is_dtype_bitvect(dtype):
    """
    Verify if the dtype is a bitvect type

    Args:
        dtype (callable): The dtype of a value. E.g. np.int32, str, torch.float

    Returns:
        A boolean saying if the dtype is a torch dtype
    """
    return dtype in [ExplicitBitVect, SparseBitVect] or isinstance(
        dtype, (ExplicitBitVect, SparseBitVect)
    )


def is_dtype_numpy(dtype):
    r"""
    Verify if the dtype is a numpy dtype

    Args:
        dtype (callable): The dtype of a value. E.g. np.int32, str, torch.float
    Returns
        A boolean saying if the dtype is a numpy dtype
    """
    is_torch = is_dtype_tensor(dtype)
    is_num = dtype in (int, float, complex)
    if hasattr(dtype, "__module__"):
        is_numpy = dtype.__module__ == "numpy"
    else:
        is_numpy = False
    return (is_num or is_numpy) and not is_torch


def is_null(obj):
    """Check if an obj is null (nan, None or array of nan)"""
    array_nan = False
    all_none = False
    try:
        tmp = to_numpy(obj)
        array_nan = np.all(np.isnan(tmp))
    except:
        pass
    try:
        all_none = all(x is None for x in obj)
    except:
        pass
    return obj is None or all_none or array_nan


def cast(fp, dtype: Optional[Callable] = None, columns: Optional[Iterable] = None):
    """Change the datatype of a list of input array

    Args:
        fp (array): Input array to cast (2D)
        dtype: datatype to cast to
        columns: column names for pandas dataframe
    """
    if fp is None or dtype is None:
        return fp
    if isinstance(fp, dict):
        fp = {k: cast(v, dtype=dtype, columns=columns) for k, v in fp.items()}
    elif dtype in [tuple, list]:
        fp = list(fp)
    elif is_dtype_numpy(dtype):
        if isinstance(fp, (list, tuple)) and not np.isscalar(fp[0]):
            fp = [to_numpy(fp_i, dtype=dtype) for fp_i in fp]
            fp = to_numpy(fp, dtype=dtype)
        else:
            fp = to_numpy(fp, dtype=dtype)
    elif is_dtype_tensor(dtype):
        if isinstance(fp, (list, tuple)) and not np.isscalar(fp[0]):
            tmp_fp = to_numpy(fp[0])
            if len(tmp_fp.shape) > 1:
                fp = torch.cat([to_tensor(fp_i, dtype=dtype) for fp_i in fp])
            else:
                fp = torch.stack([to_tensor(fp_i, dtype=dtype) for fp_i in fp])
        else:
            fp = to_tensor(fp, dtype=dtype)
    elif dtype in [pd.DataFrame, "dataframe", "pandas", "df"]:
        fp = [feat if feat is not None else [] for feat in fp]
        fp = pd.DataFrame(fp)
        if columns is not None:
            fp.columns = columns
    elif is_dtype_bitvect(dtype):
        fp = [to_fp(feat, sparse=(dtype == SparseBitVect)) for feat in fp]
    else:
        raise TypeError("The type {} is not supported".format(dtype))
    return fp
