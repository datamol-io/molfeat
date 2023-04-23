from typing import Optional
from typing import Union
import torch
import numpy as np
from packaging import version
from molfeat.calc.atom import AtomCalculator
from molfeat.calc.atom import AtomMaterialCalculator
from molfeat.calc.atom import DGLWeaveAtomCalculator
from molfeat.calc.atom import DGLCanonicalAtomCalculator
from molfeat.calc.bond import BondCalculator
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.calc.bond import DGLCanonicalBondCalculator
from molfeat.calc.bond import DGLWeaveEdgeCalculator


DTYPES_MAPPING = {
    None: None,
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.int8: "torch.int8",
    torch.int16: "torch.int16",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
    np.float16: "np.float16",
    np.float32: "np.float32",
    np.float64: "np.float64",
    np.int8: "np.int8",
    np.int16: "np.int16",
    np.int32: "np.int32",
    np.int64: "np.int64",
    np.uint8: "np.uint8",
    np.uint16: "np.uint16",
    np.uint32: "np.uint32",
    np.uint64: "np.uint64",
    bool: "bool",
    int: "int",
    float: "float",
}

DTYPES_MAPPING_REVERSE = {v: k for k, v in DTYPES_MAPPING.items()}

ATOM_FEATURIZER_MAPPING = {
    AtomCalculator: "AtomCalculator",
    AtomMaterialCalculator: "AtomMaterialCalculator",
    DGLCanonicalAtomCalculator: "DGLCanonicalAtomCalculator",
    DGLWeaveAtomCalculator: "DGLWeaveAtomCalculator",
}

ATOM_FEATURIZER_MAPPING_REVERSE = {v: k for k, v in ATOM_FEATURIZER_MAPPING.items()}

BOND_FEATURIZER_MAPPING = {
    BondCalculator: "BondCalculator",
    EdgeMatCalculator: "EdgeMatCalculator",
    DGLCanonicalBondCalculator: "DGLCanonicalBondCalculator",
    DGLWeaveEdgeCalculator: "DGLWeaveEdgeCalculator",
}

BOND_FEATURIZER_MAPPING_REVERSE = {v: k for k, v in BOND_FEATURIZER_MAPPING.items()}


def map_dtype(dtype: Optional[Union[str, torch.dtype, np.dtype]]):
    """Map a dtype to a string representation or the other way around"""

    if isinstance(dtype, str):
        mapping = DTYPES_MAPPING_REVERSE
    else:
        mapping = DTYPES_MAPPING

    if dtype not in mapping:
        msg = f"{dtype} is not a valid dtype. The valid dtypes are {list(mapping.keys())}."
        raise ValueError(msg)

    return mapping[dtype]


def compare_state(
    state_dict_1, state_dict_2, allow_version_level: Optional[Union[str, int]] = None
):
    """Compare two state dict and allow version matching
    Args:
        state_dict_1: state dict of the first object
        state_dict_2: state dict of the second object
        allow_version_level: version level for comparison.
            One of 'major', 'minor', 'micro' or 0, 1, 2
    """
    state_dict_1 = state_dict_1.copy()
    state_dict_2 = state_dict_2.copy()
    version1 = state_dict_1.pop("_molfeat_version", "")
    version2 = state_dict_2.pop("_molfeat_version", "")
    try:
        version1 = version.parse(version1)
        release1 = version1.release
    except version.InvalidVersion:
        version1 = None
        release1 = None
    try:
        version2 = version.parse(version2)
        release2 = version2.release
    except version.InvalidVersion:
        version2 = None
        release2 = None

    version_names = ["major", "minor", "micro"]
    if isinstance(allow_version_level, str):
        allow_version_level = version_names.index(allow_version_level)
        # we want an exception if fail here

    if allow_version_level is None:
        version_comp = version1 == version2
        return (state_dict_1 == state_dict_2) and version_comp

    if release1 is None or release2 is None:
        version_comp = release1 == release2
    # pad both to the same length
    else:
        longest = len(release2) - len(release1)
        release1 += (0,) * max(0, longest)
        release2 += (0,) * max(0, -longest)
        version_comp = release1[: allow_version_level + 1] == release2[: allow_version_level + 1]
    return (state_dict_1 == state_dict_2) and version_comp
