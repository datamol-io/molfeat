from typing import Optional
from typing import List
from typing import Any


import torch
import numpy as np
import datamol as dm

from rdkit.Chem import rdchem
from rdkit.Chem import AllChem
from rdkit.Chem import GetSymmSSSR
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from molfeat.utils.commons import requires_conformer
from molfeat.data import get_df as get_data
from molfeat.utils.commons import one_hot_encoding


ATOM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]
DGLLIFE_WEAVE_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

ATOM_DEGREE_LIST = list(range(11))
ATOM_TOTAL_DEGREE_LIST = list(range(6))
EXPLICIT_VALENCE_LIST = list(range(1, 7))
IMPLICIT_VALENCE_LIST = list(range(7))
# EN: (see molfeat/issues/141) changed from enumeration from rdkit.Hybridization
# to stating the list of hybridation because of RDKit introducing
# SP2D in  '2022.09.1'
HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.UNSPECIFIED,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.S,
]


DGLLIFE_HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
ATOM_NUM_H_LIST = [0, 1, 2, 3, 4]
CHARGE_LIST = [-3, -2, -1, 0, 1, 2, 3]
RADICAL_ELECTRON_LIST = list(range(5))
CHIRAL_TYPES = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]

DGLLIFE_WEAVE_CHIRAL_TYPES = [
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]


BOND_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

BOND_DIRECTION = [
    rdchem.BondDir.NONE,
    rdchem.BondDir.ENDUPRIGHT,
    rdchem.BondDir.ENDDOWNRIGHT,
]


def atom_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
    number: bool = False,
):
    """One hot encoding for the type of an atom.

    Args:
        atom : RDKit atom instance.
        allowable_set : Atom types to consider. Default is the full periodic table
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)
        number: Whether to use the atomic number of element symbol

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = ATOM_LIST if not number else list(range(1, 101))
    atom_val = atom.GetSymbol() if not number else atom.GetAtomicNum()
    return one_hot_encoding(atom_val, allowable_set, encode_unknown)


def atom_number(atom: dm.Atom):
    """Get the atomic number for an atom.

    Args:
        atom : RDKit atom instance.

    Returns:
        list: List containing one int only.

    """
    return [atom.GetAtomicNum()]


def atom_degree_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the degree of an atom.

    Result might be different if hydrogen have been added or not.

    Args:
        atom: RDKit atom instance.
        allowable_set : Atom degrees to consider. Default: ATOM_DEGREE_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.

    """
    if allowable_set is None:
        allowable_set = ATOM_DEGREE_LIST
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)


def atom_degree(
    atom: dm.Atom,
):
    """Get the degree of an atom.

    Result might be different if hydrogen have been added or not.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetDegree()]


def atom_total_degree_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the total degree of an atom including Hs.

    Args:
        atom: RDKit atom instance.
        allowable_set : Atom total degrees to consider. Default: ATOM_TOTAL_DEGREE_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.

    """
    if allowable_set is None:
        allowable_set = ATOM_TOTAL_DEGREE_LIST
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)


def atom_total_degree(atom: dm.Atom):
    """Get the total degree of an atom.

    Result might be different if hydrogen have been added or not.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetTotalDegree()]


def atom_explicit_valence_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the explicit valence of an aotm.

    Args:
        atom: RDKit atom instance.
        allowable_set : Atom explicit valence to consider. Default: EXPLICIT_VALENCE_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = EXPLICIT_VALENCE_LIST
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)


def atom_explicit_valence(atom: dm.Atom):
    """Get the explicit valence of an atom.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetExplicitValence()]


def atom_implicit_valence_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the implicit valence of an aotm.

    Args:
        atom: RDKit atom instance.
        allowable_set : Atom implicit valence to consider. Default: IMPLICIT_VALENCE_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = IMPLICIT_VALENCE_LIST
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)


def atom_implicit_valence(atom: dm.Atom):
    """Get the implicit valence of an atom.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetImplicitValence()]


def atom_hybridization_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the hybridization of an atom.

    Args:
        atom: RDKit atom instance.
        allowable_set : Atom hybridization state to consider. Default: HYBRIDIZATION_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = HYBRIDIZATION_LIST
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_total_num_H_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the total number of Hs of an atom.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom total hydrogen to consider. Default: ATOM_NUM_H_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = ATOM_NUM_H_LIST
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)


def atom_total_num_H(atom: dm.Atom):
    """Get the total number of Hs of an atom.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetTotalNumHs()]


def atom_formal_charge_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the formal charge of an atom.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom formal charge to consider. Default: HYBRIDIZATION_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = CHARGE_LIST
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)


def atom_formal_charge(atom: dm.Atom):
    """Get formal charge for an atom.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetFormalCharge()]


def atom_partial_charge(atom: dm.Atom):
    """Get Gasteiger partial charge for an atom.

    This function requires calling ``AllChem.ComputeGasteigerCharges(mol)`` first to compute Gasteiger charges.
    Nan and infinite values are set to 0

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one float only.
    """
    if not atom.HasProp("_GasteigerCharge"):
        mol = atom.GetOwningMol()
        AllChem.ComputeGasteigerCharges(mol)
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    gasteiger_charge = np.nan_to_num(float(gasteiger_charge), posinf=0.0, neginf=0.0)
    return [float(gasteiger_charge)]


def atom_num_radical_electrons_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the number of radical electrons of an atom.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom num radical electron to consider. Default: RADICAL_ELECTRON_LIST
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = RADICAL_ELECTRON_LIST
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)


def atom_num_radical_electrons(atom: dm.Atom):
    """Get the number of radical electrons for an atom.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.GetNumRadicalElectrons()]


def atom_is_aromatic_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for whether the atom is aromatic.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom aromatic state to consider. Default: [True, False]
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)


def atom_is_aromatic(atom: dm.Atom):
    """Get whether the atom is aromatic.


    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one bool only.
    """
    return [atom.GetIsAromatic()]


def atom_is_in_ring_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for whether the atom is in ring.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom ring state to consider. Default: [True, False]
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)


def atom_is_in_ring(atom: dm.Atom):
    """Get whether the atom is in ring.

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one int only.
    """
    return [atom.IsInRing()]


def atom_chiral_tag_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the chiral tag of an atom.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom chiral tag to consider. Default: CHIRAL_TYPES
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    list
        list: List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = CHIRAL_TYPES
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)


def atom_chirality_type_one_hot(
    atom: dm.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the chirality type of an atom.

     Args:
        atom: RDKit atom instance.
        allowable_set : Atom chirality to consider. Default: ["R", "S"]
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    list
        list: List of boolean values where at most one value is True.
    """

    cip_code = None
    if atom.HasProp("_CIPCode"):
        cip_code = atom.GetProp("_CIPCode")
    if allowable_set is None:
        allowable_set = ["R", "S"]
    return one_hot_encoding(cip_code, allowable_set, encode_unknown)


def atom_mass(atom: dm.Atom, coeff: float = 1):
    """Get the mass of an atom and scale it.

    Args:
        atom: RDKit atom instance.
        coeff: scaling factor for the atom mass
    Returns:
        list: List containing one float only.
    """
    return [atom.GetMass() * coeff]


def atom_is_chiral_center(atom: dm.Atom):
    """Get whether the atom is chiral center

    Args:
        atom: RDKit atom instance.
    Returns:
        list: List containing one bool only.
    """
    return [atom.HasProp("_ChiralityPossible")]


def atom_extended_properties(atom: dm.Atom, dataset: str = "origin"):
    """Get a full set of atom descriptors

    Args:
        atom: RDKit atom instance.
        dataset: which dataset to use to query the atom descriptor.
            One of 'origin', 'elements', 'elements_completed'. Default to "origin".
    Returns:
        list: List containing one float only.
    """
    if dataset == "origin":
        data = get_data("origin")
    elif dataset in ["elements", "element"]:
        data = get_data("elements")
    else:
        data = get_data("elements_completed")

    feat_shape = data.shape[-1]
    feat = [0 for _ in range(feat_shape)]
    if atom.GetSymbol() in list(data.index):
        feat = list(data.loc[atom.GetSymbol()])
    return feat


def bond_type_one_hot(
    bond: dm.Bond,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the type of a bond.

    Args:
        bond: RDKit atom instance.
        allowable_set : Bond type to consider. Default: BOND_TYPES
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: ist of boolean values where at most one value is True.

    """
    if allowable_set is None:
        allowable_set = BOND_TYPES
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)


def bond_is_conjugated_one_hot(
    bond: dm.Bond,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for whether the bond is conjugated.

    Args:
        bond: RDKit bond instance.
        allowable_set : Bond type to consider. Default: BOND_TYPES
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: list of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)


def bond_is_conjugated(bond: dm.Bond):
    """Get whether the bond is conjugated.

    Args:
        bond: RDKit bond instance.
    Returns:
        list: List containing one bool only.
    """
    return [bond.GetIsConjugated()]


def bond_is_in_ring_one_hot(
    bond: dm.Bond,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for whether the bond is in a ring of any size.

    Args:
        bond: RDKit bond instance.
        allowable_set : Bond type to consider. Default: [True, False]
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: list of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)


def bond_is_in_ring(bond: dm.Bond):
    """Get whether the bond is in ring.

    Args:
        bond: RDKit bond instance.
    Returns:
        list: List containing one bool only.
    """
    return [bond.IsInRing()]


def bond_stereo_one_hot(
    bond: dm.Bond,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """One hot encoding for the stereo configuration of a bond.

    Args:
        bond: RDKit bond instance.
        allowable_set : Bond type to consider. Default: BOND_STEREO
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: list of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = BOND_STEREO
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)


def bond_direction_one_hot(
    bond: dm.Bond,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
):
    """Get a one hot encoding for the direction of a bond.

    Args:
        bond: RDKit bond instance.
        allowable_set : Bond type to consider. Default: BOND_DIRECTION
        encode_unknown: whether to map inputs not in the allowable set to the additional last element. (Default: False)

    Returns:
        list: List of boolean values where at most one value is True.

    """
    if allowable_set is None:
        allowable_set = BOND_DIRECTION
    return one_hot_encoding(bond.GetBondDir(), allowable_set, encode_unknown)


def pairwise_2D_dist(mol: dm.Mol):
    """Compute the pairwise distance between all pairs of atoms in 2D

    Args:
        mol: Input molecule

    Returns:
        dist: Matrix of size V^2, 1
    """
    mat = GetDistanceMatrix(mol)
    return mat.reshape(-1, 1)


@requires_conformer
def pairwise_3D_dist(mol: dm.Mol, conformer_id: int = -1):
    """Compute the pairwise 3D distance between all pair of atoms

    Args:
        mol: Input molecule
        conformer_id: conformer id to use

    Returns:
        dist: Matrix of size V^2, 1
    """
    mat = Get3DDistanceMatrix(mol, confId=conformer_id)
    return mat.reshape(-1, 1)


def pairwise_dist_indicator(mol: dm.Mol, max_distance: int = 7):
    """Compute the pairwise distance matrix gated by a max distance for the weave featurizer

    Args:
        mol: input molecule
        max_distance: maximum distance to use for the thresholding
    """
    dist_mat = torch.from_numpy(pairwise_2D_dist(mol))
    # Elementwise compare if distance is bigger than 0, 1, ..., max_distance - 1
    dist_indicator = (dist_mat > torch.arange(0, max_distance).float()).float()
    return dist_indicator.numpy()


def pairwise_bond_indicator(mol: dm.Mol, allowable_set: Optional[List[Any]] = None):
    """Compute the pairwise bond indicator for weave net

    Args:
        mol: input molecule
        allowable_set: bond type to consider
    """
    num_atoms = mol.GetNumAtoms()
    if allowable_set is None:
        allowable_set = BOND_TYPES
    bond_indicators = torch.zeros(num_atoms, num_atoms, len(allowable_set))
    for bond in mol.GetBonds():
        bond_type_encoding = torch.tensor(
            bond_type_one_hot(bond, allowable_set=allowable_set)
        ).float()
        begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_indicators[begin_atom_idx, end_atom_idx] = bond_type_encoding
        bond_indicators[end_atom_idx, begin_atom_idx] = bond_type_encoding
    # Reshape from (V, V, num_bond_types) to (V^2, num_bond_types)
    bond_indicators = bond_indicators.reshape(-1, len(allowable_set))
    return bond_indicators.numpy()


def pairwise_ring_membership(mol: dm.Mol):
    """Compute the joint ring membership of all atom pairs

    Args:
        mol: Input molecule

    Returns:
        ring_membership: Matrix of size V^2, 1
    """
    num_atoms = mol.GetNumAtoms()
    sssr = GetSymmSSSR(mol)
    ring_mate = np.zeros((num_atoms, num_atoms, 1))
    for ring in sssr:
        ring = np.asarray(ring)
        num_atoms_in_ring = len(ring)
        for i in range(num_atoms_in_ring):
            ring_mate[ring[i], ring] = 1
    ring_mate = ring_mate.reshape(-1, 1)
    return ring_mate
