from typing import Union
from typing import Optional

import itertools
import numpy as np
import pandas as pd
import datamol as dm

from collections import defaultdict as ddict
from pathlib import Path
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as Desc
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem.Scaffolds import MurckoScaffold
from molfeat.calc.base import SerializableCalculator

import molfeat


def _is_ring_fully_conjugated(mol: dm.Mol, ring: list):
    """Check whether a ring is fully conjugated"""
    suppl = Chem.ResonanceMolSupplier(mol)
    first_idx_conj = -1
    first = True
    for i in range(len(ring)):
        atom_idx = ring[i]
        atom_conj_grp_idx = Chem.ResonanceMolSupplier.GetAtomConjGrpIdx(suppl, atom_idx)
        if first:
            first_idx_conj = atom_conj_grp_idx
            if first_idx_conj > 99999 or first_idx_conj == -1:
                return False
            first = False
        else:
            if atom_conj_grp_idx != first_idx_conj:
                return False
    return True


def _n_multiple_bond_in_ring(mol: dm.Mol, ring: list):
    """Get number of multiple bonds in a ring"""
    atom_i = -1
    atom_j = -1
    bond = None
    nr_multiple_bonds = 0
    multiple_bond_types = [2, 3, 12]
    for i in range(len(ring)):
        atom_i = ring[i]
        for j in range(0, i):
            atom_j = ring[j]
            bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
            if bond is not None:
                bond_type = Chem.Bond.GetBondType(bond)
                if int(bond_type) in multiple_bond_types:
                    nr_multiple_bonds += 1
    return nr_multiple_bonds


def _count_heteroatom_per_ring(mol: dm.Mol, rings: list):
    """Count number of heteroatoms in rings"""
    n_heteros = [0] * len(rings)
    for i, ring in enumerate(rings):
        for atom in ring:
            is_hetero = int(mol.GetAtomWithIdx(atom).GetAtomicNum() not in [1, 6])
            n_heteros[i] += is_hetero
    return n_heteros


def _get_ring_system(mol: dm.Mol):
    """Build ring systems in a molecule"""
    q = mol.GetRingInfo()
    simple_rings = q.AtomRings()
    rings = [set(r) for r in simple_rings]
    ring_map = [set([x]) for x in range(len(rings))]
    go_next = True
    while go_next:
        go_next = False
        for i, j in itertools.combinations(range(len(rings)), 2):
            if rings[i] & rings[j]:
                new_map = set().union(ring_map[i], ring_map[j])
                q = rings[i] | rings[j]
                min_ind, max_ind = min(i, j), max(i, j)
                del rings[max_ind], rings[min_ind]
                del ring_map[max_ind], ring_map[min_ind]
                rings.append(q)
                ring_map.append(new_map)

                go_next = True
                break
    return list(simple_rings), rings, ring_map


def _ring_atom_state(mol: dm.Mol):
    """Get the conjugated state of ring atoms"""
    ri = mol.GetRingInfo()
    ring_atom_conj_state = ddict(list)
    for ring in ri.AtomRings():
        state = _is_ring_fully_conjugated(mol, ring)
        for atom in ring:
            ring_atom_conj_state[atom].append(state)
    return ring_atom_conj_state


class ScaffoldKeyCalculator(SerializableCalculator):
    """
    Implementation of the Scaffold Keys described in
    `Identification of Bioisosteric Scaffolds using Scaffold Keys` by Peter Ertl
    """

    DESCRIPTORS = [
        "n_atom_in_rings",
        "n_atom_in_conjugated_ring",
        "n_atoms_not_in_conjugated_ring",
        "n_atom_in_chain",
        "n_atom_exocyclic",
        "n_nitrogen",
        "n_nitrogen_in_ring",
        "n_oxygen",
        "n_oxygen_in_ring",
        "n_sulfur",
        "n_heteroatoms",
        "n_heteroatoms_in_ring",
        "n_atom_spiro_atoms",
        "n_heteroatom_more_than_2_conn",
        "n_carbon_atleast_2_heteroatoms",
        "n_atom_at_least_2_nei_more_than_2_conn",
        "abs_scaffold_format_charge",
        "n_bonds",
        "n_multiple_non_conj_ring_bonds",
        "n_bonds_2_heteroatoms",
        "n_carbon_het_carbon_het_bonds",
        "n_bonds_at_least_3_conn",
        "n_exocyclic_single_bonds_carbon",
        "n_exocyclic_single_bonds_nitrogen",
        "n_non_ring_bonds_2_conj_rings",
        "n_non_ring_bonds_conj_nonconj_rings",
        "n_bonds_atoms_with_at_least_one_nei_with_2_conn",
        "n_simple_rings",
        "size_largest_ring",
        "n_simple_rings_no_heteroatoms",
        "n_simple_rings_1_heteroatoms",
        "n_simple_rings_2_heteroatoms",
        "n_simple_rings_at_least_3_heteroatoms",
        "n_simple_non_conj_5_atoms_rings",
        "n_simple_non_conj_6_atoms_rings",
        "n_ring_system",
        "n_ring_system_with_2_non_conj_simple_ring",
        "n_ring_system_with_2_conj_simple_ring",
        "n_ring_system_with_conj_non_conj_simple_ring",
        "n_ring_system_with_3_conj_simple_ring",
        "n_ring_system_with_3_non_conj_simple_ring",
        "n_ring_system_with_greater_one_conj_nonconj_simple_ring",
    ]

    NORM_PARAMS = pd.read_csv(
        Path(molfeat.__file__).parents[0].joinpath("data/skey_parameters.csv"),
        index_col=0,
    ).loc[DESCRIPTORS]

    def __init__(
        self, normalize: bool = False, verbose: bool = False, use_scaffold: bool = False, **kwargs
    ):
        """
        Init of the scaffold key function

        Args:
            normalize: whether to normalize the value of the feature
            verbose: whether to log errors
            use_scaffold: whether to convert the molecule into scaffold first
        """
        self.normalize = normalize
        self.verbose = verbose
        self.use_scaffold = use_scaffold

    def __getstate__(self):
        """Get state of the scaffold key function"""
        state = {}
        state["normalize"] = self.normalize
        state["verbose"] = self.verbose
        state["use_scaffold"] = self.use_scaffold
        return state

    def __len__(self):
        return len(self.DESCRIPTORS)

    @classmethod
    def compute_normalization(cls, features: np.ndarray):
        """Normalize input features. The normalization parameters are
        computed by the scaffolds of 2.1M molecules from CHEMBL 29.
        """
        return (features - cls.NORM_PARAMS["mean"]) / cls.NORM_PARAMS["std"]

    def n_atom_in_rings(self, mol: dm.Mol):
        """1. number of ring atoms"""
        sm = dm.from_smarts("[r]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_atom_in_conjugated_ring(self, mol: dm.Mol):
        """2. number of atoms in conjugated rings"""
        ri = mol.GetRingInfo()
        n = 0
        for ring in ri.AtomRings():
            if _is_ring_fully_conjugated(mol, ring):
                n += len(ring)
        return n

    def n_atoms_not_in_conjugated_ring(self, mol: dm.Mol):
        """
        3. number of atoms not in conjugated rings
        (i.e. atoms in aliphatic rings and non-ring atoms)
        """
        # EN: replace conjugation by aromatic
        ri = mol.GetRingInfo()
        n = 0
        for ring in ri.AtomRings():
            if not _is_ring_fully_conjugated(mol, ring):
                n += len(ring)
        return n

    def n_atom_in_chain(self, mol: dm.Mol):
        """4. number atoms in chains (not counting double-connected exo-chain atoms)"""
        sm = dm.from_smarts("[!r;!$(*=[r])]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_atom_exocyclic(self, mol: dm.Mol):
        """5. number of exocyclic atoms (connected by multiple bonds to a ring)"""
        sm = dm.from_smarts("[!r;!$(*-[r])&$(*~[r])]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_nitrogen(self, mol: dm.Mol):
        """6. number of nitrogen"""
        sm = dm.from_smarts("[#7]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_nitrogen_in_ring(self, mol: dm.Mol):
        """7. number of nitrogen in rings"""
        sm = dm.from_smarts("[#7;r]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_oxygen(self, mol: dm.Mol):
        """8. number of oxygen"""
        sm = dm.from_smarts("[#8]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_oxygen_in_ring(self, mol: dm.Mol):
        """9. number of oxygen in rings"""
        sm = dm.from_smarts("[#8]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_sulfur(self, mol: dm.Mol):
        """10. number of sulfur atoms"""
        sm = dm.from_smarts("[#16]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_heteroatoms(self, mol: dm.Mol):
        """11. number of heteroatoms"""

        sm = dm.from_smarts("[!#1&!#6]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_heteroatoms_in_ring(self, mol: dm.Mol):
        """12. number of heteroatoms in rings"""
        sm = dm.from_smarts("[!#1&!#6&r]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_atom_spiro_atoms(self, mol: dm.Mol):
        """13. number of spiro atoms"""
        return Desc.CalcNumSpiroAtoms(mol)

    def n_heteroatom_more_than_2_conn(self, mol: dm.Mol):
        """14. number of heteroatoms with more than 2 connections"""
        sm = dm.from_smarts("[!#1;!#6;!D1!D0;!D2]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_carbon_atleast_2_heteroatoms(self, mol: dm.Mol):
        """15. number of carbon atoms connected to at least 2 heteroatoms"""
        n_atoms = 0
        for atom in mol.GetAtoms():
            tmp = [x for x in atom.GetNeighbors() if x.GetAtomicNum() not in [1, 6]]
            n_atoms += len(tmp) >= 2
        return n_atoms

    def n_atom_at_least_2_nei_more_than_2_conn(self, mol: dm.Mol):
        """16. Number of atoms where at least 2 connected atoms have more than 2 connections"""
        n_atoms = 0
        for atom in mol.GetAtoms():
            tmp = [x for x in atom.GetNeighbors() if len(x.GetNeighbors()) > 2]
            n_atoms += len(tmp) > 2
        return n_atoms

    def abs_scaffold_format_charge(self, mol: dm.Mol):
        """17. absolute value of the scaffold formal charge"""
        charge = GetFormalCharge(mol)
        return abs(charge)

    def n_bonds(self, mol: dm.Mol):
        """18. number of bonds"""
        return mol.GetNumBonds()

    def n_multiple_non_conj_ring_bonds(self, mol: dm.Mol):
        """19. number of multiple, nonconjugated ring bonds"""
        extracted_rings = []
        nr_multiple_bonds_infcr = 0  # infcr: in not fully conjugated ring
        rings = Chem.GetSymmSSSR(mol)
        for i in range(len(rings)):
            extracted_rings.append(list(rings[i]))
        for ring in extracted_rings:
            if not _is_ring_fully_conjugated(mol, ring):
                nr_multiple_bonds_infcr += _n_multiple_bond_in_ring(mol, ring)
        return nr_multiple_bonds_infcr

    def n_bonds_2_heteroatoms(self, mol: dm.Mol):
        """20. number of bonds connecting 2 heteroatoms"""
        sm = dm.from_smarts("[!#1&!#6]~[!#1&!#6]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_carbon_het_carbon_het_bonds(self, mol: dm.Mol):
        """21. number of bonds connecting 2 heteroatoms through 2 carbons"""
        sm = dm.from_smarts("[!#1&!#6]~[#6]~[#6]~[!#1&!#6]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_bonds_at_least_3_conn(self, mol: dm.Mol):
        """22. number of bonds with at least 3 connections on both its atoms"""
        sm = dm.from_smarts("[$([!#1](~[!#1])(~[!#1])~[!#1])][$([!#1](~[!#1])(~[!#1])~[!#1])]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_exocyclic_single_bonds_carbon(self, mol: dm.Mol):
        """23. number of exocyclic single bonds where a ring atom is carbon"""
        sm = dm.from_smarts("[!R;!#1]-[#6;R]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_exocyclic_single_bonds_nitrogen(self, mol: dm.Mol):
        """24. number of exocyclic single bonds where a ring atom is nitrogen"""
        sm = dm.from_smarts("[!R;!#1]-[#7;R]")
        return len(mol.GetSubstructMatches(sm, uniquify=True))

    def n_non_ring_bonds_2_conj_rings(self, mol: dm.Mol):
        """25. number of non-ring bonds connecting 2 nonconjugated rings"""
        # EN: this is interpretated literally as bonds and not path
        ring_atom_conj_state = _ring_atom_state(mol)
        sm = dm.from_smarts("[R:1]!@[R:2]")
        bond_list = mol.GetSubstructMatches(sm, uniquify=True)
        result = 0
        for a_start, a_end in bond_list:
            s_state = ring_atom_conj_state.get(a_start)
            e_state = ring_atom_conj_state.get(a_end)
            if False in s_state and False in e_state:
                result += 1
        return result

    def n_non_ring_bonds_conj_nonconj_rings(self, mol: dm.Mol):
        """
        26. number of non-ring bonds connecting 2 rings,
        one of them conjugated and one non-conjugated
        """
        # EN: this is interpretated literally as bonds and not path

        ring_atom_conj_state = _ring_atom_state(mol)
        sm = dm.from_smarts("[R:1]!@[R:2]")
        bond_list = mol.GetSubstructMatches(sm, uniquify=True)
        result = 0
        for a_start, a_end in bond_list:
            s_state = ring_atom_conj_state.get(a_start)
            e_state = ring_atom_conj_state.get(a_end)
            if (True in s_state and False in e_state) or (False in s_state and True in e_state):
                result += 1
        return result

    def n_bonds_atoms_with_at_least_one_nei_with_2_conn(self, mol: dm.Mol):
        """
        27. number of bonds where both atoms have at least one neighbor
        (not considering the bond atoms) with more than 2 connections
        """
        result = 0
        huge_conn = list(
            itertools.chain(*mol.GetSubstructMatches(dm.from_smarts("[*;!D0;!D1;!D2]"), uniquify=1))
        )
        for bond in mol.GetBonds():
            a_start, a_end = bond.GetBeginAtom(), bond.GetEndAtom()
            # we need to exclud the bond atom themselves
            allowed_conn_table = [
                x for x in huge_conn if x not in [a_start.GetIdx(), a_end.GetIdx()]
            ]
            if any([x.GetIdx() in allowed_conn_table for x in a_start.GetNeighbors()]) and any(
                [y.GetIdx() in allowed_conn_table for y in a_end.GetNeighbors()]
            ):
                result += 1
        return result

    def n_simple_rings(self, mol: dm.Mol):
        """28. number of simple rings"""
        ri = mol.GetRingInfo()
        return ri.NumRings()

    def size_largest_ring(self, mol: dm.Mol):
        """29. Size of the largest ring"""
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        return max_ring_size

    def n_simple_rings_no_heteroatoms(self, mol: dm.Mol):
        """30. number of simple rings with no heteroatoms"""
        ri = mol.GetRingInfo()
        n_heteros = _count_heteroatom_per_ring(mol, ri.AtomRings())
        return sum(1 for x in n_heteros if x == 0)

    def n_simple_rings_1_heteroatoms(self, mol: dm.Mol):
        """31. number of simple rings with 1 heteroatom"""
        ri = mol.GetRingInfo()
        n_heteros = _count_heteroatom_per_ring(mol, ri.AtomRings())
        return sum(1 for x in n_heteros if x == 1)

    def n_simple_rings_2_heteroatoms(self, mol: dm.Mol):
        """32. number of simple rings with 2 heteroatom"""
        ri = mol.GetRingInfo()
        n_heteros = _count_heteroatom_per_ring(mol, ri.AtomRings())
        return sum(1 for x in n_heteros if x == 2)

    def n_simple_rings_at_least_3_heteroatoms(self, mol: dm.Mol):
        """33. number of simple rings with 3 or more heteroatoms"""
        ri = mol.GetRingInfo()
        n_heteros = _count_heteroatom_per_ring(mol, ri.AtomRings())
        return sum(1 for x in n_heteros if x >= 3)

    def n_simple_non_conj_5_atoms_rings(self, mol: dm.Mol):
        """34. number of simple non-conjugated rings with 5 atoms"""
        ri = mol.GetRingInfo()
        n = 0
        for ring in ri.AtomRings():
            if not _is_ring_fully_conjugated(mol, ring) and len(ring) == 5:
                n += 1
        return n

    def n_simple_non_conj_6_atoms_rings(self, mol: dm.Mol):
        """35. number of simple non-conjugated rings with 6 atoms"""
        ri = mol.GetRingInfo()
        n = 0
        for ring in ri.AtomRings():
            if not _is_ring_fully_conjugated(mol, ring) and len(ring) == 6:
                n += 1
        return n

    def n_ring_system(self, mol: dm.Mol):
        """36. number of ring systems"""
        simple_rings, ring_system, _ = _get_ring_system(mol)
        return len(ring_system)

    def n_ring_system_with_2_non_conj_simple_ring(self, mol: dm.Mol):
        """37. number of rings systems with 2 non-conjugated simple rings"""
        simple_rings, _, ring_map = _get_ring_system(mol)
        conj_rings_map = dict(
            (i, _is_ring_fully_conjugated(mol, x)) for i, x in enumerate(simple_rings)
        )
        result = 0
        for ring_set in ring_map:
            n_not_conj = sum(not conj_rings_map[rnum] for rnum in ring_set)
            result += n_not_conj == 2
        return result

    def n_ring_system_with_2_conj_simple_ring(self, mol: dm.Mol):
        """38. number of rings systems with 2 conjugated simple rings"""
        simple_rings, _, ring_map = _get_ring_system(mol)
        conj_rings_map = dict(
            (i, _is_ring_fully_conjugated(mol, x)) for i, x in enumerate(simple_rings)
        )
        result = 0
        for ring_set in ring_map:
            n_conj = sum(conj_rings_map[rnum] for rnum in ring_set)
            result += n_conj == 2
        return result

    def n_ring_system_with_conj_non_conj_simple_ring(self, mol: dm.Mol):
        """39 number of ring system containing 2 simple rings, one conjugated and one nonconjugated"""
        simple_rings, _, ring_map = _get_ring_system(mol)
        conj_rings_map = dict(
            (i, _is_ring_fully_conjugated(mol, x)) for i, x in enumerate(simple_rings)
        )
        result = 0
        for ring_set in ring_map:
            if len(ring_set) == 2:
                n_conj = sum(conj_rings_map[rnum] for rnum in ring_set)
                result += n_conj == 1
        return result

    def n_ring_system_with_3_conj_simple_ring(self, mol: dm.Mol):
        """40. number of rings systems with 3 conjugated simple rings"""
        simple_rings, _, ring_map = _get_ring_system(mol)
        conj_rings_map = dict(
            (i, _is_ring_fully_conjugated(mol, x)) for i, x in enumerate(simple_rings)
        )
        result = 0
        for ring_set in ring_map:
            n_conj = sum(conj_rings_map[rnum] for rnum in ring_set)
            result += n_conj == 3
        return result

    def n_ring_system_with_3_non_conj_simple_ring(self, mol: dm.Mol):
        """41. number of rings systems with 3 non-conjugated simple rings"""
        simple_rings, _, ring_map = _get_ring_system(mol)
        conj_rings_map = dict(
            (i, _is_ring_fully_conjugated(mol, x)) for i, x in enumerate(simple_rings)
        )
        result = 0
        for ring_set in ring_map:
            n_not_conj = sum(not conj_rings_map[rnum] for rnum in ring_set)
            result += n_not_conj == 3
        return result

    def n_ring_system_with_greater_one_conj_nonconj_simple_ring(self, mol: dm.Mol):
        """42. number of ring system containing 3 simple rings, at least one conjugated and one nonconjugated"""
        simple_rings, _, ring_map = _get_ring_system(mol)
        conj_rings_map = dict(
            (i, _is_ring_fully_conjugated(mol, x)) for i, x in enumerate(simple_rings)
        )
        result = 0
        for ring_set in ring_map:
            if len(ring_set) == 3:
                n_conj = sum(conj_rings_map[rnum] for rnum in ring_set)
                result += n_conj in [1, 2]
        return result

    @property
    def columns(self):
        """Get the name of all the descriptors of this calculator"""
        return list(self.DESCRIPTORS)

    def __call__(self, mol: Union[dm.Mol, str]):
        r"""
        Compute the Fingerprint of a molecule

        Args:
            mol: the molecule of interest

        Returns:
            props (np.ndarray): list of computed rdkit molecular descriptors
        """
        mol = dm.to_mol(mol)
        if self.use_scaffold and mol is not None:
            mol = MurckoScaffold.GetScaffoldForMol(mol)

        props = []
        for k in self.DESCRIPTORS:
            try:
                fn = getattr(self, k)
                props.append(fn(mol))
            except Exception as e:
                if self.verbose:
                    logger.error(e)
                props.append(float("nan"))
        props = np.asarray(props)
        if self.normalize:
            return self.compute_normalization(props)
        return props


def skdistance(
    sk1: np.ndarray,
    sk2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cdist: bool = False,
):
    """Compute the scaffold distance between two scaffold keys
    as described in https://pubs.acs.org/doi/abs/10.1021/ci5001983.
    The input features are expected to be normalized beforehand (see paper)

    Args:
        sk1: scaffold key 1
        sk2: scaffold key 2
        weights: how to weight each of the features. By default rank ordering is used.
        cdist: whether to compute the features on a batched of inputs (expected 2D)

    Returns:
        dist (float): distance between two scaffold keys
    """
    if weights is None:
        weights = 1 / (np.arange(sk1.shape[-1]) + 1)

    if cdist:
        sk1 = np.atleast_2d(sk1)
        sk2 = np.atleast_2d(sk2)
        val = np.abs(sk1[:, None] - sk2[:]) ** 1.5
        dist = np.sum(val * weights, axis=-1)
    else:
        if any((sk.ndim > 1 and sk.shape[0] != 1) for sk in [sk1, sk2]):
            raise ValueError("`cdist` mode was not detected, you need to provide single vectors")
        val = np.abs(sk1 - sk2) ** 1.5
        dist = np.sum(val * weights)
    return dist
