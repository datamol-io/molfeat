import unittest as ut
import pytest

import datamol as dm
import numpy as np

try:
    from dgllife.utils import CanonicalAtomFeaturizer
    from dgllife.utils import CanonicalBondFeaturizer
    from dgllife.utils import WeaveAtomFeaturizer
    from dgllife.utils import WeaveEdgeFeaturizer
except ImportError:
    pass

from molfeat.calc.atom import AtomCalculator
from molfeat.calc.atom import DGLCanonicalAtomCalculator
from molfeat.calc.atom import DGLWeaveAtomCalculator
from molfeat.calc.bond import BondCalculator
from molfeat.calc.bond import DGLCanonicalBondCalculator
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.calc.bond import DGLWeaveEdgeCalculator
from molfeat.calc._atom_bond_features import bond_type_one_hot
from molfeat.calc._atom_bond_features import bond_is_in_ring
from molfeat.calc._atom_bond_features import atom_degree_one_hot
from molfeat.calc._atom_bond_features import atom_one_hot
from molfeat.calc._atom_bond_features import atom_implicit_valence_one_hot

from molfeat.utils import requires


CUSTOM_BOND_FEATURIZER = {
    "bond_type_one_hot": bond_type_one_hot,
    "bond_is_in_ring": bond_is_in_ring,
    "bond_custom_val": lambda x: [0, 0],
}
CUSTOM_ATOM_FEATURIZER = {
    "atom_one_hot": atom_one_hot,
    "atom_degree_one_hot": atom_degree_one_hot,
    "atom_custom_val_1": lambda x: [0, 0],
    "atom_custom_val_2": atom_implicit_valence_one_hot,
}
CALCULATOR_SPECS = {
    # fp
    "AtomCalculator": lambda: AtomCalculator(),
    "WeaveAtomCalculator": lambda: DGLWeaveAtomCalculator(),
    "BondCalculator": lambda: BondCalculator(),
    "EdgeMatCalculator": lambda: EdgeMatCalculator(),
    "CustomBondCalculator": lambda: BondCalculator(featurizer_funcs=CUSTOM_BOND_FEATURIZER),
    "CustomAtomCalculator": lambda: AtomCalculator(featurizer_funcs=CUSTOM_ATOM_FEATURIZER),
}


@pytest.mark.parametrize(
    "calculator_builder",
    CALCULATOR_SPECS.values(),
    ids=CALCULATOR_SPECS.keys(),
)
def test_to_from_state(calculator_builder):
    calculator = calculator_builder()
    # check to_state
    state = calculator.to_state_dict()
    assert "name" in state
    assert "args" in state
    assert "_molfeat_version" in state

    # check from_state
    calculator2 = calculator.__class__.from_state_dict(state)
    state2 = calculator2.to_state_dict()
    assert state == state2


@pytest.mark.xfail(not requires.check("dgllife"), reason="3rd party module dgllife is missing")
class TestGraphCalculator(ut.TestCase):
    r"""Test cases for basic graph featurizer vs dgl generation"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
        "C.C1CCC1",
        "N",
    ]
    mols = [dm.to_mol(x) for x in smiles]

    def test_atom_calculator(self):
        atom_calc = AtomCalculator()
        dgl_atom_calc = DGLCanonicalAtomCalculator()
        true_dgl_atom_calc = CanonicalAtomFeaturizer(atom_data_field="hv")
        for mol in self.mols:
            a1 = atom_calc(mol)
            a2 = dgl_atom_calc(mol)
            a3 = true_dgl_atom_calc(mol)
            np.testing.assert_allclose(a2["hv"], a3["hv"])

    def test_bond_calculator(self):
        bond_calc = BondCalculator()
        dgl_bond_calc = DGLCanonicalBondCalculator()
        true_dgl_bond_calc = CanonicalBondFeaturizer(bond_data_field="he")
        for mol in self.mols:
            b1 = bond_calc(mol)
            b2 = dgl_bond_calc(mol)
            b3 = true_dgl_bond_calc(mol)
            if "he" in b3:
                np.testing.assert_allclose(b2["he"], b3["he"])

    def test_weave_atom_calculator(self):
        true_dgl_weave_calc = WeaveAtomFeaturizer(atom_data_field="hv")
        dgl_weave_calc = DGLWeaveAtomCalculator()
        for mol in self.mols:
            b1 = true_dgl_weave_calc(mol)
            b2 = dgl_weave_calc(mol)
            np.testing.assert_allclose(b1["hv"], b2["hv"])

    def test_weave_bond_calculator(self):
        true_dgl_weave_calc = WeaveEdgeFeaturizer(edge_data_field="he")
        dgl_weave_calc = DGLWeaveEdgeCalculator()
        for mol in self.mols:
            b1 = true_dgl_weave_calc(mol)
            b2 = dgl_weave_calc(mol)
            np.testing.assert_allclose(b1["he"], b2["he"])


if __name__ == "__main__":
    ut.main()
