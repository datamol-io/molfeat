import pytest

from molfeat.calc.atom import AtomCalculator
from molfeat.calc.bond import BondCalculator
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.calc._atom_bond_features import bond_type_one_hot
from molfeat.calc._atom_bond_features import bond_is_in_ring
from molfeat.calc._atom_bond_features import atom_degree_one_hot
from molfeat.calc._atom_bond_features import atom_one_hot
from molfeat.calc._atom_bond_features import atom_implicit_valence_one_hot

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
