import datamol as dm
import pytest

from molfeat.utils.converters import SmilesConverter


@pytest.mark.parametrize("notation", [None, "none", "smiles", "inchi", "selfies"])
def test_notation_round_trip(notation):
    smiles = "C[C@H](O)Cl"
    converter = SmilesConverter(notation)
    encoded = converter.encode(smiles)
    assert isinstance(encoded, str)
    decoded = converter.decode(encoded)
    assert isinstance(decoded, str)
    assert dm.to_smiles(dm.to_mol(decoded)) == smiles


def test_invalid_inchi_returns_none():
    assert SmilesConverter("inchi").decode("invalid inchi") is None


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_missing_selfies_is_not_an_invalid_molecule(monkeypatch, operation):
    def missing_selfies(*args, **kwargs):
        raise ModuleNotFoundError("selfies is not installed")

    function = "to_selfies" if operation == "encode" else "from_selfies"
    monkeypatch.setattr(dm, function, missing_selfies)
    converter = SmilesConverter("selfies")
    with pytest.raises(ModuleNotFoundError, match="selfies"):
        getattr(converter, operation)("CCO")


@pytest.mark.parametrize("operation,value", [("encode", "invalid smiles"), ("decode", "[invalid]")])
def test_invalid_selfies_input_still_returns_none(operation, value):
    assert getattr(SmilesConverter("selfies"), operation)(value) is None
