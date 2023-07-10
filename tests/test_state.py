import datamol as dm
import numpy as np
import pytest

from molfeat._version import __version__ as MOLFEAT_VERSION
from molfeat.calc import (
    CATS,
    FPCalculator,
    Pharmacophore2D,
    RDKitDescriptors2D,
    ScaffoldKeyCalculator,
)
from molfeat.calc._atom_bond_features import atom_chiral_tag_one_hot, atom_one_hot
from molfeat.calc.atom import AtomCalculator, AtomMaterialCalculator
from molfeat.calc.bond import BondCalculator
from molfeat.trans.base import MoleculeTransformer, PrecomputedMolTransformer
from molfeat.trans.fp import FPVecFilteredTransformer, FPVecTransformer
from molfeat.trans.graph import (
    AdjGraphTransformer,
    DGLGraphTransformer,
    MolTreeDecompositionTransformer,
    PYGGraphTransformer,
    TopoDistGraphTransformer,
)
from molfeat.trans.pretrained import (
    GraphormerTransformer,
    PretrainedDGLTransformer,
    PretrainedHFTransformer,
)
from molfeat.utils.cache import FileCache, MolToKey
from molfeat.utils.state import compare_state


def _dummy_featurizer_fn(x):
    return np.random.random(1024)


FEATURIZERS_SPEC = {
    # fp
    "FPVecTransformer": lambda: FPVecTransformer(
        kind="ecfp:4",
        length=2044,
        useFeatures=True,
        dtype=np.float64,
    ),
    "MoleculeTransformerFP": lambda: MoleculeTransformer(
        FPCalculator("ecfp", radius=2),
        dtype=np.float64,
    ),
    "FPVecFilteredTransformer": lambda: FPVecFilteredTransformer(),
    # base
    "MoleculeTransformer": lambda: MoleculeTransformer(featurizer=_dummy_featurizer_fn),
    # pretrained
    "PretrainedDGLTransformer": lambda: PretrainedDGLTransformer(),
    "GraphormerTransformer": lambda: GraphormerTransformer(),
    "PretrainedHFTransformer": lambda: PretrainedHFTransformer(),
    # graph
    "AdjGraphTransformer_with_bonds": lambda: AdjGraphTransformer(
        atom_featurizer=AtomMaterialCalculator(),
        bond_featurizer=BondCalculator(),
    ),
    "AdjGraphTransformer_with_bonds_custom_atom": lambda: AdjGraphTransformer(
        atom_featurizer=AtomMaterialCalculator(
            featurizer_funcs={
                "atom_chiral_tag_one_hot": atom_chiral_tag_one_hot,
                "atom_one_hot": atom_one_hot,
            }
        ),
        bond_featurizer=BondCalculator(),
    ),
    "AdjGraphTransformer": lambda: AdjGraphTransformer(atom_featurizer=AtomCalculator()),
    "DGLGraphTransformer": lambda: DGLGraphTransformer(),
    "TopoDistGraphTransformer": lambda: TopoDistGraphTransformer(),
    "PYGGraphTransformer": lambda: PYGGraphTransformer(),
    "MolTreeDecompositionTransformer": lambda: MolTreeDecompositionTransformer(verbose=False),
}
FEATURIZERS_ATOM_PICKLES = ["AdjGraphTransformer_with_bonds_custom_atom"]
FEATURIZERS_BUILDER_ATOM_PICKLES = [FEATURIZERS_SPEC[k] for k in FEATURIZERS_ATOM_PICKLES]
FEATURIZERS_NAMES, FEATURIZERS_BUILDER = zip(
    *[(k, v) for k, v in FEATURIZERS_SPEC.items() if k not in FEATURIZERS_ATOM_PICKLES]
)


@pytest.mark.parametrize(
    "featurizer_builder",
    FEATURIZERS_BUILDER,
    ids=FEATURIZERS_NAMES,
)
def test_to_from_state(featurizer_builder):
    featurizer: MoleculeTransformer = featurizer_builder()

    # check to_state
    state = featurizer.to_state_dict()
    assert "name" in state
    assert "args" in state
    assert "_molfeat_version" in state

    # check from_state
    featurizer = MoleculeTransformer.from_state_dict(state)
    state2 = featurizer.to_state_dict()
    assert state == state2


@pytest.mark.parametrize(
    "featurizer_builder",
    FEATURIZERS_BUILDER,
    ids=FEATURIZERS_NAMES,
)
def test_to_from_state_yaml(featurizer_builder, tmp_path):
    featurizer: MoleculeTransformer = featurizer_builder()

    featurizer_path = tmp_path / "featurizer.yaml"

    featurizer.to_state_yaml_file(featurizer_path)
    state = featurizer.to_state_dict()

    featurizer = MoleculeTransformer.from_state_yaml_file(featurizer_path)
    state2 = featurizer.to_state_dict()
    assert state == state2


@pytest.mark.parametrize(
    "featurizer_builder",
    FEATURIZERS_BUILDER,
    ids=FEATURIZERS_NAMES,
)
def test_to_from_state_json(featurizer_builder, tmp_path):
    featurizer: MoleculeTransformer = featurizer_builder()

    featurizer_path = tmp_path / "featurizer.json"

    featurizer.to_state_json_file(featurizer_path)
    state = featurizer.to_state_dict()

    featurizer = MoleculeTransformer.from_state_json_file(featurizer_path)
    state2 = featurizer.to_state_dict()
    assert state == state2


@pytest.mark.parametrize(
    "featurizer_builder",
    FEATURIZERS_BUILDER_ATOM_PICKLES,
    ids=FEATURIZERS_ATOM_PICKLES,
)
def test_state_atom_bond_pickle(featurizer_builder, tmp_path):
    featurizer: MoleculeTransformer = featurizer_builder()

    # check to_state
    state = featurizer.to_state_dict()
    assert "name" in state
    assert "args" in state
    assert "_molfeat_version" in state

    # check from_state
    featurizer = MoleculeTransformer.from_state_dict(state)
    state2 = featurizer.to_state_dict()
    featurizer2 = MoleculeTransformer.from_state_dict(state2)
    to_remove = []
    for key in state["args"]:
        if key.endswith("is_pickled") and state["args"][key] == True:
            to_remove.append(key.replace("is_pickled", "").strip("_"))
    for key_val in to_remove:
        state["args"].pop(key_val, None)
        state2["args"].pop(key_val, None)
    assert state == state2
    mols = dm.freesolv().smiles.iloc[:10]
    out = featurizer(mols)
    out2 = featurizer2(mols)
    assert len(out) == len(out2) == len(mols)
    for o1, o2 in zip(out, out2):
        if isinstance(o1, (list, tuple)):
            assert np.all([np.allclose(i, j, atol=2) for i, j in zip(o1, o2)]) == True
        else:
            np.testing.assert_array_equal(o1, o2)


def test_PrecomputedMolTransformer_state(tmp_path):
    cache_path = tmp_path / "cache.parquet"
    smiles_list = ["[NH3+]CCSc1cc(-c2ccc[nH]2)c2c3c(ccc(F)c13)NC2=O", "CCCO"]

    featurizer = FPVecTransformer(kind="ecfp", length=2048)
    cache = FileCache(cache_file=cache_path, file_type="parquet", n_jobs=-1)

    featurizer_cache = PrecomputedMolTransformer(cache=cache, featurizer=featurizer)
    fps = featurizer_cache(smiles_list)

    # sanity check
    assert len(featurizer_cache.cache.cache) == 2

    # serialize to state
    state = featurizer_cache.to_state_dict()

    assert state["name"] == "PrecomputedMolTransformer"
    assert "base_featurizer" in state
    assert "cache" in state
    assert "_molfeat_version" in state

    # reload from state
    featurizer_cache2: PrecomputedMolTransformer = MoleculeTransformer.from_state_dict(state)  # type: ignore

    assert featurizer_cache2.base_featurizer.to_state_dict() == featurizer.to_state_dict()
    assert len(featurizer_cache2.cache.cache) == 2
    assert all(
        featurizer_cache2.cache.to_dataframe(pack_bits=True)
        == featurizer_cache2.cache.to_dataframe(pack_bits=True)
    )


def test_PrecomputedMolTransformer_init_from_state_file(tmp_path):
    cache_path = str(tmp_path / "cache.parquet")
    smiles_list = ["[NH3+]CCSc1cc(-c2ccc[nH]2)c2c3c(ccc(F)c13)NC2=O", "CCCO"]

    featurizer = FPVecTransformer(kind="ecfp", length=2048)
    cache = FileCache(cache_file=cache_path, file_type="parquet", n_jobs=-1)

    featurizer_cache = PrecomputedMolTransformer(cache=cache, featurizer=featurizer)
    fps = featurizer_cache(smiles_list)

    # sanity check
    assert len(featurizer_cache.cache.cache) == 2

    # serialize to state
    state = featurizer_cache.to_state_dict()

    # export yaml state file
    state_path = str(tmp_path / "state.yaml")

    featurizer_cache.to_state_yaml_file(filepath=state_path)

    assert state["name"] == "PrecomputedMolTransformer"
    assert "base_featurizer" in state
    assert "cache" in state
    assert "_molfeat_version" in state

    # reload from state
    featurizer_cache2: PrecomputedMolTransformer = PrecomputedMolTransformer(state_path=state_path)

    assert featurizer_cache2.base_featurizer.to_state_dict() == featurizer.to_state_dict()
    assert len(featurizer_cache2.cache.cache) == 2
    assert all(
        featurizer_cache2.cache.to_dataframe(pack_bits=True)
        == featurizer_cache2.cache.to_dataframe(pack_bits=True)
    )


def test_moltokey_state():
    hasher = MolToKey()
    state = hasher.to_state_dict()

    assert state == {"hash_name": "dm.unique_id"}

    hasher = MolToKey.from_state_dict(state)
    assert hasher.hash_name == "dm.unique_id"
    assert hasher.hash_fn == dm.unique_id


def test_fp_state():
    expected_res = [
        {
            "name": "FPCalculator",
            "module": "molfeat.calc.fingerprints",
            "args": {"length": 512, "method": "ecfp", "counting": False, "nBits": 512},
            "_molfeat_version": MOLFEAT_VERSION,
        },
        {
            "name": "FPCalculator",
            "module": "molfeat.calc.fingerprints",
            "args": {
                "length": 241,
                "method": "fcfp-count",
                "counting": True,
                "nBits": 241,
            },
            "_molfeat_version": MOLFEAT_VERSION,
        },
        {
            "name": "FPCalculator",
            "module": "molfeat.calc.fingerprints",
            "args": {"length": 2048, "method": "maccs", "counting": False},
            "_molfeat_version": MOLFEAT_VERSION,
        },
    ]
    for i, (fp_name, fp_len) in enumerate(zip(["ecfp", "fcfp-count", "maccs"], [512, 241, 2048])):
        calc = FPCalculator(fp_name, fp_len)
        state = calc.to_state_dict()
        assert state == expected_res[i]
    # check reloading
    calc = FPCalculator("ecfp", length=512, radius=3, useChirality=True)
    state = calc.to_state_dict()
    mol = dm.to_mol("C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O")  # isomeric thalidomide
    expected = calc(mol)
    reloaded_calc = FPCalculator.from_state_dict(state)
    out = reloaded_calc(mol)
    np.testing.assert_array_almost_equal(expected, out)


def test_rdkit_descr_state():
    calc = RDKitDescriptors2D(replace_nan=True, augment=False)
    expected_state = {
        "name": "RDKitDescriptors2D",
        "module": "molfeat.calc.descriptors",
        "args": {
            "replace_nan": True,
            "augment": False,
            "descrs": None,
            "avg_ipc": True,
            "do_not_standardize": False,
        },
        "_molfeat_version": MOLFEAT_VERSION,
    }

    state = calc.to_state_dict()
    assert state == expected_state
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    expected = calc(mol)
    reloaded_calc = RDKitDescriptors2D.from_state_dict(state)
    out = reloaded_calc(mol)
    np.testing.assert_array_almost_equal(expected, out)


def test_pharmacophore_state():
    calc = Pharmacophore2D(factory="gobbi", length=1024, useCounts=False)
    expected_state = {
        "name": "Pharmacophore2D",
        "module": "molfeat.calc.pharmacophore",
        "args": {
            "factory": "gobbi",
            "useCounts": False,
            "minPointCount": 2,
            "maxPointCount": 3,
            "shortestPathsOnly": True,
            "includeBondOrder": False,
            "skipFeats": [],
            "trianglePruneBins": True,
            "bins": [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)],
            "length": 1024,
        },
        "_molfeat_version": MOLFEAT_VERSION,
    }

    state = calc.to_state_dict()
    assert state == expected_state
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    expected = calc(mol)
    reloaded_calc = Pharmacophore2D.from_state_dict(state)
    out = reloaded_calc(mol)
    np.testing.assert_array_almost_equal(expected, out)


def test_cats_state():
    calc = CATS(max_dist=8, scale="count", bins=[0, 3, 6, 8])
    expected_state = {
        "name": "CATS",
        "module": "molfeat.calc.cats",
        "args": {
            "max_dist": 8,
            "bins": [0, 3, 6, 8],
            "scale": "count",
            "use_3d_distances": False,
        },
        "_molfeat_version": MOLFEAT_VERSION,
    }

    state = calc.to_state_dict()
    assert state == expected_state
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    expected = calc(mol)
    reloaded_calc = CATS.from_state_dict(state)
    out = reloaded_calc(mol)
    np.testing.assert_array_almost_equal(expected, out)


def test_skeys_state():
    calc = ScaffoldKeyCalculator(normalize=True, use_scaffold=False, verbose=False)
    expected_state = {
        "name": "ScaffoldKeyCalculator",
        "module": "molfeat.calc.skeys",
        "args": {
            "normalize": True,
            "verbose": False,
            "use_scaffold": False,
        },
        "_molfeat_version": MOLFEAT_VERSION,
    }

    state = calc.to_state_dict()
    assert state == expected_state
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    expected = calc(mol)
    reloaded_calc = ScaffoldKeyCalculator.from_state_dict(state)
    out = reloaded_calc(mol)
    np.testing.assert_array_almost_equal(expected, out)


def test_filecache_state(tmp_path):
    # setup
    cache_path = tmp_path / "cache.parquet"
    cache = FileCache(cache_file=cache_path, file_type="parquet")
    cache(["CCC", "CCCCO"], featurizer=lambda x: [[0, 1, 1, 0], [0, 1, 1, 1]])

    # save state
    state = cache.to_state_dict(save_to_file=True)

    assert set(state.keys()) == {
        "_cache_name",
        "cache_file",
        "file_type",
        "mol_hasher",
        "n_jobs",
        "name",
        "clear_on_exit",
        "parquet_kwargs",
        "verbose",
    }

    assert dm.fs.exists(cache_path)

    # reload from state
    cache2 = FileCache.from_state_dict(state)
    assert len(cache2.cache) == 2
    assert all(cache.to_dataframe(pack_bits=True) == cache2.to_dataframe(pack_bits=True))


def test_state_comparison():
    state_dict1 = dict(_molfeat_version="0.1.2", value=2)
    state_dict2 = dict(_molfeat_version="0.1.2dev", value=2)
    state_dict3 = dict(_molfeat_version="0.1", value=2)
    state_dict4 = dict(value=2)
    state_dict5 = dict(_molfeat_version="1.2.3", value=100)
    state_dict6 = dict(_molfeat_version="0.1.2pre", value=0)

    expected_results = {
        "major": [True, True, True, False, False, False],
        1: [True, True, True, False, False, False],
        "micro": [True, True, False, False, False, False],
        None: [True, False, False, False, False, False],
    }
    for allowed_version in ["major", 1, "micro", None]:
        results = []
        results.append(compare_state(state_dict1, state_dict1, allow_version_level=allowed_version))
        results.append(compare_state(state_dict1, state_dict2, allow_version_level=allowed_version))
        results.append(compare_state(state_dict1, state_dict3, allow_version_level=allowed_version))
        results.append(compare_state(state_dict1, state_dict4, allow_version_level=allowed_version))
        results.append(compare_state(state_dict1, state_dict5, allow_version_level=allowed_version))
        results.append(compare_state(state_dict1, state_dict6, allow_version_level=allowed_version))
        assert expected_results[allowed_version] == results
