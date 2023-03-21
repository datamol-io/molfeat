import pytest

import joblib
import io
import functools

import datamol as dm
import numpy as np

from molfeat.calc import Pharmacophore2D
from molfeat.calc import Pharmacophore3D

FACTORIES = ["default", "cats", "pmapper", "gobbi"]
FACTORIES_DEFAULT_SIZE = [19355, 189, 19355, 39972]


def test_pharm_calculator():
    smiles_list = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
        "COc1cc(C(=O)N2CCSc3ccc(C(F)(F)F)cc32)on1",
    ]

    # check that we can compute fps for each of them
    for i, factory in enumerate(FACTORIES):
        calc = Pharmacophore2D(factory=factory, length=None)
        fps = [calc(sm) for sm in smiles_list]
        assert fps[0].shape[-1] == FACTORIES_DEFAULT_SIZE[i]


def test_pharm2d_binary_output():
    mol = dm.to_mol("COc1cc(C(=O)N2CCSc3ccc(C(F)(F)F)cc32)on1")

    featurizer = Pharmacophore2D(length=2048)
    fp = featurizer(mol)

    assert len(np.unique(fp)) == 2 or len(np.unique(fp)) == 1


def test_pharmacophore_2d_invalid_mol():
    featurizer = Pharmacophore2D()

    with pytest.raises(ValueError):
        featurizer("invalid_mol")


def test_pharmacophore_2d_wrong_factory():
    with pytest.raises(ValueError):
        Pharmacophore2D(factory="i_dont_exist")


def test_pharmacophore_2d_override():
    featurizer = Pharmacophore2D(
        useCounts=True,
        minPointCount=3,
        maxPointCount=4,
        trianglePruneBins=True,
        shortestPathsOnly=False,
        skipFeats=[
            "A",
        ],
        includeBondOrder=True,
    )

    assert featurizer.useCounts == True
    assert featurizer.minPointCount == 3
    assert featurizer.maxPointCount == 4
    assert featurizer.trianglePruneBins == True
    assert featurizer.shortestPathsOnly == False
    assert featurizer.skipFeats == [
        "A",
    ]
    assert featurizer.includeBondOrder == True

    assert featurizer.useCounts == featurizer.sig_factory.useCounts
    assert featurizer.minPointCount == featurizer.sig_factory.minPointCount
    assert featurizer.maxPointCount == featurizer.sig_factory.maxPointCount
    assert featurizer.trianglePruneBins == featurizer.sig_factory.trianglePruneBins
    assert featurizer.shortestPathsOnly == featurizer.sig_factory.shortestPathsOnly
    assert featurizer.skipFeats == featurizer.sig_factory.skipFeats
    assert featurizer.includeBondOrder == featurizer.sig_factory.includeBondOrder


def test_pharm_refold_default():
    smiles_list = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]

    calc = Pharmacophore2D(factory="default", length=2000)
    fps = [calc(sm) for sm in smiles_list]
    assert fps[0].shape[-1] == 2000


def test_pharm_refold_cats():
    smiles_list = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]

    calc = Pharmacophore2D(factory="cats", length=2000)
    fps = [calc(sm) for sm in smiles_list]
    assert fps[0].shape[-1] == 2000
    assert np.sum(fps[0][190:]) == 0


def test_pharmacophore_3d_no_conformers():
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    featurizer = Pharmacophore3D()

    with pytest.raises(ValueError):
        featurizer(mol)


def test_pharmacophore_3d_invalid_mol():
    featurizer = Pharmacophore3D()

    with pytest.raises(ValueError):
        featurizer("invalid_mol")


@pytest.mark.xfail
def test_pharmacophore_3d():
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    mol = dm.conformers.generate(mol)

    factories = ["default", "cats", "pmapper", "gobbi"]

    # NOTE(hadim): It will work as long as the conformer generation remains the same.
    excepted_sum = [50, 102, 76, 153]

    for factory, s in zip(factories, excepted_sum):
        featurizer = Pharmacophore3D(factory=factory, length=2048)
        fp = featurizer(mol)

        assert s == fp.sum()


def test_pharmacophore_3d_consensus():
    smiles_list = [
        "Nc1cnn(-c2ccccc2)c(=O)c1Cl",
        "Cc1ccn(-c2ccccc2)c(=O)c1F",
        "Cc1cnn(-c2ccccc2)c(=O)c1Cl",
        "Cc1cnn(-c2ccccc2)c(=O)c1",
    ]
    mols = [dm.to_mol(smiles) for smiles in smiles_list]
    mols = [
        dm.conformers.generate(mol, n_confs=1, use_random_coords=False, random_seed=1)
        for mol in mols
    ]

    featurizer = Pharmacophore3D(bin_step=2)

    # EN: using 'brute' or 'auto' instead for the algorithm will likely results in
    # a segmentation fault, seems to be numpy related on distance computation
    consensus_fp = featurizer.consensus_fp(mols, eps=1, min_samples_ratio=0.5, algorithm="kd_tree")
    # Check
    assert len(consensus_fp) == 2048
    assert consensus_fp.sum() == 4


def test_pharmacophore_3d_geat_features():
    mol = dm.to_mol("Nc1cnn(-c2ccccc2)c(=O)c1Cl")
    mol = dm.conformers.generate(mol)

    featurizer = Pharmacophore3D()
    features = featurizer.get_features(mol)

    assert set(features.columns.tolist()) == {
        "feature_id",
        "feature_name",
        "feature_type",
        "atom_indices",
        "coords",
    }


def test_pharmacophore_3d_get_features_from_many():
    smiles_list = [
        "Nc1cnn(-c2ccccc2)c(=O)c1Cl",
        "Cc1ccn(-c2ccccc2)c(=O)c1F",
        "Cc1cnn(-c2ccccc2)c(=O)c1Cl",
        "Cc1cnn(-c2ccccc2)c(=O)c1",
    ]
    mols = [dm.to_mol(smiles) for smiles in smiles_list]
    mols = [dm.conformers.generate(mol, n_confs=1) for mol in mols]

    featurizer = Pharmacophore3D()

    features = featurizer.get_features_from_many(mols, keep_mols=True)

    assert set(features.columns.tolist()) == {
        "feature_id",
        "feature_name",
        "feature_type",
        "atom_indices",
        "coords",
        "mol_index",
        "mol",
    }

    features = featurizer.get_features_from_many(mols, keep_mols=False)
    assert "mol" not in features.columns


def test_pharmacophore_3d_without_features():
    smiles = "S=P(N1CC1)(N1CC1)N1CC1"
    mol = dm.to_mol(smiles)
    mol = dm.conformers.generate(mol, n_confs=1)

    featurizer = Pharmacophore3D(bin_step=2)
    fp = featurizer(mol)

    assert fp.sum() == 0


def test_pharmacophore_2d_pickling():
    # test pickling itself
    featurizer = Pharmacophore2D(length=8567)

    # pickle
    file_buffer = io.BytesIO()
    joblib.dump(featurizer, file_buffer)

    # reload
    featurizer_reloaded = joblib.load(file_buffer)

    # check
    assert featurizer_reloaded.length == featurizer.length


def test_pharmacophore_3d_pickling():
    # test pickling itself
    featurizer = Pharmacophore3D(length=8567, bin_step=0.5)

    # pickle
    file_buffer = io.BytesIO()
    joblib.dump(featurizer, file_buffer)

    # reload
    featurizer_reloaded = joblib.load(file_buffer)

    # check
    assert featurizer_reloaded.length == featurizer.length
    assert featurizer_reloaded.bin_step == featurizer.bin_step


def test_pharmacophore_2d_loky():
    # Get some data
    data = dm.freesolv()
    data = data.iloc[:10]
    data["mol"] = data["smiles"].apply(dm.to_mol)

    # Build featurizer

    # Compute with processes
    featurizer_proc = Pharmacophore2D(length=1999)
    features_proc = dm.parallelized(
        featurizer_proc,
        data["mol"].values,
        progress=True,
        scheduler="processes",
    )

    # Compute with threads
    featurizer_threads = Pharmacophore2D(length=1999)
    features_threads = dm.parallelized(
        featurizer_threads,
        data["mol"].values,
        progress=True,
        scheduler="threads",
    )

    # check
    assert featurizer_threads.length == featurizer_proc.length
    assert np.allclose(features_proc, features_threads)


def test_pharmacophore_3d_loky():
    # Get some data
    data = dm.freesolv()
    data = data.iloc[:10]
    data["mol"] = data["smiles"].apply(dm.to_mol)

    conf_fn = functools.partial(dm.conformers.generate, n_confs=1)
    data["mol"] = data["mol"].apply(conf_fn)

    # Build featurizer

    # Compute with processes
    featurizer_proc = Pharmacophore3D(length=1999, bin_step=2)
    features_proc = dm.parallelized(
        featurizer_proc,
        data["mol"].values,
        progress=True,
        scheduler="processes",
    )

    # Compute with threads
    featurizer_threads = Pharmacophore3D(length=1999, bin_step=2)
    features_threads = dm.parallelized(
        featurizer_threads,
        data["mol"].values,
        progress=True,
        scheduler="threads",
    )

    # check
    assert featurizer_threads.length == featurizer_proc.length
    assert featurizer_threads.bin_step == featurizer_proc.bin_step
    assert np.allclose(features_proc, features_threads)
