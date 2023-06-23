import tempfile
import unittest as ut

import datamol as dm
import joblib
import numpy as np
import pytest

from molfeat.calc import (
    CATS,
    ElectroShapeDescriptors,
    RDKitDescriptors2D,
    RDKitDescriptors3D,
    ScaffoldKeyCalculator,
    USRDescriptors,
)
from molfeat.calc.skeys import skdistance


class TestDescPharm(ut.TestCase):
    r"""Test cases for descriptors and pharmacophore generation"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]
    EXTRA_LARGE_MOL = "CC(C)CC(NCCNC(=O)C(CCC(O)=O)NC(C)=O)C(=O)NC(Cc1ccc(O)cc1)C(=O)NC(CC(C)C)C(=O)NC(C(C)C)C(=O)NC(C)C(=O)NCC(=O)NC(CCC(O)=O)C(=O)NC(CCCNC(N)=N)C(=O)NCC(=O)NC(Cc1ccccc1)C(=O)NC(Cc1ccccc1)C(=O)NC(Cc1ccc(O)cc1)C(=O)NC(C(C)O)C(=O)N1CCCC1C(=O)NC(C)C(O)=O"

    def test_rdkit2d(self):
        calc = RDKitDescriptors2D()
        fps = calc(self.smiles[0])
        self.assertEqual(len(fps), len(calc))

        sm = "CC[C@@H]1[C@@H]2C[C@H](O)CC[C@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]3[C@H](C)CCOS(=O)(=O)O[Na])[C@@H]2[C@@H]1O"
        sm_disconnected = "CC[C@@H]1[C@@H]2C[C@H](O)CC[C@]2(C)[C@H]2CC[C@@]3(C)[C@@H](CC[C@@H]3[C@H](C)CCOS(=O)(=O)[O-])[C@@H]2[C@@H]1O.[Na+]"
        # force sanitization would return same descriptors for both of this
        fps = calc(sm)
        fps2 = calc(sm_disconnected)
        np.testing.assert_allclose(fps, fps2)

        # with the fix we should not have any value that is nan after sanitization
        # neither for Charge related or BCut2D properties
        self.assertFalse(np.isnan(fps).any())

        # we should have nan values at all bcut columns
        # if we do not standardize
        calc_nan = RDKitDescriptors2D(do_not_standardize=True)
        fps = calc_nan(sm)
        bcut_cols = [i for i, x in enumerate(calc.columns) if "bcut" in x.lower()]
        self.assertTrue(np.isnan(fps[bcut_cols]).all())

        # sanity check for large molecules
        ipc_colums = calc.columns.index("Ipc")
        fps = calc(self.EXTRA_LARGE_MOL)
        self.assertLessEqual(fps[ipc_colums], 1e3)

    def test_rdkit3d(self):
        calc = RDKitDescriptors3D()
        mol = dm.conformers.generate(dm.to_mol(self.smiles[0]))
        fps = calc(mol)
        self.assertEqual(len(fps), len(calc))

    def test_cats_2d(self):
        smiles = "Nc1cnn(-c2ccccc2)c(=O)c1Cl"
        mol = dm.to_mol(smiles)

        calc = CATS(max_dist=5, scale="raw", use_3d_distances=False)
        assert calc(mol).shape == (126,)

    def test_cats_3d_missing_conformer(self):
        smiles = "Nc1cnn(-c2ccccc2)c(=O)c1Cl"
        mol = dm.to_mol(smiles)

        calc = CATS(max_dist=5, scale="raw", use_3d_distances=True)

        with pytest.raises(ValueError):
            calc(mol)

    def test_cats_3d(self):
        smiles = "Nc1cnn(-c2ccccc2)c(=O)c1Cl"
        mol = dm.to_mol(smiles)
        mol = dm.conformers.generate(mol)

        calc = CATS(max_dist=5, scale="raw", use_3d_distances=True)
        assert calc(mol).shape == (126,)

    def test_cats_pickle(self):
        smiles = "Nc1cnn(-c2ccccc2)c(=O)c1Cl"
        mol = dm.to_mol(smiles)

        calc = CATS(max_dist=5, scale="raw", use_3d_distances=False)

        # compute fp
        fp1 = calc(mol)

        fpath = tempfile.NamedTemporaryFile().name

        # pickle
        joblib.dump(calc, fpath)

        # unpickle
        calc = joblib.load(fpath)

        # compute fp
        fp2 = calc(mol)

        # check
        assert np.allclose(fp1, fp2)

    def test_shape_descriptors(self):
        calc = USRDescriptors("usrcat")
        with self.assertRaises(ValueError) as context:
            calc(self.smiles[0])
        mol_with_conf = dm.conformers.generate(dm.to_mol(self.smiles[0]))
        out = calc(mol_with_conf)
        self.assertEqual(out.shape[-1], len(calc))

        calc2 = ElectroShapeDescriptors("mmff94")
        out2 = calc2(mol_with_conf)
        self.assertEqual(out2.shape[-1], len(calc2))

    def test_scaffkey(self):
        calc = ScaffoldKeyCalculator()
        fps = calc(self.smiles[0])
        columns = calc.columns
        col_to_check = [
            "n_atom_in_rings",
            "n_nitrogen",
            "n_heteroatoms",
            "n_ring_system",
            "n_carbon_het_carbon_het_bonds",
        ]
        expected_output = [6, 1, 4, 1, 2]
        comp_res = [fps[columns.index(x)] for x in col_to_check]
        self.assertEqual(expected_output, comp_res)

    def test_scaff_skdist(self):
        calc = ScaffoldKeyCalculator()
        smiles = dm.freesolv()["smiles"][:10].tolist()
        fps = [calc(x) for x in smiles]
        fps1 = np.asarray(fps[:6])
        fps2 = np.asarray(fps[6:])
        # compute batch
        pairwise_dist = skdistance(fps1, fps2, cdist=True)
        # compute singletons
        dist = []
        for i in range(fps2.shape[0]):
            dist.append(skdistance(fps[0], fps2[i, :], cdist=False))

        np.testing.assert_allclose(np.asarray(dist), pairwise_dist[0, :])


if __name__ == "__main__":
    ut.main()
