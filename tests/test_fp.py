import os
import shutil
import tempfile
import time
import unittest as ut

import datamol as dm
import joblib
import numpy as np
import pandas as pd
import torch
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from molfeat.calc import RDKitDescriptors2D, SerializableCalculator
from molfeat.calc.fingerprints import FPCalculator
from molfeat.calc.pharmacophore import Pharmacophore2D
from molfeat.trans import MoleculeTransformer
from molfeat.trans.base import PrecomputedMolTransformer
from molfeat.trans.concat import FeatConcat
from molfeat.trans.fp import FPVecFilteredTransformer, FPVecTransformer
from molfeat.utils.cache import DataCache, FileCache, MPDataCache


class CustomBatchCalculator(SerializableCalculator):
    def __init__(self, random_seed: int = 42, length: int = 10):
        self.random_seed = random_seed
        self.length = length
        self.rng = np.random.RandomState(self.random_seed)

    def __call__(self, mol, **kwargs):
        return self.rng.randn(self.length)

    def __len__(self):
        return self.length

    def batch_compute(self, mols, **kwargs):
        return self.rng.randn(len(mols), self.length)


class TestMolTransformer(ut.TestCase):
    r"""Test cases for FingerprintsTransformer"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]
    fp_list = [
        "atompair",
        "topological",
        "avalon",
        "rdkit",
        "ecfp",
        "pharm2D",
        "desc2D",
    ]

    def test_batchify(self):
        transf = MoleculeTransformer(featurizer="atompair")
        smiles = dm.freesolv()["smiles"].values[:200]
        fps = transf.transform(smiles)
        batched_fps = transf.batch_transform(transf, smiles, n_jobs=-1, batch_size=100)
        np.testing.assert_allclose(fps, batched_fps)

    def test_batchify_with_cache(self):
        transf = MoleculeTransformer(featurizer="atompair")
        smiles = dm.freesolv()["smiles"].values[:200]
        fps = transf.transform(smiles)
        for cache in [DataCache(name="atompair"), MPDataCache(name="atompair")]:
            precomp = PrecomputedMolTransformer(cache, featurizer=transf)
            batched_fps = precomp.batch_transform(
                precomp, smiles, n_jobs=-1, batch_size=100, concatenate=True
            )
            self.assertTrue(smiles[0] in cache)
            self.assertTrue(len(cache) == len(fps))
            np.testing.assert_array_equal(cache[smiles[0]], fps[0])
            np.testing.assert_allclose(fps, batched_fps)

    def test_custom_calculator(self):
        calc = CustomBatchCalculator()
        transf = MoleculeTransformer(featurizer=calc)
        fps = transf.transform(self.smiles)
        self.assertEqual(len(fps), len(self.smiles))
        self.assertEqual(len(fps[0]), len(calc))

    def test_transformer(self):
        for fpkind in self.fp_list:
            transf = MoleculeTransformer(featurizer=fpkind)
            fps = transf.transform(self.smiles)
            unique_len = set([len(x) for x in fps])
            self.assertEqual(len(unique_len), 1)

    def test_transformer_parallel_mol(self):
        for fpkind in ["atompair", "pharm2D"]:
            transf1 = MoleculeTransformer(featurizer=fpkind, n_jobs=1)
            transf2 = MoleculeTransformer(featurizer=fpkind, n_jobs=-1)
            smiles = dm.freesolv()["smiles"].sample(n=2000, replace=True).values
            t0 = time.time()
            out1 = transf1.transform(smiles)
            t1 = time.time()
            out2 = transf2.transform(smiles)
            t2 = time.time()
            np.testing.assert_allclose(out1, out2)
            # we should be technically saving time with parallelization
            # self.assertLessEqual(t2 - t1, t1 - t0)

    def test_fp_parallel(self):
        for fpkind in self.fp_list:
            transf = FPVecTransformer(fpkind, length=1024, n_jobs=2)
            fps = transf.transform(self.smiles)
            unique_len = set([len(x) for x in fps])
            self.assertEqual(len(unique_len), 1)

        featurizer1 = FPCalculator(
            method="ecfp",
            radius=2,
            nBits=2048,
            invariants=[],
            fromAtoms=[],
            useChirality=False,
            useBondTypes=True,
            useFeatures=False,
        )
        featurizer2 = Pharmacophore2D(factory="default", length=2000)
        for featurizer in [featurizer1, featurizer2]:
            transformer = MoleculeTransformer(
                featurizer=featurizer,
                n_jobs=-1,
                parallel_kwargs={"scheduler": "processes"},
            )
            transformer(self.smiles)

    def test_unknown_kind_exception(self):
        with self.assertRaises(ValueError) as context:
            transf = FPVecTransformer(kind="notakind", length=300)
            self.assertTrue("is not a valid" in str(context.exception))

    def test_none_mol_exception(self):
        transf = MoleculeTransformer("rdkit")
        with self.assertRaises(ValueError) as context:
            fps = transf.transform([None])
            self.assertTrue("transform molecule at" in str(context.exception))
        feat = transf.transform([None], ignore_errors=True)
        self.assertEqual(feat[0], None)

    def test_pickling(self):
        for fpkind in self.fp_list:
            transf = FPVecTransformer(fpkind, length=1024, n_jobs=2)
            smiles = dm.data.freesolv()["smiles"].values[:50]
            fps = transf.transform(smiles)
            with tempfile.NamedTemporaryFile(delete=True, suffix=f"{fpkind}.pkl") as OUT:
                joblib.dump(transf, OUT.name)
                reloaded_transf = joblib.load(OUT.name)
                reloaded_fps = reloaded_transf.transform(smiles)
                np.testing.assert_allclose(fps, reloaded_fps)

    def test_dtype_casting(self):
        transf = MoleculeTransformer("rdkit", dtype=torch.int)
        fp, ids = transf(self.smiles, enforce_dtype=True, ignore_errors=True)
        self.assertTrue(torch.is_tensor(fp))

        transf2 = MoleculeTransformer("rdkit", dtype=ExplicitBitVect)

        fp, ids = transf2(self.smiles, enforce_dtype=True, ignore_errors=True)
        self.assertTrue(type(fp[0]) == ExplicitBitVect)

    def test_3d_exception(self):
        with self.assertRaises(ValueError) as context:
            transf = MoleculeTransformer("desc3D", verbose=True)
            fp = transf.transform(self.smiles, ignore_errors=False)
            self.assertTrue("molecule with conformer" in str(context.exception))

    def test_fp_filtering(self):
        data = dm.data.freesolv().sample(n=100)
        smiles = data["smiles"].values
        transf = FPVecFilteredTransformer(
            "rdkit", length=4000, del_invariant=True, occ_threshold=0.1
        )
        transf.fit(smiles)
        out, ids = transf(smiles, ignore_errors=True)
        out2, ids2 = transf(self.smiles + ["fakemol"], ignore_errors=True)
        self.assertEqual(list(ids2), [0, 1, 2])
        self.assertEqual(len(out[0]), len(out2[0]))
        self.assertLessEqual(len(out[0]), 4000)
        self.assertEqual(len(transf), len(out[0]))

    def test_concat_callable(self):
        transf1 = FPVecTransformer("ecfp:4", length=512)
        transf2 = FPVecTransformer(
            "pharm2D",
            length=1024,
            factory="default",
            includeBondOrder=True,
            useCounts=True,
        )
        expected_length = len(transf1.featurizer) + len(transf2.featurizer)
        concat_transf = FeatConcat([transf1, transf2], dtype="df")
        self.assertEqual(expected_length, len(concat_transf.columns))
        concat_transf.fit(self.smiles)
        out, ids = concat_transf(self.smiles, enforce_dtype=True, ignore_errors=True)
        # check if dataframe
        self.assertTrue(isinstance(out, pd.DataFrame))
        self.assertEqual(expected_length, out.shape[-1])
        self.assertEqual(len(concat_transf), 2)
        self.assertEqual(concat_transf.length, expected_length)

    def test_concat_str_callable(self):
        transf1 = FPVecTransformer("maccs", length=512)
        transf2 = "rdkit"
        transf3 = "ecfp:4"
        params = {
            "ecfp:4": dict(length=2000, n_jobs=1, useChirality=False),
            "rdkit": dict(length=512),
        }
        concat_transf = FeatConcat([transf1, transf2, transf3], dtype=np.float32, params=params)
        expected_length = len(transf1) + 512 + 2000
        self.assertEqual(expected_length, len(concat_transf.columns))
        concat_transf.fit(self.smiles)
        out, ids = concat_transf(self.smiles, enforce_dtype=True, ignore_errors=True)
        # check if dataframe
        self.assertTrue(out.dtype == np.float32)
        self.assertEqual(expected_length, out.shape[-1])

    def test_concat_separator(self):
        transf = "maccs||ecfp:4||rdkit"
        concat_transf1 = FeatConcat(transf, dtype=np.float32)
        concat_transf2 = FeatConcat(transf, dtype=np.float32, params=dict(rdkit=dict(length=2000)))
        concat_transf1.fit(self.smiles)
        concat_transf2.fit(self.smiles)
        out1, ids = concat_transf1(self.smiles, enforce_dtype=True, ignore_errors=True)
        out2, ids = concat_transf1(self.smiles, enforce_dtype=True, ignore_errors=True)
        np.testing.assert_allclose(out1, out2)

    def test_caching(self):
        # check performance when cache is added from existing cache
        smiles = dm.data.freesolv().smiles.values[:50]
        desc = RDKitDescriptors2D(replace_nan=False, ignore_3D=True)
        transff = MoleculeTransformer(desc, verbose=True)
        cache = DataCache(name="rdkit2d")

        t1 = time.time()
        out1 = transff(smiles)
        elapsed1 = time.time() - t1
        # let's cache only a part of the dataset
        cache(smiles[: len(smiles) // 2], transff)

        precomp = PrecomputedMolTransformer(cache, featurizer=transff)
        t2 = time.time()
        out2 = precomp(smiles)
        elapsed2 = time.time() - t2

        # check length
        self.assertEqual(len(precomp), len(desc))
        # should be faster now.
        # disable check on very fast run
        self.assertTrue(elapsed2 <= elapsed1 or elapsed1 < 1.5)
        np.testing.assert_array_equal(out1, out2)

        # should be even faster now that the full dataset is cached
        t3 = time.time()
        out3 = precomp(smiles)
        elapsed3 = time.time() - t3
        self.assertTrue(elapsed3 <= elapsed1 or elapsed1 < 1.5)
        np.testing.assert_array_equal(out1, out3)

        # check when cache is build on the fly from a
        transff = MoleculeTransformer("desc2d")
        cache = DataCache(name="desc2d_transformer1")
        t1 = time.time()
        out1 = cache(smiles, transff)
        precomp = PrecomputedMolTransformer(cache=cache, featurizer=transff)
        elapsed1 = time.time() - t1
        out2 = precomp.transform(smiles)
        elapsed2 = time.time() - t1 - elapsed1
        out3 = precomp.transform(smiles)
        elapsed3 = time.time() - t1 - elapsed1 - elapsed2
        self.assertTrue(elapsed3 <= elapsed2 or elapsed2 < 1.5)
        self.assertTrue(elapsed3 <= elapsed1 or elapsed2 < 1.5)
        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)

        # check when cache is build on the fly while the transformer name is not defined
        cache = DataCache(name="desc2d_transformer2")
        t1 = time.time()
        out1 = cache(smiles, transff)
        precomp = PrecomputedMolTransformer(cache=cache, featurizer="desc2d")
        elapsed1 = time.time() - t1
        out2 = precomp.transform(smiles)
        elapsed2 = time.time() - t1 - elapsed1
        np.testing.assert_array_equal(out1, out2)

    def test_cached_featurizer_pickling(self):
        featurizer = FPVecTransformer(kind="desc2D")
        smiles_list = dm.data.freesolv()["smiles"].values[:50]
        feats = featurizer(smiles_list)
        with tempfile.NamedTemporaryFile(delete=True, suffix="pkl") as temp_file:
            cache = FileCache(None, file_type="pkl")
            _ = cache(smiles_list, featurizer)
            parquet_out = temp_file.name + ".parquet"
            cache.save_to_file(parquet_out, file_type="parquet")
            reloaded_cache = FileCache.load_from_file(parquet_out, file_type="parquet")
            transff = PrecomputedMolTransformer(cache=reloaded_cache, featurizer=featurizer)
            feat_cache = transff(smiles_list)
            joblib.dump(transff, temp_file.name)
            transf_reloaded = joblib.load(temp_file.name)
            self.assertEqual(len(transf_reloaded.cache), len(smiles_list))
            feat_cache_reloaded = transf_reloaded(smiles_list)
            np.testing.assert_array_equal(feat_cache_reloaded, feats)
            np.testing.assert_array_equal(feat_cache_reloaded, feat_cache)

            try:
                os.unlink(parquet_out)
            except:
                shutil.rmtree(parquet_out)


if __name__ == "__main__":
    ut.main()
