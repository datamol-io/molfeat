import time
import os
import shutil
import unittest as ut
import torch
import datamol as dm
import pandas as pd
import numpy as np
import tempfile
import joblib

from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from molfeat.calc.descriptors import MordredDescriptors
from molfeat.calc.fingerprints import FPCalculator
from molfeat.calc.pharmacophore import Pharmacophore2D
from molfeat.trans import MoleculeTransformer
from molfeat.trans.base import PrecomputedMolTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.fp import FPVecFilteredTransformer
from molfeat.trans.concat import FeatConcat
from molfeat.utils.cache import DataCache, FileCache, MPDataCache


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
        trans = MoleculeTransformer(featurizer="atompair")
        smiles = dm.freesolv()["smiles"].values[:200]
        fps = trans.transform(smiles)
        batched_fps = trans.batch_transform(trans, smiles, n_jobs=-1, batch_size=100)
        np.testing.assert_allclose(fps, batched_fps)

    def test_batchify_with_cache(self):
        trans = MoleculeTransformer(featurizer="atompair")
        smiles = dm.freesolv()["smiles"].values[:200]
        fps = trans.transform(smiles)
        for cache in [DataCache(name="atompair"), MPDataCache(name="atompair")]:
            precomp = PrecomputedMolTransformer(cache, featurizer=trans)
            batched_fps = precomp.batch_transform(
                precomp, smiles, n_jobs=-1, batch_size=100, concatenate=True
            )
            self.assertTrue(smiles[0] in cache)
            self.assertTrue(len(cache) == len(fps))
            np.testing.assert_array_equal(cache[smiles[0]], fps[0])
            np.testing.assert_allclose(fps, batched_fps)

    def test_transformer(self):
        for fpkind in self.fp_list:
            trans = MoleculeTransformer(featurizer=fpkind)
            fps = trans.transform(self.smiles)
            unique_len = set([len(x) for x in fps])
            self.assertEqual(len(unique_len), 1)

    def test_transformer_parallel_mol(self):
        for fpkind in ["atompair", "pharm2D"]:
            trans1 = MoleculeTransformer(featurizer=fpkind, n_jobs=1)
            trans2 = MoleculeTransformer(featurizer=fpkind, n_jobs=-1)
            smiles = dm.freesolv()["smiles"].sample(n=2000, replace=True).values
            t0 = time.time()
            out1 = trans1.transform(smiles)
            t1 = time.time()
            out2 = trans2.transform(smiles)
            t2 = time.time()
            np.testing.assert_allclose(out1, out2)
            # we should be technically saving time with parallelization
            # self.assertLessEqual(t2 - t1, t1 - t0)

    def test_fp_parallel(self):
        for fpkind in self.fp_list:
            trans = FPVecTransformer(fpkind, length=1024, n_jobs=2)
            fps = trans.transform(self.smiles)
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
            trans = FPVecTransformer(kind="notakind", length=300)
            self.assertTrue("is not a valid" in str(context.exception))

    def test_none_mol_exception(self):
        trans = MoleculeTransformer("rdkit")
        with self.assertRaises(ValueError) as context:
            fps = trans.transform([None])
            self.assertTrue("transform molecule at" in str(context.exception))
        feat = trans.transform([None], ignore_errors=True)
        self.assertEqual(feat[0], None)

    def test_pickling(self):
        for fpkind in self.fp_list:
            trans = FPVecTransformer(fpkind, length=1024, n_jobs=2)
            smiles = dm.data.freesolv()["smiles"].values[:50]
            fps = trans.transform(smiles)
            with tempfile.NamedTemporaryFile(delete=True, suffix=f"{fpkind}.pkl") as OUT:
                joblib.dump(trans, OUT.name)
                reloaded_trans = joblib.load(OUT.name)
                reloaded_fps = reloaded_trans.transform(smiles)
                np.testing.assert_allclose(fps, reloaded_fps)

    def test_dtype_casting(self):
        trans = MoleculeTransformer("rdkit", dtype=torch.int)
        fp, ids = trans(self.smiles, enforce_dtype=True, ignore_errors=True)
        self.assertTrue(torch.is_tensor(fp))

        trans2 = MoleculeTransformer("rdkit", dtype=ExplicitBitVect)

        fp, ids = trans2(self.smiles, enforce_dtype=True, ignore_errors=True)
        self.assertTrue(type(fp[0]) == ExplicitBitVect)

    def test_3d_exception(self):
        with self.assertRaises(ValueError) as context:
            trans = MoleculeTransformer("desc3D", verbose=True)
            fp = trans.transform(self.smiles, ignore_errors=False)
            self.assertTrue("molecule with conformer" in str(context.exception))

    def test_fp_filtering(self):
        data = dm.data.freesolv().sample(n=100)
        smiles = data["smiles"].values
        trans = FPVecFilteredTransformer(
            "rdkit", length=4000, del_invariant=True, occ_threshold=0.1
        )
        trans.fit(smiles)
        out, ids = trans(smiles, ignore_errors=True)
        out2, ids2 = trans(self.smiles + ["fakemol"], ignore_errors=True)
        self.assertEqual(list(ids2), [0, 1, 2])
        self.assertEqual(len(out[0]), len(out2[0]))
        self.assertLessEqual(len(out[0]), 4000)
        self.assertEqual(len(trans), len(out[0]))

    def test_concat_callable(self):
        trans1 = FPVecTransformer("ecfp:4", length=512)
        trans2 = FPVecTransformer(
            "pharm2D",
            length=1024,
            factory="default",
            includeBondOrder=True,
            useCounts=True,
        )
        expected_length = len(trans1.featurizer) + len(trans2.featurizer)
        concat_trans = FeatConcat([trans1, trans2], dtype="df")
        self.assertEqual(expected_length, len(concat_trans.columns))
        concat_trans.fit(self.smiles)
        out, ids = concat_trans(self.smiles, enforce_dtype=True, ignore_errors=True)
        # check if dataframe
        self.assertTrue(isinstance(out, pd.DataFrame))
        self.assertEqual(expected_length, out.shape[-1])
        self.assertEqual(len(concat_trans), 2)
        self.assertEqual(concat_trans.length, expected_length)

    def test_concat_str_callable(self):
        trans1 = FPVecTransformer("maccs", length=512)
        trans2 = "rdkit"
        trans3 = "ecfp:4"
        params = {
            "ecfp:4": dict(length=2000, n_jobs=1, useChirality=False),
            "rdkit": dict(length=512),
        }
        concat_trans = FeatConcat([trans1, trans2, trans3], dtype=np.float32, params=params)
        expected_length = len(trans1) + 512 + 2000
        self.assertEqual(expected_length, len(concat_trans.columns))
        concat_trans.fit(self.smiles)
        out, ids = concat_trans(self.smiles, enforce_dtype=True, ignore_errors=True)
        # check if dataframe
        self.assertTrue(out.dtype == np.float32)
        self.assertEqual(expected_length, out.shape[-1])

    def test_concat_separator(self):
        trans = "maccs||ecfp:4||rdkit"
        concat_trans1 = FeatConcat(trans, dtype=np.float32)
        concat_trans2 = FeatConcat(trans, dtype=np.float32, params=dict(rdkit=dict(length=2000)))
        concat_trans1.fit(self.smiles)
        concat_trans2.fit(self.smiles)
        out1, ids = concat_trans1(self.smiles, enforce_dtype=True, ignore_errors=True)
        out2, ids = concat_trans1(self.smiles, enforce_dtype=True, ignore_errors=True)
        np.testing.assert_allclose(out1, out2)

    # @pytest.mark.xfail
    def test_caching(self):
        ## check performance when cache is added from existing cache
        smiles = dm.data.freesolv().smiles.values[:50]
        desc = MordredDescriptors(replace_nan=False, ignore_3D=True)
        trans = MoleculeTransformer(desc, verbose=True)
        cache = DataCache(name="mordred")

        t1 = time.time()
        out1 = trans(smiles)
        elapsed1 = time.time() - t1
        # let's cache only a part of the dataset
        cache(smiles[: len(smiles) // 2], trans)

        precomp = PrecomputedMolTransformer(cache, featurizer=trans)
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

        ## check when cache is build on the fly from a
        trans = MoleculeTransformer("desc2d")
        cache = DataCache(name="desc2d_transformer1")
        t1 = time.time()
        out1 = cache(smiles, trans)
        precomp = PrecomputedMolTransformer(cache=cache, featurizer=trans)
        elapsed1 = time.time() - t1
        out2 = precomp.transform(smiles)
        elapsed2 = time.time() - t1 - elapsed1
        out3 = precomp.transform(smiles)
        elapsed3 = time.time() - t1 - elapsed1 - elapsed2
        self.assertTrue(elapsed3 <= elapsed2 or elapsed2 < 1.5)
        self.assertTrue(elapsed3 <= elapsed1 or elapsed2 < 1.5)
        np.testing.assert_array_equal(out1, out2)
        np.testing.assert_array_equal(out1, out3)

        ## check when cache is build on the fly while the transformer name is not defined
        cache = DataCache(name="desc2d_transformer2")
        t1 = time.time()
        out1 = cache(smiles, trans)
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
            trans = PrecomputedMolTransformer(cache=reloaded_cache, featurizer=featurizer)
            feat_cache = trans(smiles_list)
            joblib.dump(trans, temp_file.name)
            trans_reloaded = joblib.load(temp_file.name)
            self.assertEqual(len(trans_reloaded.cache), len(smiles_list))
            feat_cache_reloaded = trans_reloaded(smiles_list)
            np.testing.assert_array_equal(feat_cache_reloaded, feats)
            np.testing.assert_array_equal(feat_cache_reloaded, feat_cache)

            try:
                os.unlink(parquet_out)
            except:
                shutil.rmtree(parquet_out)


if __name__ == "__main__":
    ut.main()
