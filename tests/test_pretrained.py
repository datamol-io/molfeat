import unittest as ut
import time
import numpy as np
import datamol as dm
import torch
import pytest
import tempfile
import joblib
from molfeat.trans.pretrained import GraphormerTransformer
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained import PretrainedHFTransformer
from molfeat.utils import requires
from molfeat.utils.cache import DataCache


@pytest.mark.xfail(
    not requires.check("graphormer_pretrained"), reason="3rd party module graphormer is missing"
)
class TestGraphormerTransformer(ut.TestCase):
    r"""Test cases for FingerprintsTransformer"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]

    def test_graphormer_pickling(self):
        transf = GraphormerTransformer(dtype=np.float32, pooling="sum")
        with tempfile.NamedTemporaryFile(delete=True) as pickled_file:
            joblib.dump(transf, pickled_file.name)
            trans2 = joblib.load(pickled_file.name)
        ori_feat = transf(dm.freesolv().smiles.values[:10])
        reloaded_feat = trans2(dm.freesolv().smiles.values[:10])
        np.testing.assert_array_equal(ori_feat, reloaded_feat)

    def test_graphormer(self):
        transf = GraphormerTransformer(dtype=np.float32, pooling="sum")
        trans2 = GraphormerTransformer(dtype=np.float32, pooling="mean")
        fps = transf(self.smiles, enforce_dtype=True)
        fps2 = transf(self.smiles, enforce_dtype=True)
        fps3 = trans2(self.smiles, enforce_dtype=True)
        self.assertEqual(len(fps), 3)
        self.assertEqual(len(transf), fps[0].shape[-1])
        np.testing.assert_array_equal(fps, fps2)
        node_per_mol = np.rint(fps / fps3)
        node_per_mol = np.unique(node_per_mol)
        self.assertLessEqual(len(node_per_mol), 2)

    def test_graphormer_cache(self):
        transf = GraphormerTransformer(
            dtype=np.float32, pooling="mean", max_length=25, precompute_cache=True
        )
        t0 = time.time()
        time_buffer = 1
        fps = transf.transform(self.smiles)
        ori_run = time.time() - t0
        fps2 = transf.transform(self.smiles)
        cached_run = time.time() - t0 - ori_run
        np.testing.assert_array_equal(fps, fps2)
        # add buffers
        self.assertLessEqual(cached_run, ori_run + time_buffer)


class TestDGLTransformer(ut.TestCase):
    r"""Test cases for FingerprintsTransformer"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]

    def test_dgl_pickling(self):
        transf = PretrainedDGLTransformer(dtype=np.float32, pooling="sum")
        with tempfile.NamedTemporaryFile(delete=True) as pickled_file:
            joblib.dump(transf, pickled_file.name)
            trans2 = joblib.load(pickled_file.name)
        ori_feat = transf(dm.freesolv().smiles.values[:10])
        reloaded_feat = trans2(dm.freesolv().smiles.values[:10])
        np.testing.assert_array_equal(ori_feat, reloaded_feat)

    def test_dgl_pretrained(self):
        transf = PretrainedDGLTransformer(dtype=np.float32, pooling="sum")
        fps = transf(self.smiles, enforce_dtype=True)
        fps2 = transf(self.smiles, enforce_dtype=True)
        self.assertEqual(len(fps), 3)
        self.assertEqual(len(transf), fps[0].shape[-1])
        np.testing.assert_array_equal(fps, fps2)

    @pytest.mark.xfail(reason="Cache might not be faster")
    def test_dgl_pretrained_cache(self):
        transf = PretrainedDGLTransformer(
            dtype=np.float32, pooling="mean", max_length=25, precompute_cache=True
        )
        t0 = time.time()
        time_buffer = 1
        fps = transf.transform(self.smiles)
        ori_run = time.time() - t0
        fps2 = transf.transform(self.smiles)
        cached_run = time.time() - t0 - ori_run
        np.testing.assert_array_equal(fps, fps2)
        # add buffers
        self.assertLessEqual(cached_run, ori_run + time_buffer)


class TestHGFTransformer(ut.TestCase):
    r"""Test cases for FingerprintsTransformer"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]

    def test_hgf_pickling(self):
        transf = PretrainedHFTransformer(dtype=np.float32, pooling="sum")
        with tempfile.NamedTemporaryFile(delete=True) as pickled_file:
            joblib.dump(transf, pickled_file.name)
            trans2 = joblib.load(pickled_file.name)
        ori_feat = transf(dm.freesolv().smiles.values[:10])
        reloaded_feat = trans2(dm.freesolv().smiles.values[:10])
        np.testing.assert_array_equal(ori_feat, reloaded_feat)

    def test_hgf_pretrained(self):
        transf = PretrainedHFTransformer(dtype=np.float32, pooling="sum")
        fps = transf(self.smiles, enforce_dtype=True)
        fps2 = transf(self.smiles, enforce_dtype=True)
        self.assertEqual(len(fps), 3)
        self.assertEqual(len(transf), fps[0].shape[-1])
        np.testing.assert_array_equal(fps, fps2)

    @pytest.mark.xfail(reason="Cache might not be faster")
    def test_hgf_pretrained_cache(self):
        transf = PretrainedHFTransformer(dtype=np.float32, pooling="mean", precompute_cache=True)
        t0 = time.time()
        time_buffer = 1
        fps = transf.transform(self.smiles)
        ori_run = time.time() - t0
        fps2 = transf.transform(self.smiles)
        cached_run = time.time() - t0 - ori_run
        np.testing.assert_array_equal(fps, fps2)
        # add buffers
        self.assertLessEqual(cached_run, ori_run + time_buffer)


if __name__ == "__main__":
    ut.main()
