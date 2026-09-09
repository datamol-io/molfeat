import datamol as dm
import joblib
import numpy as np
import pytest
import tempfile
import unittest as ut

from molfeat.trans.pretrained import PretrainedHFTransformer
from molfeat.utils import requires

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not requires.check("transformers"), reason="3rd party module transformers is missing"
)
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
            transf2 = joblib.load(pickled_file.name)
        ori_feat = transf(dm.freesolv().smiles.values[:10])
        reloaded_feat = transf2(dm.freesolv().smiles.values[:10])
        np.testing.assert_array_equal(ori_feat, reloaded_feat)

    def test_hgf_pretrained(self):
        transf = PretrainedHFTransformer(dtype=np.float32, pooling="sum")
        fps = transf(self.smiles, enforce_dtype=True)
        fps2 = transf(self.smiles, enforce_dtype=True)
        self.assertEqual(len(fps), 3)
        self.assertEqual(len(transf), fps[0].shape[-1])
        np.testing.assert_array_equal(fps, fps2)

    def test_hgf_pretrained_cache(self):
        transf = PretrainedHFTransformer(dtype=np.float32, pooling="mean", precompute_cache=True)
        fps = transf.transform(self.smiles)
        fps2 = transf.transform(self.smiles)
        np.testing.assert_array_equal(fps, fps2)
        self.assertEqual(len(transf.precompute_cache), len(self.smiles))


if __name__ == "__main__":
    ut.main()
