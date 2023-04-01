import unittest as ut
import pytest
import torch
import numpy as np
from molfeat.utils import requires
from molfeat.trans.struct.esm import ESMProteinFingerprint
from molfeat.trans.struct.prot1D import ProtBioFingerprint


class Test_ESMProteinFingerprint(ut.TestCase):
    protein_name = "protein1"
    protein_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    expected_dim = 768

    def test_esm_rep(self):
        transformer = ESMProteinFingerprint(featurizer="esm1_t6_43M_UR50S", pooling="max")
        fps = transformer.transform(self.protein_seq)
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps[0].shape[-1], self.expected_dim)

        layers = [5, 6]
        transformer2 = ESMProteinFingerprint(
            featurizer="esm1_t6_43M_UR50S", layers=layers, pooling="max"
        )
        fps = transformer2.transform(self.protein_seq)
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps[0].shape[-1], self.expected_dim * len(layers))

    def test_esm_token_rep(self):
        transformer = ESMProteinFingerprint(featurizer="esm1_t6_43M_UR50S", pooling=None)
        fps = transformer.transform(self.protein_seq)
        self.assertEqual(fps[0].shape[-1], self.expected_dim)
        self.assertEqual(fps[0].shape[0], len(self.protein_seq) + 1)

    def test_esm_contact(self):
        transformer = ESMProteinFingerprint(
            featurizer="esm1_t6_43M_UR50S", pooling="max", contact=True
        )
        fps = transformer.transform(self.protein_seq)
        self.assertTrue(fps[0].shape[0] == fps[0].shape[-1] == len(self.protein_seq))


@pytest.mark.xfail(
    not requires.check("bio_embeddings"), reason="3rd party module bio_embeddings is missing"
)
class TestProtBioFingerprint(ut.TestCase):
    protein_seq = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRD",
    ]

    def test_unsupported_fp(self):
        with self.assertRaises(ValueError):
            transf = ProtBioFingerprint(featurizer="unsupported")
            transf(self.protein_seq)

    def test_unpooled_fp(self):
        trans1 = ProtBioFingerprint(featurizer="fasttext", pooling=None, device=torch.device("cpu"))
        trans2 = ProtBioFingerprint(featurizer="one_hot_encoding", pooling=None, device=None)
        out1 = trans1(self.protein_seq, enforce_dtype=True)
        out2 = trans2(self.protein_seq)
        self.assertIsInstance(out1, list)  # we expect list because we cannot concat
        self.assertEqual(len(out1), len(self.protein_seq))
        self.assertEqual(len(out1), len(out2))
        self.assertEqual(out1[0].shape[-1], len(trans1))

    def test_pooled_fp(self):
        transf = ProtBioFingerprint(
            featurizer="glove",
            pooling="mean",
            device=torch.device("cpu"),
            dtype=torch.float,
        )
        out1 = transf._transform(self.protein_seq[0])
        out2 = transf.transform(self.protein_seq)
        out3, _ = transf(self.protein_seq, enforce_dtype=True, ignore_errors=True)
        self.assertEqual(out3.shape[0], len(self.protein_seq))
        self.assertEqual(out3.shape[1], len(transf))
        self.assertEqual(len(out1.shape), 1)
        self.assertEqual(len(out2), len(self.protein_seq))
        self.assertTrue(torch.is_tensor(out3[0]))
        np.testing.assert_almost_equal(out2[0], out3[0].numpy().squeeze())
