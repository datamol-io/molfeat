import unittest as ut
import datamol as dm
import dgl
import torch
import pytest
from molfeat.trans.graph.adj import AdjGraphTransformer
from molfeat.trans.graph.adj import DGLGraphTransformer
from molfeat.trans.graph.tree import MolTreeDecompositionTransformer
from molfeat.calc.tree import TreeDecomposer
from molfeat.utils import requires


class TestMolTreeDecomposition(ut.TestCase):
    r"""Test cases for Tree decomposition"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]
    mols = [dm.to_mol(smile) for smile in smiles]
    decomposer = TreeDecomposer(cache=False)

    def test_clusters(self):
        sm = "C1=CC=CC=C1"
        mol = dm.to_mol(sm)
        nodes, edges, frags = self.decomposer(mol)
        expected_nodes = [list(range(6))]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(edges), 0)
        self.assertSetEqual(set(nodes[0]), set(expected_nodes[0]))

        sm2 = "C12=CC=CC=C1C(=C(S2)C(=O)NC)Cl"
        mol2 = dm.to_mol(sm2)
        nodes, edges, frags = self.decomposer(mol2)
        self.assertEqual(len(nodes), 8)
        self.assertEqual(len(edges), 7)

        expected_frags = ["cC", "C", "cCl", "CN", "c1ccsc1", "C=O", "c1ccccc1"]
        self.assertSetEqual(set(frags), set(expected_frags))

    def test_moltree_transformer(self):
        transf = MolTreeDecompositionTransformer()
        transf.fit(self.mols)
        tree, ids = transf(self.smiles, ignore_errors=True)
        self.assertIsInstance(tree, list)
        self.assertTrue(isinstance(tree[0], dgl.DGLGraph))


@pytest.mark.xfail(not requires.check("dgllife"), reason="3rd party module dgllife is missing")
class TestGraphTransformer(ut.TestCase):
    r"""Test cases for AdjGraphTransformer"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
        "XXXRRR",
    ]
    mols = [dm.to_mol(x) for x in smiles]

    def test_adj_transformer(self):
        transf = AdjGraphTransformer(self_loop=True, dtype=torch.float)
        data, ids = transf(self.smiles, enforce_dtype=True, ignore_errors=True)
        # graphs and node features tuple
        self.assertEqual(len(data[0]), 2)
        self.assertEqual(ids, [0, 1, 2])
        (graphs, node_feats) = zip(*data)
        self.assertTrue(torch.is_tensor(graphs[0]))
        # get graph and node feat
        mat_sum = self.mols[0].GetNumBonds() * 2 + self.mols[0].GetNumAtoms()
        self.assertAlmostEqual(graphs[0].sum().item(), mat_sum)

    def test_dgl_transformer(self):
        transf = DGLGraphTransformer()
        graphs, ids = transf(self.smiles, ignore_errors=True)
        self.assertEqual(ids, [0, 1, 2])
        self.assertTrue(isinstance(graphs[0], dgl.DGLGraph))
        self.assertTrue(graphs[0].number_of_nodes(), self.mols[0].GetNumAtoms())

        with self.assertRaises(ValueError) as context:
            _ = transf(self.smiles, ignore_errors=False)
            self.assertTrue("transform molecule at index 3" in str(context.exception))
