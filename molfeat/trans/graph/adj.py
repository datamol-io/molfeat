from functools import partial
from typing import Optional
from typing import Callable
from typing import List
from typing import Union

import torch
import datamol as dm
import numpy as np
import torch.nn.functional as F

from loguru import logger
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from molfeat.trans.base import MoleculeTransformer
from molfeat.utils import datatype
from molfeat.utils import requires
from molfeat.utils.commons import requires_conformer
from molfeat.utils.commons import pack_graph
from molfeat.calc.atom import AtomCalculator
from molfeat.calc.bond import BondCalculator
from molfeat.calc.bond import EdgeMatCalculator

if requires.check("dgl"):
    import dgl

if requires.check("dgllife"):
    from dgllife import utils as dgllife_utils

if requires.check("torch_geometric"):
    from torch_geometric.data import Data
    from torch_geometric.loader.dataloader import Collater


class GraphTransformer(MoleculeTransformer):
    """
    Base class for all graph transformers including DGL
    """

    def __init__(
        self,
        atom_featurizer: Optional[Callable] = None,
        bond_featurizer: Optional[Callable] = None,
        explicit_hydrogens: bool = False,
        canonical_atom_order: bool = True,
        self_loop: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        dtype: Optional[Callable] = None,
        **params,
    ):
        """Mol to Graph transformer base class

        Args:
            atom_featurizer: atom featurizer to use
            bond_featurizer: atom featurizer to use
            explicit_hydrogens: Whether to use explicit hydrogen in preprocessing of the input molecule
            canonical_atom_order: Whether to use a canonical ordering of the atoms
            self_loop: Whether to add self loops or not
            n_jobs: Number of job to run in parallel. Defaults to 1.
            verbose: Verbosity level. Defaults to True.
            dtype: Output data type. Defaults to None
        """

        self._save_input_args()

        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype,
            featurizer="none",
            self_loop=self_loop,
            canonical_atom_order=canonical_atom_order,
            explicit_hydrogens=explicit_hydrogens,
            **params,
        )
        if atom_featurizer is None:
            atom_featurizer = AtomCalculator()
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self._atom_dim = None
        self._bond_dim = None

    def auto_self_loop(self):
        """Patch the featurizer to auto support self loop based on the bond featurizer characteristics"""
        bf_self_loop = None
        if self.bond_featurizer is not None:
            bf_self_loop = getattr(self.bond_featurizer, "self_loop", None)
            bf_self_loop = bf_self_loop or getattr(self.bond_featurizer, "_self_loop", None)
        if bf_self_loop is not None:
            self.self_loop = bf_self_loop

    def preprocess(self, inputs, labels=None):
        """Preprocess list of input molecules
        Args:
            labels: For compatibility
        """
        inputs, labels = super().preprocess(inputs, labels)
        new_inputs = []
        for m in inputs:
            try:
                mol = dm.to_mol(
                    m, add_hs=self.explicit_hydrogens, ordered=self.canonical_atom_order
                )
            except:
                mol = None
            new_inputs.append(mol)

        return new_inputs, labels

    def fit(self, **fit_params):
        """Fit the current transformer on given dataset."""
        if self.verbose:
            logger.error("GraphTransformer featurizers cannot be fitted !")
        return self

    @property
    def atom_dim(self):
        r"""
        Get the number of features per atom

        Returns:
            atom_dim (int): Number of atom features
        """
        if self._atom_dim is None:
            try:
                self._atom_dim = len(self.atom_featurizer)
            except:
                _toy_mol = dm.to_mol("C")
                out = self.atom_featurizer(_toy_mol)
                self._atom_dim = sum([x.shape[-1] for x in out.values()])
        return self._atom_dim

    @property
    def bond_dim(self):
        r"""
        Get the number of features for a bond

        Returns:
            bond_dim (int): Number of bond features
        """
        if self.bond_featurizer is None:
            self._bond_dim = 0
        if self._bond_dim is None:
            try:
                self._bond_dim = len(self.bond_featurizer)
            except:
                _toy_mol = dm.to_mol("CO")
                out = self.bond_featurizer(_toy_mol)
                self._bond_dim = sum([x.shape[-1] for x in out.values()])
        return self._bond_dim

    def _transform(self, mol: dm.Mol):
        r"""
        Compute features for a single molecule.
        This method would potentially need to be reimplemented by child classes

        Args:
            mol: molecule to transform into features

        Returns
            feat: featurized input molecule

        """
        raise NotImplementedError

    def __call__(self, mols: List[Union[dm.Mol, str]], ignore_errors: bool = False, **kwargs):
        r"""
        Calculate features for molecules. Using __call__, instead of transform.
        Note that most Transfomers allow you to specify
        a return datatype.

        Args:
            mols:  Mol or SMILES of the molecules to be transformed
            ignore_errors: Whether to ignore errors during featurization or raise an error.
            kwargs: Named parameters for the transform method

        Returns:
            feats: list of valid features
            ids: all valid molecule positions that did not failed during featurization
                Only returned when ignore_errors is True.

        """
        features = self.transform(mols, ignore_errors=ignore_errors, **kwargs)
        if not ignore_errors:
            return features
        features, ids = self._filter_none(features)
        return features, ids


class AdjGraphTransformer(GraphTransformer):
    r"""
    Transforms a molecule into a molecular graph representation formed by an
    adjacency matrix of atoms and a set of features for each atom (and potentially bond).
    """

    def __init__(
        self,
        atom_featurizer: Optional[Callable] = None,
        bond_featurizer: Optional[Callable] = None,
        self_loop: bool = False,
        explicit_hydrogens: bool = False,
        canonical_atom_order: bool = True,
        max_n_atoms: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = False,
        dtype: Optional[Callable] = None,
        **params,
    ):
        """
        Adjacency graph transformer

        Args:
            atom_featurizer: atom featurizer to use
            bond_featurizer: bond featurizer to use
            self_loop: whether to add self loops to the adjacency matrix. Your bond featurizer needs to supports this.
            explicit_hydrogens: Whether to use explicit hydrogen in preprocessing of the input molecule
            canonical_atom_order: Whether to use a canonical ordering of the atoms
            max_n_atoms: Maximum number of atom to set the size of the graph
            n_jobs: Number of job to run in parallel. Defaults to 1.
            verbose: Verbosity level. Defaults to True.
            dtype: Output data type. Defaults to None, where numpy arrays are returned.
        """
        super().__init__(
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            max_n_atoms=max_n_atoms,
            self_loop=self_loop,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype,
            canonical_atom_order=canonical_atom_order,
            explicit_hydrogens=explicit_hydrogens,
            **params,
        )

    def _graph_featurizer(self, mol: dm.Mol):
        """Internal adjacency graph featurizer

        Returns:
            mat : N,N matrix representing the graph
        """
        adj_mat = GetAdjacencyMatrix(mol)
        if self.self_loop:
            np.fill_diagonal(adj_mat, 1)
        return adj_mat

    @staticmethod
    def _collate_batch(batch, max_n_atoms=None, pack=False):
        """
        Collate a batch of samples. Expected format is either single graphs, e.g. a list of tuples of the form (adj, feats),
        or graphs together with their labels, where each sample is of the form ((adj, feats), label).

        Args:
             batch: list
                Batch of samples.
             max_n_atoms: Max num atoms in graphs.
             pack: Whether the graph should be packed or not into a supergraph.

        Returns:
            Collated samples.

        """
        if isinstance(batch[0], (list, tuple)) and len(batch[0]) > 2:
            graphs, feats, labels = map(list, zip(*batch))
            batched_graph = AdjGraphTransformer._collate_graphs(
                zip(graphs, feats), max_n_atoms=max_n_atoms, pack=pack
            )

            if torch.is_tensor(labels[0]):
                return batched_graph, torch.stack(labels)
            else:
                return batched_graph, labels

        # Otherwise we assume the batch is composed of single graphs.
        return AdjGraphTransformer._collate_graphs(batch, max_n_atoms=max_n_atoms, pack=pack)

    @staticmethod
    def _collate_graphs(batch, max_n_atoms, pack):
        if not all([len(b) == 2 for b in batch]):
            raise ValueError("Default collate function only supports pair of (Graph, AtomFeats) ")

        graphs, feats = zip(*batch)
        # in case someone does not convert to tensor and wants to use collate
        # who would do that ?
        graphs = [datatype.to_tensor(g) for g in graphs]
        feats = [datatype.to_tensor(f) for f in feats]
        if pack:
            return pack_graph(graphs, feats)
        else:
            if max_n_atoms is None:
                cur_max_atoms = max([x.shape[0] for x in feats])
            else:
                cur_max_atoms = max_n_atoms

            graphs = torch.stack(
                [
                    F.pad(
                        g,
                        (0, cur_max_atoms - g.shape[0], 0, cur_max_atoms - g.shape[1]),
                    )
                    for g in graphs
                ]
            )
            feats = torch.stack([F.pad(f, (0, 0, 0, cur_max_atoms - f.shape[0])) for f in feats])
        return graphs, feats

    def get_collate_fn(self, pack: bool = False, max_n_atoms: Optional[int] = None):
        """Get collate function. Adj Graph are collated either through batching
        or diagonally packing the graph into a super graph. Either a format of (batch, labels) or graph is supported.

        !!! note
            Edge features are not supported yet in the default collate because
            there is no straightforward and universal way to collate them

        Args:
            pack : Whether to pack or batch the graphs.
            max_n_atoms: Maximum number of node per graph when packing is False.
                If the graph needs to be packed and it is not set, instance attributes will be used
        """
        if self.bond_featurizer is not None:
            raise ValueError(
                "Default collate function is not supported for transformer with bond featurizer"
            )
        max_n_atoms = max_n_atoms or self.max_n_atoms

        return partial(self._collate_batch, pack=pack, max_n_atoms=max_n_atoms)

    def transform(self, mols: List[Union[dm.Mol, str]], keep_dict: bool = False, **kwargs):
        r"""
        Compute the graph featurization for a set of molecules.

        Args:
            mols: a list containing smiles or mol objects
            keep_dict: whether to keep atom and bond featurizer as dict or get the underlying data
            kwargs: arguments to pass to the `super().transform`

         Returns:
             features: a list of features for each molecule in the input set
        """
        features = super().transform(mols, **kwargs)
        if not keep_dict:
            out = []
            for i, feat in enumerate(features):
                if feat is not None:
                    graph, nodes, *bonds = feat
                    if isinstance(nodes, dict):
                        nodes = nodes[self.atom_featurizer.name]
                    if len(bonds) > 0 and isinstance(bonds[0], dict):
                        try:
                            bonds = bonds[0][self.bond_featurizer.name]
                            feat = (graph, nodes, bonds)
                        except KeyError as e:
                            # more information on failure
                            logger.error("Encountered Molecule without bonds")
                            raise e
                    else:
                        feat = (graph, nodes)
                out.append(feat)
            features = out
        return features

    def _transform(self, mol: dm.Mol):
        r"""
        Transforms a molecule into an Adjacency graph with a set of atom and optional bond features

        Args:
            mol: molecule to transform into features

        Returns
            feat: featurized input molecule (adj_mat, node_feat) or (adj_mat, node_feat, edge_feat)

        """
        if mol is None:
            return None

        try:
            adj_matrix = datatype.cast(self._graph_featurizer(mol), dtype=self.dtype)
            atom_data = self.atom_featurizer(mol, dtype=self.dtype)
            feats = (adj_matrix, atom_data)
            bond_data = None
            if self.bond_featurizer is not None:
                bond_data = self.bond_featurizer(mol, flat=False, dtype=self.dtype)
                feats = (
                    adj_matrix,
                    atom_data,
                    bond_data,
                )
        except Exception as e:
            if self.verbose:
                logger.error(e)
            feats = None
        return feats


class CompleteGraphTransformer(GraphTransformer):
    """Transforms a molecule into a complete graph"""

    def _graph_featurizer(self, mol: dm.Mol):
        """Complete grah featurizer

        Args:
            mol: molecule to transform into a graph

        Returns:
            mat : N,N matrix representing the graph
        """
        n_atoms = mol.GetNumAtoms()
        adj_mat = np.ones((n_atoms, n_atoms))
        if not self.self_loop:
            np.fill_diagonal(adj_mat, 0)
        return adj_mat


class TopoDistGraphTransformer(AdjGraphTransformer):
    """
    Graph featurizer using the topological distance between each pair
    of nodes instead of the adjacency matrix.

    The `self_loop` attribute is ignored here as the distance between an atom and itself is 0.
    """

    def _graph_featurizer(self, mol: dm.Mol):
        """Graph topological distance featurizer

        Args:
            mol: molecule to transform into a graph

        Returns:
            mat : N,N matrix representing the graph
        """
        return GetDistanceMatrix(mol)


class DistGraphTransformer3D(AdjGraphTransformer):
    """
    Graph featurizer using the 3D distance between pair of atoms for the adjacency matrix
    The `self_loop` attribute is ignored here as the distance between an atom and itself is 0.

    """

    @requires_conformer
    def _graph_featurizer(self, mol: dm.Mol):
        """Graph topological distance featurizer

        Args:
            mol: molecule to transform into a graph

        Returns:
            mat : N,N matrix representing the graph
        """
        return Get3DDistanceMatrix(mol)


class DGLGraphTransformer(GraphTransformer):
    r"""
    Transforms a molecule into a molecular graph representation formed by an
    adjacency matrix of atoms and a set of features for each atom (and potentially bond).
    """

    def __init__(
        self,
        atom_featurizer: Optional[Callable] = None,
        bond_featurizer: Optional[Callable] = None,
        self_loop: bool = False,
        explicit_hydrogens: bool = False,
        canonical_atom_order: bool = True,
        complete_graph: bool = False,
        num_virtual_nodes: int = 0,
        n_jobs: int = 1,
        verbose: bool = False,
        dtype: Optional[Callable] = None,
        **params,
    ):
        """
        Adjacency graph transformer

        Args:
           atom_featurizer: atom featurizer to use
           bond_featurizer: atom featurizer to use
           self_loop: whether to use self loop or not
           explicit_hydrogens: Whether to use explicit hydrogen in preprocessing of the input molecule
           canonical_atom_order: Whether to use a canonical ordering of the atoms
           complete_graph: Whether to use a complete graph constructor or not
           num_virtual_nodes: number of virtual nodes to add
           n_jobs: Number of job to run in parallel. Defaults to 1.
           verbose: Verbosity level. Defaults to True.
           dtype: Output data type. Defaults to None, where numpy arrays are returned.
        """

        super().__init__(
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            n_jobs=n_jobs,
            self_loop=self_loop,
            num_virtual_nodes=num_virtual_nodes,
            complete_graph=complete_graph,
            verbose=verbose,
            dtype=dtype,
            canonical_atom_order=canonical_atom_order,
            explicit_hydrogens=explicit_hydrogens,
            **params,
        )

        if not requires.check("dgllife"):
            logger.error(
                "Cannot find dgllife. It's required for some features. Please install it first !"
            )
        if not requires.check("dgl"):
            raise ValueError("Cannot find dgl, please install it first !")
        if self.dtype is not None and not datatype.is_dtype_tensor(self.dtype):
            raise ValueError("DGL featurizer only supports torch tensors currently")

    def auto_self_loop(self):
        """Patch the featurizer to auto support self loop based on the bond featurizer characteristics"""
        super().auto_self_loop()
        if isinstance(self.bond_featurizer, EdgeMatCalculator):
            self.self_loop = True

    def get_collate_fn(self, *args, **kwargs):
        """Return DGL collate function for a batch of molecular graph"""
        return self._dgl_collate

    @staticmethod
    def _dgl_collate(batch):
        """
        Batch of samples to be used with the featurizer. A sample of the batch is expected to
        be of the form (graph, label) or simply a graph.

        Args:
         batch: list
            batch of samples.

        returns:
            Batched lists of graphs and labels
        """
        if isinstance(batch[0], (list, tuple)):
            graphs, labels = map(list, zip(*batch))
            batched_graph = dgl.batch(graphs)

            if torch.is_tensor(labels[0]):
                return batched_graph, torch.stack(labels)
            else:
                return batched_graph, labels

        # Otherwise we assume the batch is composed of single graphs.
        return dgl.batch(batch)

    def _graph_featurizer(self, mol: dm.Mol):
        """Convert a molecule to a DGL graph.

        This only supports the bigraph and not any virtual nodes or complete graph.

        Args:
            mol (dm.Mol): molecule to transform into features

        Returns:
            graph (dgl.DGLGraph): graph built with dgl
        """

        n_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        graph = dgl.graph()
        graph.add_nodes(n_atoms)
        bond_src = []
        bond_dst = []
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            # set up the reverse direction
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)

        if self.self_loop:
            nodes = graph.nodes().tolist()
            bond_src.extend(nodes)
            bond_dst.extend(nodes)

        graph.add_edges(bond_src, bond_dst)
        return graph

    @property
    def atom_dim(self):
        return super(DGLGraphTransformer, self).atom_dim + int(self.num_virtual_nodes > 0)

    @property
    def bond_dim(self):
        return super(DGLGraphTransformer, self).bond_dim + int(self.num_virtual_nodes > 0)

    def _transform(self, mol: dm.Mol):
        r"""
        Transforms a molecule into an Adjacency graph with a set of atom and bond features

        Args:
            mol (dm.Mol): molecule to transform into features

        Returns
            graph (dgl.DGLGraph): a dgl graph containing atoms and bond data

        """
        if mol is None:
            return None

        graph = None
        if requires.check("dgllife"):
            graph_featurizer = dgllife_utils.mol_to_bigraph

            if self.complete_graph:
                graph_featurizer = dgllife_utils.mol_to_complete_graph
            try:
                graph = graph_featurizer(
                    mol,
                    add_self_loop=self.self_loop,
                    node_featurizer=self.__recast(self.atom_featurizer),
                    edge_featurizer=self.__recast(self.bond_featurizer),
                    canonical_atom_order=self.canonical_atom_order,
                    explicit_hydrogens=self.explicit_hydrogens,
                    num_virtual_nodes=self.num_virtual_nodes,
                )
            except Exception as e:
                if self.verbose:
                    logger.error(e)
                graph = None

        elif requires.check("dgl") and not self.complete_graph:
            # we need to build the graph ourselves.
            graph = self._graph_featurizer(mol)
            if self.atom_featurizer is not None:
                graph.ndata.update(self.atom_featurizer(mol, dtype=self.dtype))

            if self.bond_featurizer is not None:
                graph.edata.update(self.bond_featurizer(mol, dtype=self.dtype))

        else:
            raise ValueError(
                "Incorrect setup, please install missing packages (dgl, dgllife) for more features"
            )
        return graph

    def __recast(self, featurizer: Callable):
        """Recast the output of a featurizer to the transformer underlying type

        Args:
            featurizer: featurizer to patch
        """
        if featurizer is None:
            return None
        dtype = self.dtype or torch.float

        def patch_feats(*args, **kwargs):
            out_dict = featurizer(*args, **kwargs)
            out_dict = {k: datatype.cast(val, dtype=dtype) for k, val in out_dict.items()}
            return out_dict

        return patch_feats


class PYGGraphTransformer(AdjGraphTransformer):
    """Graph transformer for the PYG models"""

    def _graph_featurizer(self, mol: dm.Mol):
        # we have used bond_calculator, therefore we need to
        # go over the molecules and fetch the proper bond info from the atom idx
        if self.bond_featurizer is None or (
            isinstance(self.bond_featurizer, EdgeMatCalculator)
            or hasattr(self.bond_featurizer, "pairwise_atom_funcs")
        ):
            graph = super()._graph_featurizer(mol)
            (rows, cols) = np.nonzero(graph)
            return np.vstack((rows, cols))

        # we have a regular bond calculator here instead of all pairwise atoms
        graph = []
        for i in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(i)
            a_idx_1 = bond.GetBeginAtomIdx()
            a_idx_2 = bond.GetEndAtomIdx()
            graph += [[a_idx_1, a_idx_2], [a_idx_2, a_idx_1]]
        if getattr(self.bond_featurizer, "_self_loop", False):
            graph.extend([[atom_ind, atom_ind] for atom_ind in range(mol.GetNumAtoms())])
        graph = np.asarray(graph).T
        return graph

    def _convert_feat_to_data_point(
        self,
        graph: np.ndarray,
        node_feat: np.ndarray,
        bond_feat: Optional[np.ndarray] = None,
    ):
        """Convert extracted graph features to a pyg Data object
        Args:
            graph: graph adjacency matrix
            node_feat: node features
            bond_feat: bond features

        Returns:
            datapoint: a pyg Data object
        """
        node_feat = torch.tensor(node_feat, dtype=torch.float32)
        # construct edge index array E of shape (2, n_edges)
        graph = torch.LongTensor(graph).view(2, -1)

        if bond_feat is not None:
            bond_feat = torch.tensor(bond_feat, dtype=torch.float32)
            if bond_feat.ndim == 3:
                bond_feat = bond_feat[graph[0, :], graph[1, :]]

        d = Data(x=node_feat, edge_index=graph, edge_attr=bond_feat)
        return d

    def transform(self, mols: List[Union[dm.Mol, str]], **kwargs):
        r"""
        Compute the graph featurization for a set of molecules.

        Args:
            mols: a list containing smiles or mol objects
            kwargs: arguments to pass to the `super().transform`

         Returns:
             features: a list of Data point for each molecule in the input set
        """
        features = super().transform(mols, keep_dict=False, **kwargs)
        return [self._convert_feat_to_data_point(*feat) for feat in features]

    def get_collate_fn(
        self,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        return_pair: Optional[bool] = True,
        **kwargs,
    ):
        """
        Get collate function for pyg graphs

        Args:
            follow_batch: Creates assignment batch vectors for each key in the list. (default: :obj:`None`)
            exclude_keys: Will exclude each key in the list. (default: :obj:`None`)
            return_pair: whether to return a pair of X,y or a databatch (default: :obj:`True`)

        Returns:
            Collated samples.
        """
        collator = Collater(follow_batch=follow_batch, exclude_keys=exclude_keys)
        return partial(self._collate_batch, collator=collator, return_pair=return_pair)

    @staticmethod
    def _collate_batch(batch, collator: Callable, return_pair: bool = False, **kwargs):
        """
        Collate a batch of samples.

        Args:
            batch: Batch of samples.
            collator: collator function
            return_pair: whether to return a pair of (X,y) a databatch
        Returns:
            Collated samples.
        """
        if isinstance(batch[0], (list, tuple)) and len(batch[0]) > 1:
            graphs, labels = map(list, zip(*batch))
            for graph, label in zip(graphs, labels):
                graph.y = label
            batch = graphs
        batch = collator(batch)
        if return_pair:
            return (batch, batch.y)
        return batch
