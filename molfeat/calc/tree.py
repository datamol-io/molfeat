from typing import Iterable
from typing import List
from typing import Union
from typing import Tuple
from typing import Optional
from collections import defaultdict

import os
import pathlib
import platformdirs
import fsspec
import datamol as dm

from loguru import logger
from joblib import Memory
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from rdkit.Chem import GetSymmSSSR
from rdkit.Chem import MolFragmentToSmiles

MAX_W = 1000
MST_MAX_WEIGHT = 100


class TreeDecomposer:
    """
    Molecule to tree decomposition.
    The decomposition follows algorithm 2 of https://arxiv.org/pdf/1802.04364.pdf
    """

    def __init__(self, tmp_dir: os.PathLike = None, cache: bool = True):
        """Tree Decomposer with cache system

        Args:
            tmp_dir (os.PathLike, optional): cache path. Defaults to None.
            cache (bool, optional): Whether to cache the computation. Defaults to True.
        """
        self.cache = cache
        if tmp_dir is None and cache:
            cache_dir = pathlib.Path(platformdirs.user_cache_dir(appname="molfeat"))
            tmp_dir = cache_dir / "treedecomp"
        if tmp_dir:
            self.tmp_dir = pathlib.Path(str(tmp_dir))
            self.tmp_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.tmp_dir = None
        memory = Memory(self.tmp_dir, verbose=1)
        if self.cache:
            self.decomposition_into_tree = memory.cache(
                self.decomposition_into_tree, ignore=["self", "mol"]
            )

    def decomposition_into_tree(self, mol: dm.Mol, inchikey: str = None):
        """
        Find the maximum spanning tree over all the clusters of a molecule

        Args:
            mol: The molecule of interest
            inchikey: Optional inchi key of the molecule

        Returns:
            (nodes, edges [, frags]): A tuple
            - nodes (List(List[int])): list of all the cluster formed by the decomposition Each cluster is a list of integers
                which represent the ids of the atoms in clusters.
            - edges (List(Tuple[int, int])): list of the edges.
                Each edges is a tuple of source node and destination node represented by their position in the list of clusters
            - frags (List[str]): list of strings representing the smiles of the clusters.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:  # special case
            return [[0]], [], [dm.to_smiles(mol)]

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i, atoms in enumerate(cliques):
            for atom in atoms:
                nei_list[atom].append(i)

        # Merge Rings with intersection > 2 atoms
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2:
                continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2:
                        continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i, atoms in enumerate(cliques):
            for atom in atoms:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
                # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:  # Multiple (n>2) complex rings
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

        edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        frags = [MolFragmentToSmiles(mol, cluster) for cluster in cliques]

        if len(edges) == 0:
            return cliques, edges, frags

        # Compute Maximum Spanning Tree
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
        return cliques, edges, frags

    def __call__(
        self,
        mol: Union[dm.Mol, str],
        as_smiles: bool = True,
    ):
        """
        Find the maximum spanning tree over all the clusters of a molecule

        Args:
            mol: the molecule of interest
            as_smiles: Whether to return molecular fragment as smiles or molecules
        """
        mol = dm.to_mol(mol)
        cliques, edges, frags = self.decomposition_into_tree(mol, dm.to_inchikey(mol))
        if not as_smiles:
            frags = [dm.to_mol(x) for x in frags]
        return cliques, edges, frags

    def get_vocab(
        self,
        mol_list: List[Union[str, dm.Mol]],
        output_file: Optional[os.PathLike] = None,
        log: bool = False,
    ):
        r"""
        Generate the list of all possible fragments given a set of molecules.
        This can be useful to build the vocabulary of fragments that can be found in a dataset of molecule
        before doing some learning.

        Args:
            mol_list (Iterable[dm.Mol]): A collection of molecules
            output_file: path to a file that will be used to store the generated set of fragments.
            log (bool, optional): Whether to print intermediate results to stdout

        Returns:
            res (List[str]): List of the smiles of all fragments found in the molecule collection
        """
        res = set()
        decomposer = TreeDecomposer(tmp_dir=self.tmp_dir, cache=True)
        output = dm.parallelized(decomposer, mol_list)
        for _, _, frag in output:
            res.update(frag)
        if log:
            logger.debug("A vocab of {} elements has been generated".format(len(res)))
        res = sorted(res, key=len)
        if output_file:
            with fsspec.open(output_file, "w") as outstream:
                outstream.write("\n".join(res))
        return res
