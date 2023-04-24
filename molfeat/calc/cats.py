"""
CATS 2D and 3D implementation based on original work by
Rajarshi Guha <rguha@indiana.edu> 08/26/07 and Chris Arthur 1/11/2015 Rdkit port
This version modernizes the code, improve performance, add supports for 3D
as well as allowing distance binning.
see: https://masterchemoinfo.u-strasbg.fr/Documents/Conferences/Lecture1_Pharmacophores_Schneider.pdf
"""

from typing import Union
from typing import List

from collections import defaultdict as ddict
import functools

import datamol as dm
import numpy as np

from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem.rdmolops import Get3DDistanceMatrix

from molfeat.utils.datatype import to_numpy
from molfeat.calc.base import SerializableCalculator


class CATS(SerializableCalculator):
    """Cats descriptors calculator based on PPPs (potential pharmacophore points). Can be either 2D or 3D.

    !!! note:
        We need to consider all pairwise combination of the 6 PPPs described in `CATS2D.SMARTS`
        which would be $P(6,2) + 6$. However, as we only consider lexicographic order, the total size
        is then $\frac{P(6,2)}{2} + 6 = 21$, explaining the size of `CATS2D.DESCRIPTORS`

    !!! tip
        The CATS descriptor are sensitive to the number of atoms in a molecule, meaning, you would get different
        results if you add or remove hydrogen atoms

    """

    SMARTS = {
        "D": ["[!$([#6,H0,-,-2,-3])]"],
        "A": ["[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"],
        "P": ["[*+]", "[#7H2]"],
        "N": ["[*-]", "[C&$(C(=O)O),P&$(P(=O)),S&$(S(=O)O)]"],
        "L": [
            "[Cl,Br,I]",
            "[S;D2;$(S(C)(C))]",
            "[C;D2;$(C(=C)(=C))]",
            "[C;D3;$(C(=C)(C)(C))]",
            "[C;D4;$(C(C)(C)(C)(C))]",
            "[C;D3;H1;$(C(C)(C)(C))]",
            "[C;D2;H2;$(C(C)(C))]",
        ],
        "R": ["[a]"],
    }

    DESCRIPTORS = [
        "DD",
        "AD",
        "DP",
        "DN",
        "DL",
        "DR",
        "AA",
        "AP",
        "AN",
        "AL",
        "AR",
        "PP",
        "NP",
        "LP",
        "PR",
        "NN",
        "LN",
        "NR",
        "LL",
        "LR",
        "RR",
    ]

    MAX_DIST_DEFAULT_2D = 8
    MAX_DIST_DEFAULT_3D = 5

    def __init__(
        self,
        max_dist: Union[float, int] = None,
        bins: List[int] = None,
        scale: str = "raw",
        use_3d_distances: bool = False,
        **kwargs,
    ):
        """Calculator for the CATS descriptors.

        `max_dist` and `bins` will both determine the length of the fingerprint vector,
        which you can get by calling `len(calc)`

        Args:
            max_dist: Maximum distance between pairs. When set to None, the default for 2D is
                set to `max_dist=8` and for 3D to `max_dist=5`.
            bins: Bins to use. Defaults to equal spacing `[0, max_dist[`.
            scale: How to scale the values. Supported values are:
                 - 'raw' for the raw values.
                 - 'num' for values normalized by the number of atoms.
                 - 'count' for scaling based on occurence of the PPP.
            use_3d_distances: Whether to use the 3D distances instead of the topological distances.
                If set to True, the input molecules must contain a conformer.
            kwargs: silently ignored extra parameters for compatibility with other calculators.
        """

        # Set the max_dist default is set to None
        if max_dist is None:
            if use_3d_distances:
                max_dist = CATS.MAX_DIST_DEFAULT_3D
            else:
                max_dist = CATS.MAX_DIST_DEFAULT_2D

        self.max_dist = max_dist
        self.use_3d_distances = use_3d_distances

        if bins is None:
            bins = list(np.arange(1, np.floor(self.max_dist + 1), 1))

        # we don't allow interaction that exceed our distance threshold.
        bins = [x for x in bins if x <= self.max_dist]

        # we start distance indexing at 0
        if 0 not in bins:
            bins += [0]

        self.bins = list(sorted(bins))

        self.scale = scale

        self._set_columns()

    def _set_columns(self):
        self._columns = []
        for label in self.DESCRIPTORS:
            for i in range(len(self.bins)):
                self._columns.append(f"{label}.bins-{i}")

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _pattern_to_mols(cls, smarts_dict=None):
        """Convert dict of list of smarts to rdkit molecules"""

        if smarts_dict is None:
            smarts_dict = cls.SMARTS

        smarts_mols = ddict(list)
        for label, patterns in smarts_dict.items():
            patterns = [dm.from_smarts(patt) for patt in patterns]
            smarts_mols[label] = patterns

        return smarts_mols

    def _get_pcore_group(self, mol: Union[dm.Mol, str]):
        """
        Assign a PPP (potential pharmacophore points) to individual atoms of a molecule.

        !!! note
            The return value is a list of length `N_atoms` of the
            input molecule. The i'th element of the list contains
            a list of PPP labels that were identified for the i'th atom

        Args:
            mol: the molecule of interest

        Returns:
            ppp_labels (List[list]): list of all PPP labels for each atoms
        """

        smarts_mols = CATS._pattern_to_mols()

        ppp_labels = ["" for x in range(0, mol.GetNumAtoms())]
        for label, patterns in smarts_mols.items():
            for pattern in patterns:
                matched = False
                for matchbase in mol.GetSubstructMatches(pattern, uniquify=True):
                    for idx in matchbase:
                        if ppp_labels[idx] == "":
                            ppp_labels[idx] = [label]
                        else:
                            tmp = ppp_labels[idx]
                            tmp.append(label)
                            ppp_labels[idx] = tmp
                    matched = True
                if matched:
                    break
        return ppp_labels

    def _get_ppp_matrix(self, n_atoms: int, ppp_labels: List):
        """Compute PPP matrix from label list

        Args:
            n_atoms (int): number of atoms
            ppp_labels (list): PPP labels returned by

        Returns:
            pppm (dict): PPP matrix where the keys are the coordinate
        """

        pppm = {}
        for i in range(0, n_atoms):
            ppp_i = ppp_labels[i]
            if ppp_i == "":
                continue
            for j in range(0, n_atoms):
                ppp_j = ppp_labels[j]
                if ppp_j == "":
                    continue
                pairs = []
                for x in ppp_i:
                    for y in ppp_j:
                        if (x, y) not in pairs and (y, x) not in pairs:
                            ## make sure to add the labels in increasing
                            ## lexicographical order
                            if x < y:
                                tmp = (x, y)
                            else:
                                tmp = (y, x)
                            pairs.append(tmp)
                pppm[(i, j)] = pairs
        return pppm

    def _calculate(self, mol, dist_mat):
        """Calculate the CATS2D descriptors for current molecule, given a distance matrix"""

        n_atoms = mol.GetNumAtoms()
        ppp_labels = self._get_pcore_group(mol)
        ppp_mat = self._get_ppp_matrix(n_atoms, ppp_labels)

        # get the counturence of each of the PPP's
        ppp_count = dict(zip(["D", "N", "A", "P", "L", "R"], [0] * 6))
        for label in ppp_labels:
            for ppp in label:
                ppp_count[ppp] = ppp_count[ppp] + 1

        # lets calculate the CATS2D raw descriptor
        # bins: a, b, c ==> [a, b], [b, c], [c, *]
        # a is always 0
        desc = [[0 for x in range(len(self.bins))] for x in range(0, len(self.DESCRIPTORS))]
        for (x, y), labels in ppp_mat.items():
            dist = dist_mat[x, y]
            # ignore all interactions greater than the max distance we set
            # we cannot have negative distance
            if dist > self.max_dist or dist < 0:
                continue

            for pair in labels:
                idx = self.DESCRIPTORS.index(f"{pair[0]}{pair[1]}")
                vals = desc[idx]
                dist_bin = np.digitize(dist, self.bins)
                # indexing at 0
                vals[dist_bin - 1] += 1
                desc[idx] = vals

        if self.scale == "num":
            for row in range(0, len(desc)):
                for col in range(0, len(desc[0])):
                    desc[row][col] = float(desc[row][col]) / n_atoms

        elif self.scale == "count":
            #  get the scaling factors
            facs = [0] * len(self.DESCRIPTORS)
            count = 0
            for ppp in self.DESCRIPTORS:
                facs[count] = ppp_count[ppp[0]] + ppp_count[ppp[1]]
                count += 1

            # each row in desc corresponds to a PPP pair
            # so the scale factor is constant over cols of a row
            count = 0
            for i in range(0, len(desc)):
                if facs[i] == 0:
                    continue
                for j in range(0, len(desc[0])):
                    desc[i][j] = desc[i][j] / float(facs[i])

        res = []
        for row in desc:
            for col in row:
                res.append(col)
        return res

    def __len__(self):
        """Return the length of the calculator"""
        return len(self._columns)

    def __call__(self, mol: Union[dm.Mol, str], conformer_id: int = -1):
        """Get CATS 2D descriptors for a molecule

        Args:
            mol: the molecule of interest.
            conformer_id: Optional conformer id. Only relevant when `use_3d_distances`
                is set to True.

        Returns:
            props (np.ndarray): list of computed rdkit molecular descriptors
        """

        mol = dm.to_mol(mol)

        if self.use_3d_distances:
            if mol.GetNumConformers() < 1:  # type: ignore
                raise ValueError("Expected a molecule with conformers information.")

            dist_mat = Get3DDistanceMatrix(mol, confId=conformer_id)

        else:
            dist_mat = GetDistanceMatrix(mol).astype(int)

        out = self._calculate(mol, dist_mat)
        return to_numpy(out)

    @property
    def columns(self):
        """Get the descriptors columns"""
        return self._columns

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["max_dist"] = self.max_dist
        state["bins"] = self.bins
        state["scale"] = self.scale
        state["use_3d_distances"] = self.use_3d_distances
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        self.__dict__.update(state)
        self._set_columns()
