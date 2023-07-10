from typing import List, Optional, Union

import datamol as dm
import numpy as np
from numpy.linalg import norm
from rdkit.Chem import rdForceFieldHelpers, rdMolDescriptors, rdPartialCharges
from scipy.special import cbrt

from molfeat.calc.base import SerializableCalculator
from molfeat.utils.commons import requires_conformer


class USRDescriptors(SerializableCalculator):
    """Descriptors for the shape of a molecule.

    !!! note:
        The following shape descriptors are offered:
            * USR: UltraFast Shape Recognition
            * USRCAT: Ultrafast Shape Recognition with CREDO Atom Types
    """

    def __init__(self, method: str = "USR", replace_nan: bool = False, **kwargs):
        """Constructor for ShapeDescriptors

        Args:
            method: Shape descriptor method to use. One of 'USR', 'USRCAT'. Default to 'USR'
            replace_nan: Whether to replace nan or infinite values. Defaults to False.
        """
        self.method = method.upper()
        if self.method not in ["USR", "USRCAT"]:
            raise ValueError(f"Shape descriptor {self.method} is not supported")
        self.replace_nan = replace_nan
        self._columns = None

    def __getstate__(self):
        state = {}
        state["method"] = self.method
        state["replace_nan"] = self.replace_nan
        state["_columns"] = self._columns
        return state

    @property
    def columns(self):
        """
        Get the name of all the descriptors of this calculator
        """
        if self._columns is None:
            if self.method == "USR":
                self._columns = [f"usr-{i}" for i in range(1, 13)]
            elif self.method == "USRCAT":
                self._columns = [f"usr-{i}" for i in range(1, 61)]
        return self._columns

    def __len__(self):
        """Compute descriptors length"""
        return len(self.columns)

    @requires_conformer
    def __call__(self, mol: Union[dm.Mol, str], conformer_id: Optional[int] = -1) -> np.ndarray:
        r"""
        Get rdkit 3D descriptors for a molecule

        Args:
            mol: the molecule of interest
            conformer_id: Optional conformer id. Defaults to -1.

        Returns:
            shape_descriptors: list of computed molecular descriptors
        """
        if self.method == "USR":
            shape_descr = rdMolDescriptors.GetUSR(mol, confId=conformer_id)
        elif self.method == "USRCAT":
            shape_descr = rdMolDescriptors.GetUSRCAT(mol, confId=conformer_id)
        if self.replace_nan:
            shape_descr = np.nan_to_num(shape_descr, self.replace_nan)
        return np.asarray(shape_descr)


class ElectroShapeDescriptors(SerializableCalculator):
    """Compute Electroshape descriptors as described by

    Armstrong et al. ElectroShape: fast molecular similarity calculations incorporating shape, chirality and electrostatics.
    J Comput Aided Mol Des 24, 789-801 (2010). http://dx.doi.org/doi:10.1007/s10822-010-9374-0
    """

    SUPPORTED_CHARGE_MODELS = ["gasteiger", "tripos", "mmff94", "formal"]

    def __init__(
        self,
        charge_model: str = "gasteiger",
        replace_nan: bool = False,
        electron_scaling: float = 25.0,
        **kwargs,
    ):
        """Constructor for ElectroShape descriptor

        Args:
            charge_model: charge model to use. One of ('gasteiger', 'tripos', 'mmff94', 'formal'). Defaults to "gasteiger".
                Note that formal charges will be computed on the fly if not provided in the input molecules.
                The `tripos` charge models comes from TRIPOS force field and are often parsed from mol2 files.
            replace_nan: whether to replace NaN values. Defaults False
            electron_scaling: scaling factor to convert electron charges to Angstroms. Defaults to 25.0.
        """

        self.charge_model = charge_model
        self.replace_nan = replace_nan
        self.electron_scaling = electron_scaling
        self._columns = None

    @property
    def columns(self):
        """
        Get the name of all the descriptors of this calculator
        """
        if self._columns is None:
            self._columns = []
            for i in range(1, 6):
                self._columns.extend([f"dist-{i}-mean", f"dist-{i}-std", f"dist-{i}-crb"])

        return self._columns

    def __getstate__(self):
        state = {}
        state["charge_model"] = self.charge_model
        state["replace_nan"] = self.replace_nan
        state["electron_scaling"] = self.electron_scaling
        state["_columns"] = self._columns
        return state

    def __len__(self):
        """Return the length of the calculator"""
        return len(self.columns)

    @staticmethod
    def compute_charge(mol: Union[dm.Mol, str], charge_model: str = None):
        """
        Get the molecular charge of the molecule.

        Args:
            charge_model: charge model to use. One of ('gasteiger', 'tripos', 'mmff94', 'formal'). Defaults to "gasteiger".
        """

        if charge_model not in ElectroShapeDescriptors.SUPPORTED_CHARGE_MODELS:
            raise ValueError(
                f"Unknown charge model {charge_model}. You should provide one of {ElectroShapeDescriptors.SUPPORTED_CHARGE_MODELS}"
            )
        mol = dm.to_mol(mol)
        atom_charge = []
        atom_list = list(mol.GetAtoms())

        # force compute the partial charges if not provided
        if charge_model == "gasteiger" and not atom_list[0].HasProp("_GasteigerCharge"):
            rdPartialCharges.ComputeGasteigerCharges(mol)
        elif charge_model == "mmff94" and not atom_list[0].HasProp("_MMFF94Charge"):
            ff_infos = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            for i, atom in enumerate(atom_list):
                atom.SetDoubleProp("_MMFF94Charge", ff_infos.GetMMFFPartialCharge(i))

        for atom in mol.GetAtoms():
            if charge_model == "formal":
                atom_charge.append(atom.GetFormalCharge())
            elif charge_model == "gasteiger":
                atom_charge.append(atom.GetDoubleProp("_GasteigerCharge"))
            elif charge_model == "mmff94":
                atom_charge.append(atom.GetDoubleProp("_MMFF94Charge"))
            elif charge_model == "tripos":
                atom_charge.append(atom.GetDoubleProp("_TriposPartialCharge"))
        return np.asarray(atom_charge)

    @requires_conformer
    def __call__(self, mol: Union[dm.Mol, str], conformer_id: Optional[int] = -1):
        r"""
        Get rdkit 3D descriptors for a molecule

        Args:
            mol: the molecule of interest
            conformer_id (int, optional): Optional conformer id. Defaults to -1.

        Returns:
            shape_descriptor (np.ndarray): computed shape descriptor
        """

        mol = dm.to_mol(mol)
        coords = mol.GetConformer(conformer_id).GetPositions()
        charge = self.compute_charge(mol, self.charge_model)
        if self.replace_nan:
            charge = np.nan_to_num(charge)

        desc_4d = np.column_stack((coords, charge * self.electron_scaling))

        c1 = desc_4d.mean(axis=0)
        distances_c1 = norm(desc_4d - c1, axis=1)

        c2 = desc_4d[distances_c1.argmax()]  # atom position furthest from c1
        distances_c2 = norm(desc_4d - c2, axis=1)

        c3 = desc_4d[distances_c2.argmax()]  # atom position furthest from c2
        distances_c3 = norm(desc_4d - c3, axis=1)

        vector_a = c2 - c1
        vector_b = c3 - c1
        vector_as = vector_a[:3]  # spatial parts of these vectors
        vector_bs = vector_b[:3]  # spatial parts of these vectors
        cross_ab = np.cross(vector_as, vector_bs)
        vector_c = (norm(vector_a) / (2 * norm(cross_ab))) * cross_ab
        vector_c1s = c1[:3]

        max_charge = np.array(np.amax(charge) * self.electron_scaling)
        min_charge = np.array(np.amin(charge) * self.electron_scaling)

        c4 = np.append(vector_c1s + vector_c, max_charge)
        c5 = np.append(vector_c1s + vector_c, min_charge)

        distances_c4 = norm(desc_4d - c4, axis=1)
        distances_c5 = norm(desc_4d - c5, axis=1)

        distances_list = [
            distances_c1,
            distances_c2,
            distances_c3,
            distances_c4,
            distances_c5,
        ]

        shape_descriptor = np.zeros(15)

        i = 0
        for distances in distances_list:
            mean = np.mean(distances)
            shape_descriptor[0 + i] = mean
            shape_descriptor[1 + i] = np.std(distances)
            shape_descriptor[2 + i] = cbrt(np.sum(((distances - mean) ** 3) / distances.size))
            i += 3
        if self.replace_nan:
            return np.nan_to_num(shape_descriptor)
        return shape_descriptor


def usrdistance(
    shape_1,
    shape_2,
    weights: Optional[List[float]] = None,
):
    """Computes similarity between molecules

    Args:
        shape_1: USR shape descriptor of first molecule
        shape_2: USR shape descriptor
        weights: List of scaling factor to use for

    Returns:
        dist: Distance [0-1] between shapes of molecules, 0 indicates identical molecules
    """

    # case for usr shape descriptors
    if weights is None:
        weights = []
    if (
        (shape_1.shape[-1] == shape_2.shape[-1] == 12)
        or (shape_1.shape[-1] == shape_2.shape[-1] == 60)
        or (shape_1.shape[-1] == shape_2.shape[-1] == 15)
    ):
        dist = rdMolDescriptors.GetUSRScore(shape_1, shape_2, weights=weights)
        return dist

    raise Exception(
        "Given vectors are not valid USR shape descriptors "
        "or come from different methods. Correct vector lengths"
        "are: 12 for USR, 60 for USRCAT, 15 for Electroshape"
    )
