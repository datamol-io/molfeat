import copy
from collections import OrderedDict
from typing import List, Optional, Union

import datamol as dm
import numpy as np
from loguru import logger
from rdkit.Chem import (
    Descriptors,
    Descriptors3D,
    FindMolChiralCenters,
    rdMolDescriptors,
    rdPartialCharges,
)
from rdkit.Chem.QED import properties

from molfeat.calc.base import SerializableCalculator
from molfeat.utils.commons import requires_conformer, requires_standardization
from molfeat.utils.datatype import to_numpy


def _charge_descriptors_computation(mol: dm.Mol):
    """Recompute properly the RDKIT 2D Descriptors related to charge

    Args:
        mol: input molecule for which the charge descriptors should be recomputed
    """
    descrs = {}
    # we disconnect the metal from the molecule then we add the hydrogen atoms and finally
    # make sure that gasteiger is recomputed. This fixes the nan and inf issues,
    # while also making sure we are closer the the proper interpretation
    mol = dm.standardize_mol(mol, disconnect_metals=True)
    mol = dm.add_hs(mol, explicit_only=False)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    atomic_charges = [float(at.GetProp("_GasteigerCharge")) for at in mol.GetAtoms()]
    atomic_charges = np.clip(atomic_charges, a_min=-500, a_max=500)
    min_charge, max_charge = np.nanmin(atomic_charges), np.nanmax(atomic_charges)
    descrs["MaxPartialCharge"] = max_charge
    descrs["MinPartialCharge"] = min_charge
    descrs["MaxAbsPartialCharge"] = max(np.abs(min_charge), np.abs(max_charge))
    descrs["MinAbsPartialCharge"] = min(np.abs(min_charge), np.abs(max_charge))
    return descrs


class RDKitDescriptors2D(SerializableCalculator):
    r"""
    Compute a list of available  rdkit 2D descriptors for a molecule.
    The descriptor calculator does not mask errors in featurization and will propagate them
    """

    DESCRIPTORS_FN = {name: fn for (name, fn) in Descriptors.descList}

    def __init__(
        self,
        replace_nan: Optional[bool] = False,
        augment: Optional[bool] = True,
        descrs: List = None,
        avg_ipc: Optional[bool] = True,
        do_not_standardize: Optional[bool] = False,
        **kwargs,
    ):
        """RDKit descriptor computation

        Args:
            replace_nan: Whether to replace nan or infinite values. Defaults to False.
            augment: Whether to augment the descriptors with some additional custom features
            descrs: Subset of available features to consider if not None
            avg_ipc: Whether to average IPC values or to use rdkit original
            do_not_standardize: Whether to force standardization of molecule before computation of the descriptor.
                Set to True if you want molfeat<=0.5.3 behaviour
        """
        self.replace_nan = replace_nan
        self.augment = augment
        self.descrs = descrs
        self.avg_ipc = avg_ipc
        self.do_not_standardize = do_not_standardize
        all_features = [d[0] for d in Descriptors.descList]
        if self.augment:
            all_features += [
                "NumAtomStereoCenters",
                "NumUnspecifiedAtomStereoCenters",
                "NumBridgeheadAtoms",
                "NumAmideBonds",
                "NumSpiroAtoms",
                "Alerts",
            ]
        if descrs is not None:
            self._columns = [x for x in descrs if x in all_features]
            unknown_descrs = set(descrs) - set(all_features)
            if len(unknown_descrs) > 0:
                logger.warning(f"Following features are not supported: {unknown_descrs}")
        else:
            self._columns = all_features

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["replace_nan"] = self.replace_nan
        state["augment"] = self.augment
        state["descrs"] = self.descrs
        state["_columns"] = self._columns

        # EN: set `avg_ipc` and `standardize`` default value to False for compat until next release
        state["avg_ipc"] = getattr(self, "avg_ipc", False)
        state["do_not_standardize"] = getattr(self, "do_not_standardize", False)
        return state

    def _compute_extra_features(self, mol: Union[dm.Mol, str]):
        """Compute the extra properties required for the augmented features version

        Args:
            mol: Input molecule

        Returns:
            props (dict): Dict of extra molecular properties
        """
        mol = copy.deepcopy(mol)
        FindMolChiralCenters(mol, force=True)
        # "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters", "NumBridgeheadAtoms", "NumAmideBonds", "NumSpiroAtoms"
        p_obj = rdMolDescriptors.Properties()
        props = OrderedDict(zip(p_obj.GetPropertyNames(), p_obj.ComputeProperties(mol)))
        # Alerts
        qed_props = properties(mol)
        props["Alerts"] = qed_props.ALERTS
        return props

    @property
    def columns(self):
        """
        Get the name of all the descriptors of this calculator
        """
        return self._columns

    def __len__(self):
        """Return the length of the calculator"""
        return len(self._columns)

    @requires_standardization(disconnect_metals=True, remove_salt=True)
    def __call__(self, mol: Union[dm.Mol, str]):
        r"""
        Get rdkit basic descriptors for a molecule

        Args:
            mol: the molecule of interest

        Returns:
            props (np.ndarray): list of computed rdkit molecular descriptors
        """
        mol = dm.to_mol(mol)
        vals = []
        props = {}
        if self.augment:
            props = self._compute_extra_features(mol)
        fixed_charge_descr = _charge_descriptors_computation(mol)
        for name in self.columns:
            val = float("nan")
            if name in fixed_charge_descr:
                val = fixed_charge_descr[name]
            elif name == "Ipc" and self.avg_ipc:  # bug fix of the rdkit IPC value
                val = self.DESCRIPTORS_FN[name](mol, avg=True)
            elif name in self.DESCRIPTORS_FN:
                val = self.DESCRIPTORS_FN[name](mol)
            elif name in props:
                val = props[name]
            else:
                raise ValueError(f"Property: {name} is not supported !")
            vals.append(val)
        vals = to_numpy(vals)
        if self.replace_nan:
            vals = np.nan_to_num(vals)
        return vals


class RDKitDescriptors3D(SerializableCalculator):
    """
    Compute a list of 3D rdkit descriptors
    """

    def __init__(
        self,
        replace_nan: bool = False,
        ignore_descrs: list = ["CalcGETAWAY"],
        **kwargs,
    ):
        """Compute 3D descriptors

        Args:
            replace_nan (bool, optional): Whether to replace nan or infinite values. Defaults to False.
            ignore_descrs (list, optional): Descriptors to ignore for performance issues. Defaults to ["CalcGETAWAY"].
        """
        self.replace_nan = replace_nan

        self._descr = [
            "CalcAsphericity",
            "CalcEccentricity",
            "CalcInertialShapeFactor",
            "CalcNPR1",
            "CalcNPR2",
            "CalcPMI1",
            "CalcPMI2",
            "CalcPMI3",
            "CalcRadiusOfGyration",
            "CalcSpherocityIndex",
            "CalcPBF",
        ]

        self.ignore_descrs = ignore_descrs or []
        self._vec_descr = [
            "CalcAUTOCORR3D",
            "CalcRDF",
            "CalcMORSE",
            "CalcWHIM",
            "CalcGETAWAY",
        ]
        self._vec_descr_length = [80, 210, 224, 114, 273]
        self._columns = [x for x in self._descr if x not in self.ignore_descrs]
        for desc, desc_len in zip(self._vec_descr, self._vec_descr_length):
            if desc in self.ignore_descrs:
                continue
            for pos in range(desc_len):
                self._columns.append(f"{desc}_{pos}")

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["replace_nan"] = self.replace_nan
        state["ignore_descrs"] = self.ignore_descrs
        state["_columns"] = self._columns
        return state

    def __len__(self):
        """Get the length of the descriptor"""
        return len(self._columns)

    @property
    def columns(self):
        """Get the descriptors columns"""
        return self._columns

    @requires_conformer
    def __call__(self, mol: Union[dm.Mol, str], conformer_id: Optional[int] = -1) -> np.ndarray:
        r"""
        Get rdkit 3D descriptors for a molecule

        Args:
            mol: the molecule of interest
            conformer_id: Optional conformer id. Defaults to -1.

        Returns:
            props: list of computed molecular descriptors
        """

        mol = dm.to_mol(mol)
        desc_val = []
        for desc in self._descr:
            val = float("nan")
            if desc not in self.ignore_descrs:
                try:
                    val = getattr(Descriptors3D.rdMolDescriptors, desc)(mol, confId=conformer_id)
                except:
                    pass
                desc_val.append(val)
        for i, desc in enumerate(self._vec_descr):
            val = [float("nan")] * self._vec_descr_length[i]
            if desc not in self.ignore_descrs:
                try:
                    val = getattr(Descriptors3D.rdMolDescriptors, desc)(mol, confId=conformer_id)
                except:
                    pass
                desc_val.extend(val)

        desc_val = to_numpy(desc_val)
        if self.replace_nan:
            desc_val = np.nan_to_num(desc_val)
        return desc_val
