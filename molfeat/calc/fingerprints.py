from typing import Union
from typing import Optional

import copy
import datamol as dm
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdReducedGraphs
from rdkit.Chem import rdmolops
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter

from loguru import logger
from molfeat.calc._mhfp import SECFP
from molfeat.calc._map4 import MAP4
from molfeat.calc._serializable_classes import (
    SerializableMorganFeatureAtomInvGen,
    SERIALIZABLE_CLASSES,
)
from molfeat.calc.base import SerializableCalculator
from molfeat.utils.datatype import to_numpy, to_fp
from molfeat.utils.commons import fold_count_fp


FP_GENERATORS = {
    "ecfp": rdFingerprintGenerator.GetMorganGenerator,
    "fcfp": rdFingerprintGenerator.GetMorganGenerator,
    "topological": rdFingerprintGenerator.GetTopologicalTorsionGenerator,
    "atompair": rdFingerprintGenerator.GetAtomPairGenerator,
    "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator,
    "ecfp-count": rdFingerprintGenerator.GetMorganGenerator,
    "fcfp-count": rdFingerprintGenerator.GetMorganGenerator,
    "topological-count": rdFingerprintGenerator.GetTopologicalTorsionGenerator,
    "atompair-count": rdFingerprintGenerator.GetAtomPairGenerator,
    "rdkit-count": rdFingerprintGenerator.GetRDKitFPGenerator,
}

FP_FUNCS = {
    "maccs": rdMolDescriptors.GetMACCSKeysFingerprint,
    "avalon": pyAvalonTools.GetAvalonFP,
    "pattern": rdmolops.PatternFingerprint,
    "layered": rdmolops.LayeredFingerprint,
    "map4": MAP4,
    "secfp": SECFP,
    "erg": rdReducedGraphs.GetErGFingerprint,
    "estate": lambda x, **params: EStateFingerprinter.FingerprintMol(x)[0],
    "avalon-count": pyAvalonTools.GetAvalonCountFP,
    **FP_GENERATORS,
}


FP_DEF_PARAMS = {
    "maccs": {},
    "avalon": {
        "nBits": 512,
        "isQuery": False,
        "resetVect": False,
        "bitFlags": pyAvalonTools.avalonSimilarityBits,
    },
    "ecfp": {
        "radius": 2,  # ECFP4
        "fpSize": 2048,
        "includeChirality": False,
        "useBondTypes": True,
        "countSimulation": False,
        "countBounds": None,
        "atomInvariantsGenerator": None,
        "bondInvariantsGenerator": None,
    },
    "fcfp": {
        "radius": 2,
        "fpSize": 2048,
        "includeChirality": False,
        "useBondTypes": True,
        "countSimulation": False,
        "countBounds": None,
        "atomInvariantsGenerator": SerializableMorganFeatureAtomInvGen(),
        "bondInvariantsGenerator": None,
    },
    "topological": {
        "includeChirality": False,
        "torsionAtomCount": 4,
        "countSimulation": True,
        "countBounds": None,
        "fpSize": 2048,
        "atomInvariantsGenerator": None,
    },
    "atompair": {
        "minDistance": 1,
        "maxDistance": 30,
        "includeChirality": False,
        "use2D": True,
        "countSimulation": True,
        "countBounds": None,
        "fpSize": 2048,
        "atomInvariantsGenerator": None,
    },
    "rdkit": {
        "minPath": 1,
        "maxPath": 7,
        "useHs": True,
        "branchedPaths": True,
        "useBondOrder": True,
        "countSimulation": False,
        "countBounds": None,
        "fpSize": 2048,
        "numBitsPerFeature": 2,
        "atomInvariantsGenerator": None,
    },
    "pattern": {
        "fpSize": 2048,
        "atomCounts": [],
        "setOnlyBits": None,
        "tautomerFingerprints": False,
    },
    "layered": {
        "fpSize": 2048,
        "minPath": 1,
        "maxPath": 7,
        "atomCounts": [],
        "setOnlyBits": None,
        "branchedPaths": True,
        "fromAtoms": 0,
    },
    "map4": {"dimensions": 2048, "radius": 2},
    "secfp": {
        "n_permutations": 128,
        "nBits": 2048,
        "radius": 3,
        "min_radius": 1,
        "rings": True,
        "kekulize": False,
        "isomeric": False,
        "seed": 0,
    },
    "mhfp": {
        "n_permutations": 128,
        "radius": 3,
        "min_radius": 1,
        "rings": True,
        "kekulize": False,
        "isomeric": False,
        "seed": 0,
    },
    "erg": {"atomTypes": 0, "fuzzIncrement": 0.3, "minPath": 1, "maxPath": 15},
    "estate": {},
    # COUNTING FP
    "ecfp-count": {
        "radius": 2,  # ECFP4
        "fpSize": 2048,
        "includeChirality": False,
        "useBondTypes": True,
        "includeRedundantEnvironments": False,
        "countBounds": None,
        "atomInvariantsGenerator": None,
        "bondInvariantsGenerator": None,
    },
    "fcfp-count": {
        "radius": 2,
        "fpSize": 2048,
        "includeChirality": False,
        "useBondTypes": True,
        "includeRedundantEnvironments": False,
        "atomInvariantsGenerator": SerializableMorganFeatureAtomInvGen(),
        "bondInvariantsGenerator": None,
    },
    "topological-count": {
        "includeChirality": False,
        "torsionAtomCount": 4,
        "countSimulation": True,
        "countBounds": None,
        "fpSize": 2048,
        "atomInvariantsGenerator": None,
    },
    "avalon-count": {
        "nBits": 512,
        "isQuery": False,
        "bitFlags": pyAvalonTools.avalonSimilarityBits,
    },    
    "atompair-count": {
        "minDistance": 1,
        "maxDistance": 30,
        "includeChirality": False,
        "use2D": True,
        "countSimulation": True,
        "countBounds": None,
        "fpSize": 2048,
        "atomInvariantsGenerator": None,
    },
    "rdkit-count": {
        "minPath": 1,
        "maxPath": 7,
        "useHs": True,
        "branchedPaths": True,
        "useBondOrder": True,
        "countSimulation": False,
        "countBounds": None,
        "fpSize": 2048,
        "numBitsPerFeature": 1,
        "atomInvariantsGenerator": None,
    },
}


class FPCalculator(SerializableCalculator):
    """Fingerprint bit calculator for a molecule"""

    def __init__(
        self,
        method: str,
        length: Optional[int] = None,
        counting: bool = False,
        **method_params,
    ):
        """Compute the given fingeprint for a molecule

        !!! note
            For efficiency reason, count fingerprints are hashed and potentially
            re-folded and the count corresponds to the number of bits set to true

        Args:
            method (str): Name of the fingerprint method to use. See FPCalculator.available_fingerprints() for a list
            length (int, optional): Length of the fingerprint. Defaults to None.
                The default corresponds to the fingerpint default.
            counting (bool, optional): Whether to use the count version of the fingerprint
            method_params (dict): any parameters to the fingerprint algorithm.
                See FPCalculator.default_parameters(method) for all the parameters required by a given method.
        """
        self.method = method.lower()
        self.counting = counting or "-count" in self.method
        if self.counting and "-count" not in self.method:
            self.method = self.method + "-count"
        self.input_length = length
        if self.method not in FP_FUNCS:
            raise ValueError(f"Method {self.method} is not a supported featurizer")
        default_params = copy.deepcopy(FP_DEF_PARAMS[method])
        unknown_params = set(method_params.keys()).difference(set(default_params.keys()))
        if unknown_params:
            logger.error(f"Params: {unknown_params} are not valid for {method}")
        self.params = default_params
        self.params.update(
            {k: method_params[k] for k in method_params if k in default_params.keys()}
        )
        self._length = self._set_length(length)

    @staticmethod
    def available_fingerprints():
        """Get the list of available fingerprints"""
        return list(FP_FUNCS.keys())

    @staticmethod
    def default_parameters(method: str):
        """Get the default parameters for a given fingerprint method

        Args:
            method: name of the fingerprint method
        """
        return FP_DEF_PARAMS[method].copy()

    @property
    def columns(self):
        """
        Get the name of all the descriptors of this calculator
        """
        return [f"fp_{i}" for i in range(self._length)]

    def __len__(self):
        """Return the length of the calculator"""
        return self._length

    def _set_length(self, length=None):
        """Get the length of the featurizer"""
        fplen = length
        len_key = None
        if self.method == "maccs":
            fplen = 167
        elif self.method == "estate":
            fplen = 79
        elif self.method == "erg":
            fplen = 315
        elif self.method == "rdkit-count" and not fplen:
            fplen = 2048
        elif "nBits" in self.params.keys():
            len_key = "nBits"
            fplen = self.params[len_key]
        elif "n_permutations" in self.params.keys():
            # special case for mhfp
            len_key = "n_permutations"
            fplen = self.params[len_key]
        elif "fpSize" in self.params.keys():
            len_key = "fpSize"
            fplen = self.params[len_key]
        elif "dimensions" in self.params.keys():
            len_key = "dimensions"
            fplen = self.params[len_key]
        if len_key is not None and length:
            self.params[len_key] = length
            fplen = length
        return fplen

    def __call__(self, mol: Union[dm.Mol, str], raw: bool = False):
        r"""
        Compute the Fingerprint of a molecule

        Args:
            mol: the molecule of interest
            raw: whether to keep original datatype or convert to numpy. Useful for rdkit's similarity functions

        Returns:
            props (np.ndarray): list of computed rdkit molecular descriptors
        """
        mol = dm.to_mol(mol)

        fp_func = FP_FUNCS[self.method]
        if self.method in FP_GENERATORS:
            fp_func = fp_func(**self.params)
            if self.counting:
                fp_val = fp_func.GetCountFingerprint(mol)
            else:
                fp_val = fp_func.GetFingerprint(mol)
        else:
            fp_val = fp_func(mol, **self.params)
        if self.counting:
            fp_val = fold_count_fp(fp_val, self._length)
        if not raw:
            fp_val = to_numpy(fp_val)
        if self.counting and raw:
            # converint the counted values to SparseInt again
            fp_val = to_fp(fp_val, bitvect=False)
        return fp_val

    def __getstate__(self):
        # EN: note that the state is standardized with all the parameter
        # because of the possibility of default changing after
        state = {}
        state["length"] = self.input_length
        state["input_length"] = self.input_length
        state["method"] = self.method
        state["counting"] = self.counting
        state["params"] = {
            k: (v if v.__class__.__name__ not in SERIALIZABLE_CLASSES else v.__class__.__name__)
            for k, v in self.params.items()
        }
        return state

    def __setstate__(self, state: dict):
        """Set the state of the featurizer"""
        self.__dict__.update(state)
        self.params = {
            k: (v if v not in SERIALIZABLE_CLASSES else SERIALIZABLE_CLASSES[v]())
            for k, v in self.params.items()
        }
        self._length = self._set_length(self.input_length)

    def to_state_dict(self):
        """Get the state dictionary"""
        state_dict = super().to_state_dict()
        cur_params = self.params
        default_params = copy.deepcopy(FP_DEF_PARAMS[state_dict["args"]["method"]])

        state_dict["args"].update(
            {
                k: (
                    cur_params[k]
                    if cur_params[k].__class__.__name__ not in SERIALIZABLE_CLASSES
                    else cur_params[k].__class__.__name__
                )
                for k in cur_params
                if (cur_params[k] != default_params[k] and cur_params[k] is not None)
            }
        )
        # we want to keep all the additional parameters in the state dict
        return state_dict
