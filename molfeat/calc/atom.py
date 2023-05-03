from typing import Union
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional

from collections import defaultdict
from functools import partial
from functools import lru_cache

import inspect
import importlib
import os
import datamol as dm
import numpy as np

from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import GetSymmSSSR

from molfeat._version import __version__ as MOLFEAT_VERSION
from molfeat.calc.base import SerializableCalculator
from molfeat.calc._atom_bond_features import atom_one_hot
from molfeat.calc._atom_bond_features import atom_degree_one_hot
from molfeat.calc._atom_bond_features import atom_extended_properties
from molfeat.calc._atom_bond_features import atom_implicit_valence_one_hot
from molfeat.calc._atom_bond_features import atom_hybridization_one_hot
from molfeat.calc._atom_bond_features import atom_is_aromatic
from molfeat.calc._atom_bond_features import atom_num_radical_electrons
from molfeat.calc._atom_bond_features import atom_is_in_ring
from molfeat.calc._atom_bond_features import atom_formal_charge
from molfeat.calc._atom_bond_features import atom_total_num_H_one_hot
from molfeat.calc._atom_bond_features import atom_is_chiral_center
from molfeat.calc._atom_bond_features import atom_chiral_tag_one_hot
from molfeat.calc._atom_bond_features import atom_partial_charge
from molfeat.calc._atom_bond_features import DGLLIFE_HYBRIDIZATION_LIST
from molfeat.calc._atom_bond_features import DGLLIFE_WEAVE_ATOMS
from molfeat.calc._atom_bond_features import DGLLIFE_WEAVE_CHIRAL_TYPES
from molfeat.utils import datatype
from molfeat.utils.commons import concat_dict
from molfeat.utils.commons import hex_to_fn
from molfeat.utils.commons import fn_to_hex


class AtomCalculator(SerializableCalculator):
    """
    Base class for computing atom properties compatible with DGLLife
    """

    DEFAULT_FEATURIZER = {
        "atom_one_hot": atom_one_hot,
        "atom_degree_one_hot": atom_degree_one_hot,
        "atom_implicit_valence_one_hot": atom_implicit_valence_one_hot,
        "atom_hybridization_one_hot": atom_hybridization_one_hot,
        "atom_is_aromatic": atom_is_aromatic,
        "atom_formal_charge": atom_formal_charge,
        "atom_num_radical_electrons": atom_num_radical_electrons,
        "atom_is_in_ring": atom_is_in_ring,
        "atom_total_num_H_one_hot": atom_total_num_H_one_hot,
        "atom_chiral_tag_one_hot": atom_chiral_tag_one_hot,
        "atom_is_chiral_center": atom_is_chiral_center,
    }

    def __init__(
        self,
        featurizer_funcs: Dict[str, Callable] = None,
        concat: bool = True,
        name: str = "hv",
    ):
        """
        Init function of the atom property calculator

        Args:
            featurizer_funcs : Mapping of feature name to the featurization function.
                For compatibility a list of callable/function is still accepted, and the corresponding
                featurizer name will be automatically generated. Each function is of signature
                ``func(dm.Atom) -> list or 1D numpy array``.
            concat: Whether to concat all the data into a single value in the output dict
            name: Name of the key name of the concatenated features
        """
        self._input_kwargs = locals().copy()
        self._input_kwargs.pop("self")
        # we also remove the featurizer funcs
        self._input_kwargs.pop("featurizer_funcs", None)
        self._toy_mol = dm.to_mol("CCO")
        self._feat_sizes = dict()
        if featurizer_funcs is None:
            featurizer_funcs = self.DEFAULT_FEATURIZER
        if not isinstance(featurizer_funcs, dict):
            get_name = lambda x: getattr(x, "__name__", repr(x))
            featurizer_funcs = dict((get_name(x), x) for x in featurizer_funcs)
        self.featurizer_funcs = featurizer_funcs
        for k in self.featurizer_funcs.keys():
            self.feat_size(feat_name=k)
        self.concat = concat
        self.name = name

    def to_state_dict(self):
        """
        Convert the Atom calculator to a state dict
        Due to some constraints and cross-version compatibility,  the featurizer functions
        need to be pickled and not just return a list
        """
        state_dict = {}
        state_dict["name"] = self.__class__.__name__
        state_dict["module"] = self.__class__.__module__
        state_dict["args"] = self._input_kwargs
        featurizer_fn_pickled = {}
        for fname, ffunc in self.featurizer_funcs.items():
            featurizer_fn_pickled[fname] = fn_to_hex(ffunc)
        state_dict["args"]["featurizer_funcs"] = featurizer_fn_pickled
        state_dict["_molfeat_version"] = MOLFEAT_VERSION

        signature = inspect.signature(self.__init__)
        val = {
            k: v.default
            for k, v in signature.parameters.items()
            # if v.default is not inspect.Parameter.empty
        }
        to_remove = [k for k in state_dict["args"] if k not in val.keys()]
        for k in to_remove:
            state_dict["args"].pop(k)

        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict, override_args: Optional[dict] = None):
        """Create an instance of an atom calculator from a state dict

        Args:
            state_dict: state dictionary to use to create the atom calculator
            override_args: optional dictionary of arguments to override the ones in the state dict
                at construction of the new object
        """
        # EN: at this moment, version compatibility is not enforced
        cls_name = state_dict.get("name", cls.__name__)
        module_name = state_dict.get("module", cls.__module__)
        module = importlib.import_module(module_name)
        klass = getattr(module, cls_name)
        kwargs = state_dict["args"].copy()
        # now we need to unpickle the featurizer functions
        featurizer_fn_pickled = kwargs.pop("featurizer_funcs", None)
        if featurizer_fn_pickled is not None:
            featurizer_fn_loaded = {}
            for k, v in featurizer_fn_pickled.items():
                featurizer_fn_loaded[k] = hex_to_fn(v)
            kwargs["featurizer_funcs"] = featurizer_fn_loaded
        kwargs.update(**(override_args or {}))
        return klass(**kwargs)

    def _concat(self, data_dict: Dict[str, Iterable]):
        """Concatenate the data into a single value

        Args:
            data_dict: mapping of feature names to tensor/arrays
        Returns:
            concatenated_dict: a dict with a single key where all array have been concatenated
        """
        return concat_dict(data_dict, new_name=self.name)

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Args:
            feat_name: Feature for query.

        Returns:
            int: Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert (
                len(self.featurizer_funcs) == 1
            ), "feat_name should be provided if there are more than one features"
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            raise ValueError(
                "Expect feat_name to be in {}, got {}".format(
                    list(self.featurizer_funcs.keys()), feat_name
                )
            )

        if feat_name not in self._feat_sizes:
            atom = self._toy_mol.GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))
        return self._feat_sizes[feat_name]

    def __len__(self):
        """Get length of the property estimator"""
        return sum(v for k, v in self._feat_sizes.items() if k != self.name)

    def __call__(self, mol: Union[dm.Mol, str], dtype: Callable = None):
        """
        Get rdkit basic descriptors for a molecule

        Args:
            mol: the molecule of interest
            dtype: requested data type

        Returns:
            dict:  For each function in self.featurizer_funcs with the key ``k``, store the computed feature under the key ``k``.
        """
        mol = dm.to_mol(mol)
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list).astype(np.float32)
            processed_features[feat_name] = feat

        if self.concat:
            processed_features = self._concat(processed_features)

        if dtype is not None:
            for feat_name, feat in processed_features.items():
                feat = datatype.cast(feat, dtype=dtype)
                processed_features[feat_name] = feat

        return processed_features


class AtomMaterialCalculator(AtomCalculator):
    """Atom calculator with the extend atomic property list
    which have been collected from various material science packages
    """

    DEFAULT_FEATURIZER = {
        "atom_one_hot": atom_one_hot,
        "atom_extended_properties": atom_extended_properties,
        "atom_degree_one_hot": atom_degree_one_hot,
        "atom_implicit_valence_one_hot": atom_implicit_valence_one_hot,
        "atom_hybridization_one_hot": atom_hybridization_one_hot,
        "atom_is_aromatic": atom_is_aromatic,
        "atom_formal_charge": atom_formal_charge,
        "atom_num_radical_electrons": atom_num_radical_electrons,
        "atom_is_in_ring": atom_is_in_ring,
        "atom_chiral_tag_one_hot": atom_chiral_tag_one_hot,
        "atom_is_chiral_center": atom_is_chiral_center,
    }


class DGLCanonicalAtomCalculator(AtomCalculator):
    """Default canonical featurizer for atoms used by dgllife"""

    DEFAULT_FEATURIZER = {
        "atom_one_hot": atom_one_hot,
        "atom_degree_one_hot": atom_degree_one_hot,
        "atom_implicit_valence_one_hot": atom_implicit_valence_one_hot,
        "atom_formal_charge": atom_formal_charge,
        "atom_num_radical_electrons": atom_num_radical_electrons,
        "atom_hybridization_one_hot": partial(
            atom_hybridization_one_hot, allowable_set=DGLLIFE_HYBRIDIZATION_LIST
        ),
        "atom_is_aromatic": atom_is_aromatic,
        "atom_total_num_H_one_hot": atom_total_num_H_one_hot,
    }

    def _concat(self, data_dict: Dict[str, Iterable]):
        """Concatenate the data into a single value

        Args:
            data_dict: mapping of feature names to tensor/arrays
        Returns:
            concatenated_dict: a dict with a single key where all array have been concatenated
        """
        out = concat_dict(data_dict, new_name=self.name, order=list(self.featurizer_funcs.keys()))
        return out


class DGLWeaveAtomCalculator(DGLCanonicalAtomCalculator):
    """Default atom featurizer used by WeaveNet in DGLLife"""

    DEFAULT_FEATURIZER = {
        "atom_one_hot": partial(
            atom_one_hot, allowable_set=DGLLIFE_WEAVE_ATOMS, encode_unknown=True
        ),
        "atom_chiral_tag_one_hot": partial(
            atom_chiral_tag_one_hot, allowable_set=DGLLIFE_WEAVE_CHIRAL_TYPES
        ),
        "atom_formal_charge": atom_formal_charge,
        "atom_partial_charge": atom_partial_charge,
        "atom_is_aromatic": atom_is_aromatic,
        "atom_hybridization_one_hot": partial(
            atom_hybridization_one_hot, allowable_set=DGLLIFE_HYBRIDIZATION_LIST[:3]
        ),
    }

    def __init__(self, concat: bool = True, name: str = "hv"):
        featurizer_funcs = self.DEFAULT_FEATURIZER
        featurizer_funcs["atom_weavenet_props"] = self.atom_weave_props
        super().__init__(concat=concat, name=name, featurizer_funcs=featurizer_funcs)

    def _get_atom_state_info(self, feats):
        """Get atom Donor/Acceptor state information from chemical pharmacophore features

        Args:
            feats: computed chemical features
        """
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in feats:
            if feats.GetFamily() == "Donor":
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == "Acceptor":
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True
        return is_donor, is_acceptor

    @staticmethod
    @lru_cache(maxsize=None)
    def _feat_factory_cache():
        """Build and cache chemical features caching for speed"""
        fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        chem_feats = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        return chem_feats

    @lru_cache
    def _compute_weave_net_properties(self, mol: dm.Mol):
        # Get information for donor and acceptor
        chem_feats = self._feat_factory_cache()
        mol_feats = chem_feats.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self._get_atom_state_info(mol_feats)
        sssr = GetSymmSSSR(mol)
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for i in range(num_atoms):
            cur_atom_props = [float(is_donor[i]), float(is_acceptor[i])]
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            cur_atom_props.extend(count)
            atom_features.append(cur_atom_props)
        return atom_features

    def atom_weave_props(self, atom: dm.Atom):
        """Get the WeaveNet properties for an atom"""
        mol = atom.GetOwningMol()
        feats = self._compute_weave_net_properties(mol)
        return feats[atom.GetIdx()]

    def __call__(self, mol: Union[dm.Mol, str], dtype: Callable = None):
        """
        Get rdkit basic descriptors for a molecule

        Args:
            mol: the molecule of interest
            dtype: requested data type

        Returns:
            dict:  For each function in self.featurizer_funcs with the key ``k``, store the computed feature under the key ``k``.
        """
        AllChem.ComputeGasteigerCharges(mol)
        return super().__call__(
            mol,
            dtype,
        )
