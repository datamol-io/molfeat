from typing import Union
from typing import Callable
from typing import Optional
from typing import Dict
from typing import Iterable

from collections import defaultdict

import importlib
import inspect
import datamol as dm
import numpy as np

from molfeat._version import __version__ as MOLFEAT_VERSION
from molfeat.calc.base import SerializableCalculator
from molfeat.calc._atom_bond_features import bond_type_one_hot
from molfeat.calc._atom_bond_features import bond_is_conjugated
from molfeat.calc._atom_bond_features import bond_is_in_ring
from molfeat.calc._atom_bond_features import bond_direction_one_hot
from molfeat.calc._atom_bond_features import bond_stereo_one_hot
from molfeat.calc._atom_bond_features import pairwise_ring_membership
from molfeat.calc._atom_bond_features import pairwise_3D_dist
from molfeat.calc._atom_bond_features import pairwise_2D_dist
from molfeat.calc._atom_bond_features import pairwise_bond_indicator
from molfeat.calc._atom_bond_features import pairwise_dist_indicator
from molfeat.utils import datatype
from molfeat.utils.commons import concat_dict
from molfeat.utils.commons import hex_to_fn
from molfeat.utils.commons import fn_to_hex


class BondCalculator(SerializableCalculator):
    """
    A class for bond featurizer which loops over all bonds in a molecule and
    featurizes them with the ``featurizer_funcs``. The constructed graph is assumed to be
    a bi-directed graph by default.
    """

    DEFAULT_FEATURIZER = {
        "bond_type_one_hot": bond_type_one_hot,
        "bond_stereo_one_hot": bond_stereo_one_hot,
        "bond_is_in_ring": bond_is_in_ring,
        "bond_is_conjugated": bond_is_conjugated,
        "bond_direction_one_hot": bond_direction_one_hot,
    }

    def __init__(
        self,
        featurizer_funcs: Union[list, dict] = None,
        self_loop: bool = False,
        concat: bool = True,
        name: str = "he",
    ):
        """
        Init function of the bond property calculator

        Args:
            featurizer_funcs: Mapping feature name to the featurization function.
            self_loop: Whether self loops will be added. Default to False. If True, an additional
                column of binary values to indicate the identity of self loops will be added.
                The other features of the self loops will be zero.
            concat: Whether to concat all the data into a single value in the output dict
            name: Name of the key name of the concatenated features
        """
        self._input_kwargs = locals().copy()
        self._input_kwargs.pop("self")
        # remove featurizer_funcs too
        self._input_kwargs.pop("featurizer_funcs", None)
        self._toy_mol = dm.to_mol("CO")
        self._feat_sizes = dict()
        if featurizer_funcs is None:
            featurizer_funcs = self.DEFAULT_FEATURIZER
        if not isinstance(featurizer_funcs, dict):
            get_name = lambda x: getattr(x, "__name__", repr(x))
            featurizer_funcs = dict((get_name(x), x) for x in featurizer_funcs)
        self.featurizer_funcs = featurizer_funcs
        self._self_loop = self_loop
        self.concat = concat
        self.name = name
        for k in self.featurizer_funcs.keys():
            self.feat_size(feat_name=k)
        if self._self_loop:
            self._feat_sizes["self_loop"] = 1

    def to_state_dict(self):
        """Convert the Atom calculator to a state dict
        Due to some constraints and cross-version compatibility,  the featurizer functions
        need to be pickled and not just list
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
            #    if v.default is not inspect.Parameter.empty
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

    def feat_size(self, feat_name: Optional[str] = None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, ``feat_name`` can be None.

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
            bond = self._toy_mol.GetBondWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](bond))
        return self._feat_sizes[feat_name]

    def __len__(self):
        """Get length of the property estimator"""
        return sum(v for k, v in self._feat_sizes.items() if k != self.name)

    def __call__(self, mol: Union[dm.Mol, str], dtype: Callable = None, **kwargs):
        """Featurize all bonds in a molecule.

        Args:
            mol: the molecule of interest
            dtype: requested data type

        Returns:
            dict: For each function in self.featurizer_funcs with the key ``k``,
                store the computed feature under the key ``k``.
        """
        mol = dm.to_mol(mol)
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = feat

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                # add a new label that says the feat are not self loop
                # feats = np.concatenate([feats, np.zeros((feats.shape[0], 1))], axis=1)
                # add a label at the last position that says it's a selfloop
                add_edges = np.zeros((num_atoms, feats.shape[1]))
                # self_loop_feats[:, -1] = 1
                feats = np.concatenate([feats, add_edges], axis=0)
                processed_features[feat_name] = feats
            self_loop_feats = np.concatenate(
                [np.zeros((num_bonds * 2, 1)), np.ones((num_atoms, 1))]
            )

            processed_features["self_loop"] = self_loop_feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            old_concat = self.concat
            self.concat = False
            processed_features = self(self._toy_mol)
            self.concat = old_concat
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = np.zeros((num_atoms, feats.shape[1]))
                processed_features[feat_name] = feats
        if self.concat and (num_bonds > 0 or self._self_loop):
            processed_features = self._concat(processed_features)
        if dtype is not None:
            for feat_name, feat in processed_features.items():
                feat = datatype.cast(feat, dtype=dtype)
                processed_features[feat_name] = feat

        return processed_features


class EdgeMatCalculator(BondCalculator):
    """Generate edge featurizer matrix"""

    DEFAULT_PAIRWISE_FEATURIZER = {
        "pairwise_2D_dist": pairwise_2D_dist,
        # "pairwise_3D_dist": pairwise_3D_dist,
        "pairwise_ring_membership": pairwise_ring_membership,
    }

    def __init__(
        self,
        featurizer_funcs: Union[list, dict] = None,
        pairwise_atom_funcs: Union[list, dict, str] = "default",
        name: str = "he",
    ):
        """
        Init function of the edge matrix property calculator

        Args:
            featurizer_funcs: Mapping feature name to the featurization function.
            pairwise_atom_funcs: Mapping feature name to pairwise featurization function.
                Use the keywords "default" for the default values
        """
        if pairwise_atom_funcs == "default":
            pairwise_atom_funcs = self.DEFAULT_PAIRWISE_FEATURIZER
        if not isinstance(pairwise_atom_funcs, dict):
            get_name = lambda x: getattr(x, "__name__", repr(x))
            pairwise_atom_funcs = dict((get_name(x), x) for x in pairwise_atom_funcs)
        self.pairwise_atom_funcs = pairwise_atom_funcs
        super().__init__(featurizer_funcs=featurizer_funcs, concat=True, name=name)
        # add conf data to toy mol
        self._toy_mol = dm.conformers.generate(self._toy_mol, n_confs=1, minimize_energy=False)
        for k in self.pairwise_atom_funcs.keys():
            self.feat_size(feat_name=k)

    def to_state_dict(self):
        """Convert the Atom calculator to a state dict
        Due to some constraints and cross-version compatibility,  the featurizer functions
        need to be pickled and not just list
        """
        state_dict = super().to_state_dict()
        # repeat for the pairwise one
        pairwise_atom_fn_pickled = {}
        for fname, ffunc in self.pairwise_atom_funcs.items():
            pairwise_atom_fn_pickled[fname] = fn_to_hex(ffunc)
        state_dict["args"]["pairwise_atom_funcs"] = pairwise_atom_fn_pickled
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

        pairwise_atom_fn_pickled = kwargs.pop("pairwise_atom_funcs", None)
        if pairwise_atom_fn_pickled is not None:
            pairwise_atom_fn_loaded = {}
            for k, v in pairwise_atom_fn_pickled.items():
                pairwise_atom_fn_loaded[k] = hex_to_fn(v)
            kwargs["pairwise_atom_funcs"] = pairwise_atom_fn_loaded
        kwargs.update(**(override_args or {}))
        return klass(**kwargs)

    def feat_size(self, feat_name: Optional[str] = None):
        """Get the feature size for ``feat_name``.

        Args:
            feat_name: Feature for query.

        Returns:
            int: Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name not in self.featurizer_funcs and feat_name not in self.pairwise_atom_funcs:
            raise ValueError(
                "Expect feat_name to be in {}, got {}".format(
                    list(self.featurizer_funcs.keys()), feat_name
                )
            )
        if feat_name not in self._feat_sizes:
            if feat_name in self.featurizer_funcs:
                bond = self._toy_mol.GetBondWithIdx(0)
                self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](bond))
            elif feat_name in self.pairwise_atom_funcs:
                self._feat_sizes[feat_name] = self.pairwise_atom_funcs[feat_name](
                    self._toy_mol
                ).shape[-1]
            else:
                raise ValueError(f"Feature name {feat_name} is not defined !")
        return self._feat_sizes[feat_name]

    def __call__(self, mol: Union[dm.Mol, str], dtype: Callable = None, flat: bool = True):
        """Featurize all bonds in a molecule.

        Args:
            mol: the molecule of interest
            dtype: requested data type
            flat: whether to return a collapsed N^2, M or a N, N, M matrix

        Returns:
            dict: For each function in self.featurizer_funcs with the key ``k``,
                store the computed feature under the key ``k``.
        """

        mol = dm.to_mol(mol)
        num_bonds = mol.GetNumBonds()
        num_atoms = mol.GetNumAtoms()
        feat_size = len(self)
        edge_matrix = None

        if self.pairwise_atom_funcs is not None:
            feat_size -= sum(self._feat_sizes[x] for x in self.pairwise_atom_funcs.keys())
        if self.featurizer_funcs is not None and len(self.featurizer_funcs) > 0:
            edge_matrix = np.zeros((num_atoms, num_atoms, feat_size))
            # Compute features for each bond
            for i in range(num_bonds):
                bond = mol.GetBondWithIdx(i)
                a_idx_1 = bond.GetBeginAtomIdx()
                a_idx_2 = bond.GetEndAtomIdx()
                bond_features = defaultdict(list)
                for feat_name, feat_func in self.featurizer_funcs.items():
                    feat = feat_func(bond)
                    bond_features[feat_name].extend([feat])
                bond_features = self._concat(bond_features)[self.name]
                edge_matrix[a_idx_1, a_idx_2] = bond_features
                edge_matrix[a_idx_2, a_idx_1] = bond_features

            edge_matrix = edge_matrix.reshape(-1, feat_size)
        if self.pairwise_atom_funcs is not None:
            pwise_features = dict()
            for pname, pfunc in self.pairwise_atom_funcs.items():
                pwise_features[pname] = pfunc(mol)
            pwise_features = self._concat(pwise_features)[self.name]
            if edge_matrix is not None:
                edge_matrix = np.concatenate([edge_matrix, pwise_features], axis=-1)
            else:
                edge_matrix = pwise_features
        if not flat:
            edge_matrix = edge_matrix.reshape(num_atoms, num_atoms, -1)
        if dtype is not None:
            edge_matrix = datatype.cast(edge_matrix, dtype=dtype)
        return {self.name: edge_matrix}


class DGLCanonicalBondCalculator(BondCalculator):
    DEFAULT_FEATURIZER = {
        "bond_type_one_hot": bond_type_one_hot,
        "bond_is_conjugated": bond_is_conjugated,
        "bond_is_in_ring": bond_is_in_ring,
        "bond_stereo_one_hot": bond_stereo_one_hot,
    }

    def _concat(self, data_dict: Dict[str, Iterable]):
        """Concatenate the data into a single value

        Args:
            data_dict: mapping of feature names to tensor/arrays
        Returns:
            concatenated_dict: a dict with a single key where all array have been concatenated
        """
        return concat_dict(data_dict, new_name=self.name, order=list(self.featurizer_funcs.keys()))


class DGLWeaveEdgeCalculator(EdgeMatCalculator):
    """Edge featurizer used by WeaveNets

    The edge featurization is introduced in `Molecular Graph Convolutions:
    Moving Beyond Fingerprints <https://arxiv.org/abs/1603.00856>`__.

    This featurization is performed for a complete graph of atoms with self loops added,
    which considers the following default:

    * Number of bonds between each pairs of atoms
    * One-hot encoding of bond type if a bond exists between a pair of atoms
    * Whether a pair of atoms belongs to a same ring

    """

    DEFAULT_FEATURIZER = {}
    DEFAULT_PAIRWISE_FEATURIZER = {
        "pairwise_dist_indicator": pairwise_dist_indicator,
        "pairwise_bond_indicator": pairwise_bond_indicator,
        "pairwise_ring_membership": pairwise_ring_membership,
    }

    def _concat(self, data_dict: Dict[str, Iterable]):
        """Concatenate the data into a single value

        Args:
            data_dict: mapping of feature names to tensor/arrays
        Returns:
            concatenated_dict: a dict with a single key where all array have been concatenated
        """

        # To reproduce DGLDefault, we need to keep the order of dict insertion
        return concat_dict(
            data_dict, new_name=self.name, order=list(self.pairwise_atom_funcs.keys())
        )
