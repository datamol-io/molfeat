from typing import Optional
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any

import os
import random
import functools
import importlib.resources as pkg_resources

import pandas as pd
import numpy as np
import datamol as dm
import fsspec

from sklearn.cluster import OPTICS

from rdkit import RDConfig
from rdkit.Chem import rdmolops
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory

from loguru import logger

from molfeat.calc.base import SerializableCalculator
from molfeat.utils.datatype import to_numpy
from molfeat.utils.datatype import to_fp
from molfeat.utils.commons import fold_count_fp
from molfeat.utils import commons
from molfeat import viz

from pmapper.pharmacophore import Pharmacophore as Pharm


class Pharmacophore2D(SerializableCalculator):
    """2D Pharmacophore.

    The fingerprint is computed using `Generate.Gen2DFingerprint` from RDKit.

    An explanation of pharmacophore fingerprints and how the bits are set
    is available in the RDKit book. In particular the following figure describes the process.
    ![Pharmacophore](https://www.rdkit.org/docs/_images/picture_10.jpg){ align=left }
    """

    def __init__(
        self,
        factory: Union[str, MolChemicalFeatureFactory] = "pmapper",
        length: Optional[int] = 2048,
        useCounts: bool = None,
        minPointCount: int = None,
        maxPointCount: int = None,
        shortestPathsOnly: bool = None,
        includeBondOrder: bool = None,
        skipFeats: List[str] = None,
        trianglePruneBins: bool = None,
        bins: List[Tuple[int, int]] = None,
        **kwargs,
    ):
        """Pharmacophore computation.

        Args:
            factory: Which features factory to use. One of "default", "cats", "gobbi" , "pmapper" or path
                to a feature definition or a feature factory object
            length: Optional desired length. If provided, the fp will be refold or padded to that length.
                If set to None, fallback to the default for the provided sig factory.
            minPointCount: Minimum number of points.
            maxPointCount: Maximum number of points.
            trianglePruneBins: Whether to prune the triangle inequality.
            includeBondOrder: Whether to consider bond order.
            shortestPathsOnly: Whether to only use the shortest path between pharmacophores.
            useCounts: Whether take into account the count information. This will also impact how the folding works.
            bins: Bins to use.
        """

        self.factory = factory
        self.useCounts = useCounts
        self.minPointCount = minPointCount
        self.maxPointCount = maxPointCount
        self.shortestPathsOnly = shortestPathsOnly
        self.includeBondOrder = includeBondOrder
        self.skipFeats = skipFeats
        self.trianglePruneBins = trianglePruneBins
        self.bins = bins

        self.length = length

        self._init_sig_factory()

    def __call__(self, mol: Union[dm.Mol, str], raw: bool = False):
        """Compute the Pharmacophore fingeprint for the input molecule.

        Args:
            mol: the molecule of interest
            raw: Whether to return the raw fingerprint or a Numpy array.

        Returns:
            fp: the computed fingerprint as a Numpy array or as a raw object.
        """

        # Get a molecule
        mol = dm.to_mol(mol)

        if mol is None:
            raise ValueError("The input molecule is not valid.")

        # Get distance matrix
        use_bond_order = self.sig_factory.includeBondOrder
        d_mat = rdmolops.GetDistanceMatrix(mol, use_bond_order)

        # Generate the fingerprint
        fp = Generate.Gen2DFingerprint(mol, self.sig_factory, dMat=d_mat)

        # Posprocessing
        if self.length and self._should_fold:
            # refold the fingerprint
            fp = fold_count_fp(fp, dim=self.length, binary=not (self.useCounts or False))
            if raw:
                fp = to_fp(fp, bitvect=True)

        if not raw:
            fp = to_numpy(fp)

        return fp

    def _init_sig_factory(self):
        """Init the feature factory for this pharmacophore."""

        self.sig_factory = get_sig_factory(
            self.factory,
            useCounts=self.useCounts,
            minPointCount=self.minPointCount,
            maxPointCount=self.maxPointCount,
            shortestPathsOnly=self.shortestPathsOnly,
            includeBondOrder=self.includeBondOrder,
            skipFeats=self.skipFeats,
            trianglePruneBins=self.trianglePruneBins,
            bins=self.bins,
        )

        # Reinject used params to the class attributes
        # It might be useful in case the default values are changed
        # and when serializing the object.
        self.useCounts = self.sig_factory.useCounts
        self.minPointCount = self.sig_factory.minPointCount
        self.maxPointCount = self.sig_factory.maxPointCount
        self.shortestPathsOnly = self.sig_factory.shortestPathsOnly
        self.includeBondOrder = self.sig_factory.includeBondOrder
        self.skipFeats = self.sig_factory.skipFeats
        self.trianglePruneBins = self.sig_factory.trianglePruneBins
        self.bins = self.sig_factory.GetBins()

    @property
    @functools.lru_cache(maxsize=None)
    def _should_fold(self):
        return self.sig_factory.GetSigSize() != len(self)

    @property
    def feature_factory(self):
        return self.sig_factory.featFactory

    def __len__(self):
        """Returns the length of the pharmacophore"""
        return self.length or self.sig_factory.GetSigSize()

    @property
    def columns(self):
        """Get the name of all the descriptors of this calculator."""

        if not self.length:
            return [self.sig_factory.GetBitDescription(x) for x in range(len(self))]
        else:
            return [f"Desc:{i}" for i in range(self.length)]

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["factory"] = self.factory
        state["useCounts"] = self.useCounts
        state["minPointCount"] = self.minPointCount
        state["maxPointCount"] = self.maxPointCount
        state["shortestPathsOnly"] = self.shortestPathsOnly
        state["includeBondOrder"] = self.includeBondOrder
        state["skipFeats"] = self.skipFeats
        state["trianglePruneBins"] = self.trianglePruneBins
        state["bins"] = self.bins
        state["length"] = self.length
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        self.__dict__.update(state)
        self._init_sig_factory()


class Pharmacophore3D(SerializableCalculator):
    """3D Pharmacophore.

    The fingerprint is computed using [`pmapper`](https://github.com/DrrDom/pmapper).

    This featurizer supports building a consensus pharmacophore from a set of molecules.
    """

    def __init__(
        self,
        factory: Union[str, MolChemicalFeatureFactory] = "pmapper",
        length: int = 2048,
        bin_step: float = 1,
        min_features: int = 2,
        max_features: int = 3,
        use_modulo: bool = True,
        tolerance: float = 0,
    ):
        """Pharmacophore computation.

        Args:
            factory: Which features factory to use. One of "default", "cats", "gobbi" , "pmapper" or path
                to a feature definition or a feature factory object
            length: Optional desired length. If provided, the fp will be refold or padded to that length.
                If set to None, fallback to the default for the provided sig factory.
            bin_step: Bin step to use.
            min_features: Minimum number of features to use.
            max_features: Maximum number of features to use.
            use_modulo: whether to use modulo to compute the pharmacophore fingerprint
            tolerance: tolerance value to use when computing the pharmacophore fingerprint
        """

        self.factory = factory
        self.length = length
        self.bin_step = bin_step
        self.min_features = min_features
        self.max_features = max_features
        self.use_modulo = use_modulo
        self.tolerance = tolerance

        self._init_feature_factory()

    def __call__(self, mol: Union[dm.Mol, str], conformer_id: int = -1, raw: bool = False):
        """Compute the Pharmacophore fingeprint for the input molecule.

        Args:
            mol: the molecule of interest
            conformer_id: the conformer id to use.
            raw: Whether to return the raw fingerprint or a Numpy array.

        Returns:
            fp: the computed fingerprint as a Numpy array.
        """

        # Get a molecule
        mol = dm.to_mol(mol)

        if mol is None:
            raise ValueError("The input molecule is not valid.")

        if mol.GetNumConformers() < 1:  # type: ignore
            raise ValueError("Expected a molecule with conformers information.")

        # Get the features for the mol
        features = self.get_features(mol, conformer_id=conformer_id)

        # Convert features dataframe to coordinates
        if features.empty:
            features_coords = []
        else:
            features_coords = features[["feature_name", "coords"]].values.tolist()

        # Compute the fingerprint
        fp = self.compute_fp_from_coords(features_coords, raw=raw)

        return fp

    def consensus_fp(
        self,
        mols: List[dm.Mol],
        align: bool = True,
        conformer_id: int = -1,
        copy: bool = True,
        min_samples_ratio: float = 0.5,
        eps: float = 2,
        raw: bool = False,
        **cluster_kwargs,
    ):
        """Compute a consensus fingerprint from a list of molecules.

        Args:
            mols: a list of molecules.
            align: Whether to align the conformers of the molecules.
            conformer_id: Optional conformer id.
            copy: Whether to copy the molecules before clustering.
            min_samples_ratio: Percentages of mols that must contain a pharmacophoric point
                to be considered as a core point.
            eps: The maximum distance between two samples for one to be considered as
                in the neighborhood of the other.
            raw: Whether to return the raw fingerprint or a Numpy array.
            cluster_kwargs: additional keyword arguments for the clustering algorithm.
        """

        # Get all the features
        features = self.get_features_from_many(
            mols,
            keep_mols=True,
            align=align,
            conformer_id=conformer_id,
            copy=copy,
        )

        # Retrieve the aligned molecules
        mols = features.groupby("mol_index").first()["mol"].tolist()
        # Cluster the features
        clustered_features = self.cluster_features(
            features, min_samples_ratio=min_samples_ratio, eps=eps, **cluster_kwargs
        )
        # Convert features dataframe to coordinates
        if clustered_features.empty:
            features_coords = []
        else:
            features_coords = clustered_features[["feature_name", "coords"]].values.tolist()
        # Compute the fingerprint
        fp = self.compute_fp_from_coords(features_coords, raw=raw)

        return fp

    def _init_feature_factory(self):
        """Init the feature factory."""
        self.feature_factory = get_feature_factory(self.factory)

    def get_features(self, mol: dm.Mol, conformer_id: int = -1) -> pd.DataFrame:
        """Retrieve the features for a given molecule.

        Args:
            mol: the molecule of interest

        Returns:
            features: the features as a Numpy array
        """
        features_data = []

        # Extract the features for this molecule
        features = self.feature_factory.GetFeaturesForMol(mol, confId=conformer_id)

        # Extract all the feature atom indices for this molecule
        for feature in features:
            datum = {}
            datum["feature_id"] = feature.GetId()
            datum["feature_name"] = feature.GetFamily()
            datum["feature_type"] = feature.GetType()
            datum["atom_indices"] = feature.GetAtomIds()
            datum["coords"] = np.array(feature.GetPos())

            features_data.append(datum)

        features_data = pd.DataFrame(features_data)

        return features_data

    def get_features_from_many(
        self,
        mols: List[dm.Mol],
        align: bool = True,
        conformer_id: int = -1,
        copy: bool = True,
        keep_mols: bool = False,
    ):
        """Extract all the features from a list of molecules after an optional
        alignement step.

        Args:
            mols: List of molecules with conformers.
            align: Whether to align the conformers of the molecules.
            conformer_id: Optional conformer id.
            copy: Whether to copy the molecules before clustering.
            keep_mols: Whether to keep the molecules in the returned dataframe.
        """

        if not all([mol.GetNumConformers() >= 1 for mol in mols]):
            raise ValueError("One or more input molecules is missing a conformer.")

        # Make a copy of the molecules since they are going to be modified
        if copy:
            mols = [dm.copy_mol(mol) for mol in mols]

        # Align the conformers
        if align:
            mols, _ = commons.align_conformers(mols, copy=False, conformer_id=conformer_id)

        all_features = pd.DataFrame()

        for i, mol in enumerate(mols):
            features = self.get_features(mol)
            features["mol_index"] = i

            if keep_mols:
                features["mol"] = mol

            all_features = pd.concat([all_features, features], ignore_index=True)

        return all_features

    def compute_fp_from_coords(
        self,
        features_coords: List[Tuple[str, Tuple[float]]],
        raw: bool = False,
    ):
        """Compute a fingerprint from a list of features.

        Args:
            features_coords: Features coords: `[('A', (1.23, 2.34, 3.45)), ('A', (4.56, 5.67, 6.78)), ...]`.
            raw: Whether to return the raw fingerprint or a Numpy array.
        """

        # Init the pmapper engine
        ph_engine = Pharm(bin_step=self.bin_step)
        # Convert coords to list in case those are arrays
        features_coords = [(name, tuple(coords)) for name, coords in features_coords]
        # Load pharmacophore points
        ph_engine.load_from_feature_coords(features_coords)
        # Init the iterator over the pharmacophore points
        points_iterator = ph_engine.iterate_pharm(
            min_features=self.min_features,
            max_features=self.max_features,
            tol=self.tolerance,
            return_feature_ids=False,
        )

        # Compute the fingerprint
        on_bits = set()
        for h in points_iterator:
            if self.use_modulo:
                on_bits.add(int(h, 16) % self.length)  # type: ignore
            else:
                random.seed(int(h, 16))  # type: ignore
                on_bits.add(random.randrange(self.length))

        if raw:
            return np.array(on_bits)

        fp = np.zeros(self.length, dtype=int)
        fp[list(on_bits)] = 1

        return fp

    def cluster_features(
        self,
        features: pd.DataFrame,
        min_samples_ratio: float = 0.5,
        n_mols: int = None,
        eps: float = np.inf,
        **kwargs,
    ):
        """Cluster a set of pharmacophoric features using OPTICS.
        The only reason why we are not using SpectralClustering is because of the need to provide
        the number of clusters.

        Args:
            features: A dataframe of features.
            min_samples_ratio: Percentages of mols that must contain a pharmacophoric point
                to be considered as a core point.
            n_mols: Optional number of compounds to compute `min_samples` from the
                `min_samples_ratio` value. If not set it will use `mol_index` from
                the `features` dataframe.
            eps: The maximum distance between two samples for one to be considered as
                in the neighborhood of the other. This is max_eps in OPTICS
            kwargs: Any additional parameters to pass to `sklearn.cluster.OPTICS`.
        """

        if n_mols is None:
            n_mols = len(features["mol_index"].unique())

        # Compute min_samples
        min_samples = max(int(round(min_samples_ratio * n_mols, 0)), 1)
        clusters = []
        feature_id = 0
        for _, rows in features.groupby("feature_name"):
            feature_name = rows.iloc[0]["feature_name"]
            if min_samples > rows.shape[0]:
                logger.info(
                    f"Feature {feature_name} does not have enough molecule ({len(rows)}), skipping"
                )
                continue
            coords = np.vstack(rows["coords"].values)

            # Init clustering
            optics = OPTICS(min_samples=min_samples, max_eps=eps, **kwargs)
            optics = optics.fit(coords)
            labels = optics.labels_
            # a node that is not a core would basically be a node that cannot be labeled
            # thus border nodes are considered core
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[labels == 1] = True

            # Find the centroids (consensus points)
            unique_labels = set(labels)
            for k in unique_labels:
                if k == -1:
                    continue
                class_member_mask = labels == k
                cluster_coords = coords[class_member_mask & core_samples_mask]
                if len(cluster_coords) == 0:
                    continue
                cluster_centroid = cluster_coords.mean(axis=0)

                cluster = {}
                cluster["feature_id"] = feature_id
                cluster["feature_name"] = feature_name
                cluster["coords"] = cluster_centroid
                cluster["cluster_size"] = len(cluster_coords)

                clusters.append(cluster)
                feature_id += 1

        clusters = pd.DataFrame(clusters)

        return clusters

    ## Viz methods

    def show(
        self,
        mol: dm.Mol,
        features: pd.DataFrame = None,
        alpha: float = 1.0,
        sphere_radius: float = 0.4,
        show_legend: bool = True,
    ):
        """Show a 3D view of a given molecule with the pharmacophoric features.

        Args:
            mol: the molecule of interest
            alpha: Alpha value for the colors (currently not working).
            sphere_radius: Radius of the spheres for the features.
            show_legend: Display the legend (the layout is bad but at least it
                shows the legend).
        """

        if features is None:
            features = self.get_features(mol)

        return viz.show_pharm_features(
            mol,
            features=features,
            feature_factory=self.feature_factory,
            alpha=alpha,
            sphere_radius=sphere_radius,
            show_legend=show_legend,
        )

    def show_many(
        self,
        mols: List[dm.Mol],
        align: bool = True,
        conformer_id: int = -1,
        copy: bool = True,
        min_samples_ratio: float = 0.5,
        eps: float = 2,
        alpha: float = 1.0,
        sphere_radius: float = 0.4,
        show_legend: bool = True,
    ):
        """Show a 3D view of a given molecule with the pharmacophoric features.

        Args:
            mols: a list of molecules.
            align: Whether to align the conformers of the molecules.
            conformer_id: Optional conformer id.
            copy: Whether to copy the molecules before clustering.
            min_samples_ratio: Percentages of mols that must contain a pharmacophoric point
                to be considered as a core point.
            eps: The maximum distance between two samples for one to be considered as
                in the neighborhood of the other.
            alpha: Alpha value for the colors (currently not working).
            sphere_radius: Radius of the spheres for the features.
            show_legend: Display the legend (the layout is bad but at least it
                shows the legend).
        """

        # Get all the features
        features = self.get_features_from_many(
            mols,
            keep_mols=True,
            align=align,
            conformer_id=conformer_id,
            copy=copy,
        )

        # Retrieve the aligned molecules
        mols = features.groupby("mol_index").first()["mol"].tolist()

        # Cluster the features
        clustered_features = self.cluster_features(
            features,
            min_samples_ratio=min_samples_ratio,
            eps=eps,
        )

        return viz.show_pharm_features(
            mols,
            features=clustered_features,
            feature_factory=self.feature_factory,
            alpha=alpha,
            sphere_radius=sphere_radius,
            show_legend=show_legend,
        )

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["factory"] = self.factory
        state["length"] = self.length
        state["bin_step"] = self.bin_step
        state["min_features"] = self.min_features
        state["max_features"] = self.max_features
        state["use_modulo"] = self.use_modulo
        state["tolerance"] = self.tolerance
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        self.__dict__.update(state)
        self._init_feature_factory()


## Factory related utility functions


def get_feature_factory(
    factory: Union[str, MolChemicalFeatureFactory]
) -> MolChemicalFeatureFactory:
    """Build a feature factory."""

    if isinstance(factory, MolChemicalFeatureFactory):
        feature_factory = factory

    elif factory == "pmapper":
        with pkg_resources.path("pmapper", "smarts_features.fdef") as fdef_name:
            feature_factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))  # type: ignore

    elif factory == "gobbi":
        feature_factory = Gobbi_Pharm2D.factory.featFactory

    elif factory == "cats":
        with pkg_resources.open_text("molfeat.data", "cats_features.fdef") as instream:
            feature_factory = ChemicalFeatures.BuildFeatureFactoryFromString(instream.read())  # type: ignore

    elif factory == "default":
        # Load default feature definition file
        fdefFile = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefFile)  # type: ignore

    elif dm.fs.exists(factory):
        with fsspec.open(factory, "r") as instream:
            fdef = instream.read()
            feature_factory = ChemicalFeatures.BuildFeatureFactoryFromString(fdef)  # type: ignore

    else:
        raise ValueError(f"The factory '{factory}' is not supported.")

    return feature_factory


def get_sig_factory(
    factory: Union[str, MolChemicalFeatureFactory],
    useCounts: bool = None,
    minPointCount: int = None,
    maxPointCount: int = None,
    shortestPathsOnly: bool = None,
    includeBondOrder: bool = None,
    skipFeats: List[str] = None,
    trianglePruneBins: bool = None,
    bins: List[Tuple[int, int]] = None,
    init_factory: bool = True,
):
    """Build a signature factory."""

    # Get feature factory
    feature_factory = get_feature_factory(factory)

    # Get default params and override them as needed
    params, bins = get_sig_factory_params(
        factory,
        useCounts=useCounts,
        minPointCount=minPointCount,
        maxPointCount=maxPointCount,
        shortestPathsOnly=shortestPathsOnly,
        includeBondOrder=includeBondOrder,
        skipFeats=skipFeats,
        trianglePruneBins=trianglePruneBins,
        bins=bins,
    )

    # Build signature factory
    sig_factory = SigFactory(feature_factory, **params)

    # Set bins
    sig_factory.SetBins(bins)

    # Init the factory
    if init_factory:
        sig_factory.Init()

    return sig_factory


def get_sig_factory_params(
    factory_name: str,
    useCounts: bool = None,
    minPointCount: int = None,
    maxPointCount: int = None,
    shortestPathsOnly: bool = None,
    includeBondOrder: bool = None,
    skipFeats: List[str] = None,
    trianglePruneBins: bool = None,
    bins: List[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Any], list]:
    """Get the default parameter for a given sig factory allowing some of them to be overriden.

    Args:
        factory_name: The name of the factory.
    """

    # Get default params.

    if factory_name == "cats":
        default_bins = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
        ]
        params = dict(
            useCounts=True,
            minPointCount=2,
            maxPointCount=2,
            trianglePruneBins=True,
            shortestPathsOnly=True,
            includeBondOrder=False,
        )

    elif factory_name == "gobbi":
        default_bins = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)]
        params = dict(
            useCounts=False,
            minPointCount=2,
            maxPointCount=3,
            trianglePruneBins=True,
            shortestPathsOnly=True,
            includeBondOrder=False,
        )

    elif factory_name == "pmapper":
        default_bins = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)]
        params = dict(
            useCounts=False,
            minPointCount=2,
            maxPointCount=3,
            trianglePruneBins=False,
            shortestPathsOnly=True,
            includeBondOrder=False,
        )

    elif factory_name == "default":
        params = dict(
            useCounts=False,
            minPointCount=2,
            maxPointCount=3,
            trianglePruneBins=False,
            shortestPathsOnly=True,
            skipFeats=["ZnBinder", "LumpedHydrophobe"],
            includeBondOrder=False,
        )
        default_bins = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 100)]

    else:
        raise ValueError(f"Default values for {factory_name} are not known.")

    # Override default params when set.

    if minPointCount is not None:
        params["minPointCount"] = minPointCount

    if maxPointCount is not None:
        params["maxPointCount"] = maxPointCount

    if trianglePruneBins is not None:
        params["trianglePruneBins"] = trianglePruneBins

    if includeBondOrder is not None:
        params["includeBondOrder"] = includeBondOrder

    if useCounts is not None:
        params["useCounts"] = useCounts

    if skipFeats is not None:
        params["skipFeats"] = skipFeats  # type: ignore

    if shortestPathsOnly is not None:
        params["shortestPathsOnly"] = shortestPathsOnly

    bins = bins or default_bins

    return params, bins
