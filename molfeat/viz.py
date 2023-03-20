from typing import List
from typing import Union
from typing import Dict

import matplotlib.cm
import matplotlib.colors

import datamol as dm
import pandas as pd

from rdkit.Chem import rdMolChemicalFeatures


def colors_from_feature_factory(
    feature_factory: rdMolChemicalFeatures.MolChemicalFeatureFactory,
    cmap_name: str = "Set1",
    alpha: float = 1.0,
):
    """Get a list of colors for a given feature factory. For the same
    `feature_factory` the returned colors will be the same.

    Args:
        feature_factory: Feature factory to use.
        cmap_name: Matplotlib colormap name.
        alpha: Alpha value for the colors.

    Returns:
        colors: Dict of feature_name as keys and colors as values.
    """
    cmap_name = "Set1"

    cmap = matplotlib.cm.get_cmap(cmap_name)
    cmap_n = cmap.N  # type: ignore

    colors = {}
    for i, name in enumerate(feature_factory.GetFeatureFamilies()):
        color: List[float] = list(cmap(i % cmap_n))
        color[3] = alpha
        colors[name] = color

    return colors


def show_mols(mols: Union[dm.Mol, List[dm.Mol]]):
    """Generate a view of the molecules.

    Args:
        mols: A mol or a list of mols.

    Returns:
        nglview.widget.NGLWidget
    """

    import nglview as nv

    if isinstance(mols, dm.Mol):
        mols = [mols]

    view = nv.NGLWidget()
    for mol in mols:
        component = view.add_component(mol)
        component.clear()  # type: ignore
        component.add_ball_and_stick(multipleBond=True)  # type: ignore

    return view


def show_pharm_features(
    mols: Union[dm.Mol, List[dm.Mol]],
    features: pd.DataFrame,
    feature_factory: rdMolChemicalFeatures.MolChemicalFeatureFactory,
    alpha: float = 1.0,
    sphere_radius: float = 0.4,
    show_legend: bool = True,
):
    """Generate a view of the molecules with pharmacophoric features.

    Args:
        mols: A mol or a list of mols.
        features: Features data. Columns must contain at least
            "feature_name", "feature_id", and "feature_coords".
        feature_factory: Feature factory to display consistent colors.
        alpha: Alpha value for the colors (currently not working).
        sphere_radius: Radius of the spheres for the features.
        show_legend: Display the legend (the layout is bad but at least it
            shows the legend).

    Returns:
        nglview.widget.NGLWidget
    """

    import ipywidgets as ipy

    # Get mols view
    mol_view = show_mols(mols)

    # Get colors
    colors = colors_from_feature_factory(feature_factory, alpha=alpha)

    # Add features to the viz
    for _, row in features.iterrows():
        color = colors[row["feature_name"]]
        label = f"{row['feature_name']}_{row['feature_id']}"
        mol_view.shape.add_sphere(row["coords"], color, sphere_radius, label)  # type: ignore

    if not show_legend:
        return mol_view

    # Build legend widget
    colors_widget = _build_colors_widget(colors)

    main_layout = ipy.Layout(
        display="flex",
        flex_flow="column",
        align_content="center",
    )
    main_widget = ipy.HBox([mol_view, colors_widget], layout=main_layout)  # type: ignore

    return main_widget


def _build_colors_widget(colors: Dict[str, list]):
    import ipywidgets as ipy

    box_layout = ipy.Layout(
        display="flex",
        flex_flow="column",
        align_content="center",
        border="solid",
    )

    items = []
    for name, color in colors.items():
        item = ipy.Button(description=name)
        item.style.button_color = matplotlib.colors.to_hex(color)  # type: ignore
        items.append(item)

    box = ipy.Box(children=items, layout=box_layout)

    return box
