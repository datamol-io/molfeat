import pytest

from molfeat.calc import Pharmacophore2D
from molfeat import viz


def test_colors_from_feature_factory():
    # Get a factory
    calc = Pharmacophore2D(factory="pmapper")
    feature_factory = calc.sig_factory.featFactory  # type: ignore

    # Get colors
    colors = viz.colors_from_feature_factory(feature_factory)

    # Check
    assert isinstance(colors, dict)


def test_colors_respect_requested_colormap():
    calc = Pharmacophore2D(factory="pmapper")
    feature_factory = calc.sig_factory.featFactory  # type: ignore

    set1 = viz.colors_from_feature_factory(feature_factory, cmap_name="Set1")
    set2 = viz.colors_from_feature_factory(feature_factory, cmap_name="Set2")

    assert set1 != set2


def test_visualization_dependency_is_optional(monkeypatch):
    monkeypatch.setattr(viz.requires, "check", lambda dependency: False)

    with pytest.raises(ImportError, match=r"molfeat\[viz\]"):
        viz._import_matplotlib()
