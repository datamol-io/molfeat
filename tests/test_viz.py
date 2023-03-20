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
