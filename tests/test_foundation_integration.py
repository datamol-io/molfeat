import numpy as np

import pytest

from molfeat.trans.pretrained import CheMeleonTransformer, MolJEPATransformer

pytestmark = pytest.mark.integration


def test_chemeleon_official_checkpoint_representation():
    """Keep the dependency-free adapter aligned with the official Chemprop path."""
    transformer = CheMeleonTransformer()
    features = transformer(["CCO", "c1ccccc1", "[Na+]"])

    assert features.shape == (3, 2048)
    np.testing.assert_allclose(
        features[:, [13, 19, 26, 30]],
        np.asarray(
            [
                [0.2620092928, 1.3849297762, 0.0301855858, 1.0129538774],
                [0.0, 0.0, 0.1095901355, 0.7540491223],
                [0.3380137682, 0.2504425049, 0.0653740093, 0.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(features, axis=1),
        np.asarray([5.3112206, 3.6752324, 4.5894136], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.timeout(600)
def test_moljepa_official_checkpoint_representation():
    """Exercise the pinned custom code and checkpoint as an explicit opt-in."""
    transformer = MolJEPATransformer(
        trust_remote_code=True,
        accept_noncommercial_license=True,
    )
    first = transformer(["CCO"])
    second = transformer(["CCO"])

    assert first.shape == (1, 512)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first[0, [0, 1, 2, 13, 127, 255, 511]],
        np.asarray(
            [
                0.0545312725,
                -0.0020755976,
                -0.0356405340,
                -0.0531620234,
                -0.0529748611,
                0.1173758656,
                0.0882600695,
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(first, axis=1),
        np.asarray([3.1669044], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
