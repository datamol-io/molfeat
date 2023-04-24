from typing import Union

import datamol as dm
from loguru import logger
from molfeat.utils import requires

if requires.check("map4"):
    from map4 import MAP4Calculator
else:
    MAP4Calculator = requires.mock("map4")


def MAP4(
    x: Union[dm.Mol, str],
    dimensions: int = 2048,
    radius: int = 2,
    is_counted: bool = False,
    is_folded: bool = True,
    return_strings: bool = False,
    **kwargs,
):
    """Compute MHFP fingerprint

    Args:
        x: input molecule
        dimensions (int, optional): Length of the fingerprint (default: 2048).
        radius (int, optional): Radius of the fingerprint (default: 3)
        is_counted (bool, optional): Whether to use counted fingerprints (default: True)
        is_folded (bool, optional): Whether to fold the fingerprint (default: True)
        return_strings (bool, optional): Whether to return strings values (default: False)

    Returns:
        fp: fingerprint
    """
    if not requires.check("map4"):
        logger.error(
            "`map4` is not available, please install it: https://github.com/reymond-group/map4"
        )
        raise ImportError("Cannot import `map4`")

    if isinstance(x, str):
        x = dm.to_mol(x)
    map4_encoder = MAP4Calculator(
        dimensions=dimensions,
        radius=radius,
        is_counted=is_counted,
        is_folded=is_folded,
        return_strings=return_strings,
    )
    encoded_fp = map4_encoder.calculate(x)
    return encoded_fp
