from .base import SerializableCalculator, _CALCULATORS
from .cats import CATS
from .descriptors import RDKitDescriptors2D, RDKitDescriptors3D
from .fingerprints import FPCalculator, FP_FUNCS
from .pharmacophore import Pharmacophore2D, Pharmacophore3D
from .shape import ElectroShapeDescriptors, USRDescriptors
from .skeys import ScaffoldKeyCalculator


def get_calculator(name: str, **params):
    """Get molecular calculator based on name

    Args:
        name: Name of the featurizer
        params (dict): Parameters of the featurizer

    Raises:
        ValueError: When featurizer is not supported

    Returns:
        featurizer: Callable
    """
    if not isinstance(name, str):
        return name

    CALC_MAP = {k.lower(): v for k, v in _CALCULATORS.items()}
    name = name.lower()
    if name in FP_FUNCS.keys():
        featurizer = FPCalculator(name, **params)
    elif name == "desc3d":
        featurizer = RDKitDescriptors3D(**params)
    elif name == "desc2d":
        featurizer = RDKitDescriptors2D(**params)
    elif name == "cats":
        featurizer = CATS(**params)
    elif name == "cats2d":
        params["use_3d_distances"] = False
        featurizer = CATS(**params)
    elif name == "cats3d":
        params["use_3d_distances"] = True
        featurizer = CATS(**params)
    elif name == "pharm2d":
        featurizer = Pharmacophore2D(**params)
    elif name == "pharm3d":
        featurizer = Pharmacophore3D(**params)
    elif name.startswith("usr"):
        params["method"] = name
        featurizer = USRDescriptors(**params)
    elif name == "electroshape":
        featurizer = ElectroShapeDescriptors(**params)
    elif name in ["scaffoldkeys", "skeys", "scaffkeys"]:
        featurizer = ScaffoldKeyCalculator(**params)
    elif name == "none":
        featurizer = None
    # for any generic calculator that has been automatically registered
    elif name in CALC_MAP.keys():
        featurizer = CALC_MAP[name](**params)
    else:
        raise ValueError(f"{name} is not a supported internal featurizer")
    return featurizer
