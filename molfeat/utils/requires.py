from typing import Optional
import importlib
from packaging import version as pkg_version


def check(
    module: str, min_version: Optional[str] = None, max_version: Optional[str] = None
):
    """Check if module is available for import

    Args:
        module: name of the module to check
        min_version: optional minimum version string to check
        max_version: optional maximum version string to check
    """
    imported_module = None
    version = None
    min_version = pkg_version.parse(min_version) if min_version is not None else None
    max_version = pkg_version.parse(max_version) if max_version is not None else None
    try:
        imported_module = importlib.import_module(module)
        version = getattr(imported_module, "__version__", None)
    except ImportError as _:
        return False
    if version is not None:
        try:
            version = pkg_version.parse(version)
        except pkg_version.InvalidVersion as _:
            # EN: packaging v22 removed LegacyVersion which has consequences
            version = None
    return version is None or (
        (min_version is None or version >= min_version)
        and (max_version is None or version <= max_version)
    )


def mock(name: str):
    """Mock a function to raise an error

    Args:
        name: name of the module or function to mock

    """
    return lambda: (_ for _ in ()).throw(Exception(f"{name} is not available"))


## the following are provided for convenience

DGL_IS_AVAILABLE = False
MAP4_IS_AVAILABLE = False
MORDRED_IS_AVAILABLE = False
FCD_IS_AVAILABLE = False

try:
    import dgl

    DGL_IS_AVAILABLE = True
except ImportError:
    pass
try:
    import map4

    MAP4_IS_AVAILABLE = True
except ImportError:
    pass

try:
    import mordred

    MORDRED_IS_AVAILABLE = True
except ImportError:
    pass

try:
    import fcd_torch

    FCD_IS_AVAILABLE = True
except ImportError:
    pass


def check_dgl():
    """Check a dgl is available"""
    return DGL_IS_AVAILABLE


def check_map4():
    """Check map4 availability"""
    return MAP4_IS_AVAILABLE


def check_mordred():
    """Check mordred availability"""
    return MORDRED_IS_AVAILABLE


def check_fcd():
    """Check fcd availability"""
    return FCD_IS_AVAILABLE
