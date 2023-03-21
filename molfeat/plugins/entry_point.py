from typing import Any
from typing import Optional
from typing import Sequence
from typing import List

import sys
import re
import functools
import traceback

from inspect import ismodule
from inspect import isclass
from loguru import logger

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points as import_entry_points
    from importlib_metadata import EntryPoint
else:
    from importlib.metadata import entry_points as import_entry_points
    from importlib.metadata import EntryPoint

from molfeat.plugins.exception import (
    LoadingEntryPointError,
    MissingEntryPointError,
    MultipleEntryPointError,
)
from molfeat.utils.commons import get_class_name
from . import factories


@functools.lru_cache(maxsize=1)
def eps():
    return import_entry_points()


ENTRY_POINT_GROUP_FACTORYCLASS_MAPPING = {
    "molfeat.calc": factories.CalculatorFactory,
    "molfeat.trans": factories.TransformerFactory,
    "molfeat.trans.pretrained": factories.PretrainedTransformerFactory,
    "molfeat.trans.struct": factories.DefaultFactory,
    "molfeat.trans.graph": factories.DefaultFactory,
    "molfeat.utils": factories.DefaultFactory,
}


def load_entry_point(group: str, name: str) -> Any:
    """
    Load the class registered under the entry point for a given name and group

    Args:
        group: the entry point group
        name: the name of the entry point

    Returns:
        class registered at the given entry point

    Raises:
        MissingEntryPointError: if the entry point was not registered
        MultipleEntryPointError: if the entry point could not be uniquely resolved
        LoadingEntryPointError: if the entry point could not be loaded
    """
    entry_point = get_entry_point(group, name)

    try:
        loaded_entry_point = entry_point.load()
    except ImportError:
        raise LoadingEntryPointError(
            f"Failed to load entry point '{name}':\n{traceback.format_exc()}"
        )

    return loaded_entry_point


def get_entry_points(group: str):
    """
    Return a list of all the entry points within a specific group

    Args:
        group: the entry point group

    Returns:
        a list of entry points
    """
    return eps().select(group=group)


def get_entry_point(group: str, name: str) -> EntryPoint:
    """
    Return an entry point with a given name within a specific group
    Args:
        group: the entry point group
        name: the name of the entry point
    """
    found = eps().select(group=group, name=name)
    if name not in found.names:
        raise MissingEntryPointError(f"Entry point '{name}' not found in group '{group}'")
    # If multiple entry points are found and they have different values we raise, otherwise if they all
    # correspond to the same value, we simply return one of them
    if len(found) > 1 and len(set(ep.value for ep in found)) != 1:
        raise MultipleEntryPointError(
            f"Multiple entry points '{name}' found in group '{group}': {found}"
        )
    return found[name]


@functools.lru_cache(maxsize=100)
def is_registered_entry_point(
    class_module: str, class_name: str, groups: Optional[Sequence[str]] = None
) -> bool:
    """Verify whether the class with the given module and class name is a registered entry point.

    !!! note
        This function only checks whether the class has a registered entry point. It does explicitly not verify
        if the corresponding class is also importable. Use `load_entry_point` for this purpose instead.

    Args:
        class_module: the module of the class
        class_name: the name of the class
        groups: optionally consider only these entry point groups to look for the class

    Returns:
        True if the class is a registered entry point, False otherwise.
    """
    for group in eps().groups if groups is None else groups:
        for entry_point in get_entry_points(group):
            if class_module == entry_point.module and class_name == entry_point.attr:
                return True
    return False


def validate_registered_entry_points():
    """Validate all registered entry points by loading them with the corresponding factory.

    Raises:
        EntryPointError: if any of the registered entry points cannot be loaded. This can happen if:
            * The entry point cannot uniquely be resolved
            * The resource registered at the entry point cannot be imported
            * The resource's type is incompatible with the entry point group that it is defined in.
    """
    for entry_point_group, factory in ENTRY_POINT_GROUP_FACTORYCLASS_MAPPING.items():
        entry_points = get_entry_points(entry_point_group)
        for entry_point in entry_points:
            factory(entry_point.name)


def load_registered_plugins(
    add_submodules: bool = True,
    groups: Optional[List[str]] = None,
    plugins: Optional[List[str]] = None,
    verbose: bool = True,
):
    """Load all registered entry points by loading them with the corresponding factory and adding them to the corresponding module attribute.

    Args:
        add_submodules: if True, add the loaded entry point to the corresponding module attribute.
        groups: if provided, only load entry points from the given groups.
        plugins: if provided, only load entry points or modules/classes that matches entry in the plugins list.
        verbose: if True, log a warning if an entry point cannot be loaded.

    Raises:
        EntryPointError: if any of the registered entry points cannot be loaded. This can happen if:
            * The entry point cannot uniquely be resolved
            * The resource registered at the entry point cannot be imported
            * The resource's type is incompatible with the entry point group that it is defined in.
    """
    for entry_point_group, factory in ENTRY_POINT_GROUP_FACTORYCLASS_MAPPING.items():
        if groups is not None and entry_point_group not in groups:
            continue
        entry_points = get_entry_points(entry_point_group)
        for entry_point in entry_points:
            try:
                loaded_module = factory(entry_point.name, entry_point_group=entry_point_group)
                if _is_valid_plugin(loaded_module, plugins):
                    setattr(
                        sys.modules[entry_point.group],
                        loaded_module.__name__,
                        loaded_module,
                    )
                    if add_submodules:
                        if not ismodule(loaded_module):
                            module_to_add = loaded_module.__module__
                        else:
                            module_to_add = loaded_module
                        sys.modules[f"{entry_point.group}.{entry_point.name}"] = module_to_add
            except AttributeError as e:
                if verbose:
                    logger.warning(
                        f"Could not load entry point {entry_point.name} from group {entry_point.group}"
                    )
                    logger.exception(e)


def _is_valid_plugin(obj: Any, allowed_plugins: Optional[List[str]] = None):
    """Check whether the given object matches any of the allowed  allowed_plugins.

    Args:
        obj: the object to check
        allowed_plugins: a list of regex patterns to match against the object's name

    Returns:
        True if the object matches any of the allowed plugins, False otherwise.
    """

    if allowed_plugins is None or len(allowed_plugins) == 0:
        return True
    if ismodule(obj):
        obj = obj.__name__
    elif isclass(obj):
        obj = get_class_name(obj)
    for plugin in allowed_plugins:
        if re.search(plugin, obj):
            return True
    return False
