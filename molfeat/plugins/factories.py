from inspect import isclass
from inspect import ismodule

from typing import TYPE_CHECKING
from typing import overload
from typing import Any
from typing import Union
from typing import Optional
from typing import Callable
from typing import Literal
from typing import NoReturn
from typing import Tuple
from typing import Type

import sys

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint
else:
    from importlib.metadata import EntryPoint

from molfeat.plugins.exception import InvalidEntryPointTypeError

__all__ = (
    "CalculatorFactory",
    "TransformerFactory",
    "PretrainedTransformerFactory",
    "DefaultFactory",
)

if TYPE_CHECKING:
    from molfeat.calc import SerializableCalculator
    from molfeat.trans import MoleculeTransformer
    from molfeat.trans.pretrained import PretrainedMolTransformer


def raise_invalid_type_error(
    entry_point_name: str, entry_point_group: str, valid_classes: Tuple[Any, ...]
) -> NoReturn:
    """Raise an `InvalidEntryPointTypeError` with formatted message.

    Args:
        entry_point_name: name of the entry point
        entry_point_group: name of the entry point group
        valid_classes: tuple of valid classes for the given entry point group

    Raises:
        InvalidEntryPointTypeError: always
    """
    template = "entry point `{}` registered in group `{}` is invalid because its type is not one of the supported types ({})"
    args = (
        entry_point_name,
        entry_point_group,
        ", ".join([e.__name__ for e in valid_classes]),
    )
    raise InvalidEntryPointTypeError(template.format(*args))


def BaseFactory(group: str, name: str, load: bool = True) -> Union[EntryPoint, Any]:
    """Return the plugin class registered under a given entry point group and name.

    Args:
        group: entry point group
        name: entry point name
        load: if True, load the matched entry point and return the loaded resource instead of the entry point itself.
    Return:
        the plugin class
    Raises:
        MissingEntryPointError: entry point was not registered
        MultipleEntryPointError: entry point could not be uniquely resolved
        LoadingEntryPointError: entry point could not be loaded
    """
    # circular import
    from .entry_point import get_entry_point, load_entry_point

    if load is True:
        return load_entry_point(group, name)

    return get_entry_point(group, name)


@overload
def CalculatorFactory(
    entry_point_name: str,
    load: Literal[True] = True,
    entry_point_group: Optional[str] = None,
) -> Union[Type["SerializableCalculator"], Callable]:
    ...


@overload
def CalculatorFactory(
    entry_point_name: str, load: Literal[False], entry_point_group: Optional[str] = None
) -> EntryPoint:
    ...


def CalculatorFactory(
    entry_point_name: str,
    load: bool = True,
    entry_point_group: Optional[str] = None,
) -> Union[EntryPoint, Type["SerializableCalculator"], Callable]:
    """Return the `SerializableCalculator` sub class registered under the given entry point.

    Args:
        entry_point_name: the entry point name.
        load: if True, load the matched entry point and return the loaded resource instead of the entry point itself.
        entry_point_group: the optional entry point group to use

    Return:
        sub class of :py:class:`~molfeat.calc.SerializableCalculator`
    """
    from molfeat.calc import SerializableCalculator

    if entry_point_group is None:
        entry_point_group = "molfeat.calc"
    entry_point = BaseFactory(entry_point_group, entry_point_name, load=load)
    valid_classes = (SerializableCalculator,)

    if not load:
        return entry_point

    # if the entry point is a module, nothing to do
    if ismodule(entry_point):
        return entry_point
    if isclass(entry_point) and issubclass(entry_point, valid_classes):
        return entry_point

    raise_invalid_type_error(entry_point_name, entry_point_group, valid_classes)


@overload
def TransformerFactory(
    entry_point_name: str,
    load: Literal[True] = True,
    entry_point_group: Optional[str] = None,
) -> Union[Type["MoleculeTransformer"], Callable]:
    ...


@overload
def TransformerFactory(
    entry_point_name: str, load: Literal[False], entry_point_group: Optional[str] = None
) -> EntryPoint:
    ...


def TransformerFactory(
    entry_point_name: str,
    load: bool = True,
    entry_point_group: Optional[str] = None,
) -> Union[EntryPoint, Type["MoleculeTransformer"], Callable]:
    """Return the `MoleculeTransformer` sub class registered under the given entry point.

    Args:
        entry_point_name: the entry point name.
        load: if True, load the matched entry point and return the loaded resource instead of the entry point itself.
        entry_point_group: the optional entry point group to use

    Returns:
        sub class of :py:class:`~molfeat.trans.MoleculeTransformer`

    Raise:
        InvalidEntryPointTypeError: if the type of the loaded entry point is invalid.
    """
    from molfeat.trans import MoleculeTransformer
    from molfeat.trans import BaseFeaturizer

    if entry_point_group is None:
        entry_point_group = "molfeat.trans"
    entry_point = BaseFactory(entry_point_group, entry_point_name, load=load)
    valid_classes = (MoleculeTransformer, BaseFeaturizer)

    if not load:
        return entry_point

    # if the entry point is a module, nothing to do
    if ismodule(entry_point):
        return entry_point
    if isclass(entry_point) and issubclass(entry_point, valid_classes):
        return entry_point

    raise_invalid_type_error(entry_point_name, entry_point_group, valid_classes)


@overload
def PretrainedTransformerFactory(
    entry_point_name: str,
    load: Literal[True] = True,
    entry_point_group: Optional[str] = None,
) -> Union[Type["PretrainedMolTransformer"], Callable]:
    ...


@overload
def PretrainedTransformerFactory(entry_point_name: str, load: Literal[False]) -> EntryPoint:
    ...


def PretrainedTransformerFactory(
    entry_point_name: str,
    load: bool = True,
    entry_point_group: Optional[str] = None,
) -> Union[EntryPoint, Type["PretrainedMolTransformer"], Callable]:
    """Return the PretrainedMolTransformer sub class registered under the given entry point.

    Args:
        entry_point_name: the entry point name.
        load: if True, load the matched entry point and return the loaded resource instead of the entry point itself.
        entry_point_group: the optional entry point group to use

    Returns:
        sub class of :py:class:`~molfeat.trans.pretrained.PretrainedMolTransformer`

    Raise:
        InvalidEntryPointTypeError: if the type of the loaded entry point is invalid.
    """
    from molfeat.trans import MoleculeTransformer
    from molfeat.trans.pretrained import PretrainedMolTransformer

    if entry_point_group is None:
        entry_point_group = "molfeat.trans.pretrained"
    entry_point = BaseFactory(entry_point_group, entry_point_name, load=load)
    valid_classes = (PretrainedMolTransformer, MoleculeTransformer)

    if not load:
        return entry_point
    # if the entry point is a module, nothing to do
    if ismodule(entry_point):
        return entry_point
    if isclass(entry_point) and issubclass(entry_point, valid_classes):
        return entry_point

    raise_invalid_type_error(entry_point_name, entry_point_group, valid_classes)


@overload
def DefaultFactory(
    entry_point_name: str,
    load: Literal[True] = True,
    entry_point_group: str = None,
) -> Union[Type["PretrainedMolTransformer"], Callable]:
    ...


@overload
def DefaultFactory(
    entry_point_name: str, load: Literal[False], entry_point_group: str = None
) -> EntryPoint:
    ...


def DefaultFactory(
    entry_point_name: str,
    load: bool = True,
    entry_point_group: str = None,
) -> Union[EntryPoint, Type["PretrainedMolTransformer"], Callable]:
    """Return the Default factory for extending capabilities given a specific module.

    Args:
        entry_point_name: the entry point name.
        load: if True, load the matched entry point and return the loaded resource instead of the entry point itself.
        entry_point_group: the optional entry point group to use

    Returns:
        sub class or module of

    Raise:
        InvalidEntryPointTypeError: if the type of the loaded entry point is invalid.
    """

    if entry_point_group is None:
        entry_point_group = "molfeat"
    entry_point = BaseFactory(entry_point_group, entry_point_name, load=load)

    if not load:
        return entry_point
    # if the entry point is a module, nothing to do
    if ismodule(entry_point):
        return entry_point
    raise_invalid_type_error(entry_point_name, entry_point_group, ())
