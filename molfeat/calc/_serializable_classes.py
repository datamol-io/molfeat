from typing import Optional
from typing import Dict
from typing import Any

from rdkit.Chem import rdFingerprintGenerator

SERIALIZABLE_CLASSES = {}


def register_custom_serializable_class(cls: type):
    SERIALIZABLE_CLASSES[cls.__name__] = cls
    return cls


@register_custom_serializable_class
class SerializableMorganFeatureAtomInvGen:
    """A serializable wrapper class for `rdFingerprintGenerator.GetMorganFeatureAtomInvGen()`"""

    def __init__(self):
        self._generator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()

    def __getstate__(self):
        return None

    def __setstate__(self, state: Optional[None]):
        self._generator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()

    def __deepcopy__(self, memo: Dict[int, Any]):
        new_instance = SerializableMorganFeatureAtomInvGen()
        memo[id(self)] = new_instance
        return new_instance

    def __getattr__(self, name: str):
        try:
            generator = object.__getattribute__(self, "_generator")
        except AttributeError:
            raise AttributeError("'_generator' is not initialized")

        try:
            return getattr(generator, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


@register_custom_serializable_class
class SerializableMorganFeatureBondInvGen:
    """A serializable wrapper class for `rdFingerprintGenerator.GetMorganFeatureBondInvGen()`"""

    def __init__(self):
        self._generator = rdFingerprintGenerator.GetMorganFeatureBondInvGen()

    def __getstate__(self):
        return None

    def __setstate__(self, state: Optional[None]):
        self._generator = rdFingerprintGenerator.GetMorganFeatureBondInvGen()

    def __deepcopy__(self, memo: Dict[int, Any]):
        new_instance = SerializableMorganFeatureBondInvGen()
        memo[id(self)] = new_instance
        return new_instance

    def __getattr__(self, name: str):
        try:
            generator = object.__getattribute__(self, "_generator")
        except AttributeError:
            raise AttributeError("'_generator' is not initialized")

        try:
            return getattr(generator, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
