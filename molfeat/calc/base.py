from typing import Optional
import abc
import importlib
import inspect
import json
import joblib
import yaml
import fsspec
import importlib
from loguru import logger
from molfeat._version import __version__ as MOLFEAT_VERSION


_CALCULATORS = {}


class _CalculatorMeta(abc.ABCMeta):
    """Metaclass to register calculator automatically"""

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)
        if name in _CALCULATORS.keys():
            logger.warning(
                f"The {name!r} interaction has been superseded by a "
                f"new class with id {id(cls):#x}"
            )
        if name != "SerializableCalculator":
            # do not register the base class
            _CALCULATORS[name] = cls


class SerializableCalculator(abc.ABC, metaclass=_CalculatorMeta):
    """Interface to define a serializable calculator

    ???+ tip "Subclassing SerializableCalculator"
        When subclassing a calculator, you must implement the __call__ method.
        If your calculator also implements a `batch_compute` method, it will be used
        by `MoleculeTransformer` to accelerate featurization.

        ```python
        from molfeat.calc import SerializableCalculator

        class MyCalculator(SerializableCalculator):

            def __call__(self, mol, **kwargs):
                # you have to implement this
                ...

            def __len__(self):
                # you don't have to implement this but are encouraged to do so
                # this is used to determine the length of the output
                ...

            @property
            def columns(self):
                # you don't have to implement this
                # use this to return the name of each entry returned by your featurizer
                ...

            def batch_compute(self, mols:list, **dm_parallelized_kwargs):
                # you don't need to implement this
                # but you should if there is an efficient batching process
                # By default dm.parallelized arguments will also be passed as input
                ...
        ```
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @classmethod
    def from_state_dict(cls, state: dict, override_args: Optional[dict] = None):
        """Load from state dictionary

        Args:
            state: dictionary to use to create the the calculator
            overrride_args: optional dictionary of arguments to override the ones in the state dict
                at construction of the new object
        """
        cls_name = state.get("name", cls.__name__)
        module_name = state.get("module", cls.__module__)
        module = importlib.import_module(module_name)
        klass = getattr(module, cls_name)
        kwargs = state["args"].copy()
        kwargs.update(**(override_args or {}))
        return klass(**kwargs)

    def to_state_dict(self):
        """Get the state dictionary"""
        state_dict = {}
        state_dict["name"] = self.__class__.__name__
        state_dict["module"] = self.__class__.__module__
        state_dict["args"] = self.__getstate__()
        state_dict["_molfeat_version"] = MOLFEAT_VERSION
        # we would like to keep input arguments as is.
        signature = inspect.signature(self.__init__)
        val = {k: v.default for k, v in signature.parameters.items()}
        to_remove = [k for k in state_dict["args"] if k not in val.keys()]
        for k in to_remove:
            state_dict["args"].pop(k)
        return state_dict

    def to_state_json(self) -> str:
        """Output this instance as a JSON representation"""
        return json.dumps(self.to_state_dict())

    def to_state_yaml(self) -> str:
        """Output this instance as a YAML representation"""
        return yaml.dump(self.to_state_dict(), Dumper=yaml.SafeDumper)

    def to_state_json_file(self, filepath: str):
        """Save the state of this instance as a JSON file"""
        with fsspec.open(filepath, "w") as f:
            f.write(self.to_state_json())  # type: ignore

    def to_state_yaml_file(self, filepath: str):
        """Save the state of this instance as a YAML file"""
        with fsspec.open(filepath, "w") as f:
            f.write(self.to_state_yaml())  # type: ignore

    @classmethod
    def from_state_json(
        cls,
        state_json: str,
        override_args: Optional[dict] = None,
    ):
        state_dict = yaml.safe_load(state_json)
        return cls.from_state_dict(state_dict, override_args=override_args)

    @classmethod
    def from_state_yaml(
        cls,
        state_yaml: str,
        override_args: Optional[dict] = None,
    ):
        state_dict = yaml.load(state_yaml, Loader=yaml.SafeLoader)
        return cls.from_state_dict(state_dict, override_args=override_args)

    @classmethod
    def from_state_json_file(
        cls,
        filepath: str,
        override_args: Optional[dict] = None,
    ):
        with fsspec.open(filepath, "r") as f:
            featurizer = cls.from_state_json(f.read(), override_args=override_args)  # type: ignore
        return featurizer

    @classmethod
    def from_state_yaml_file(
        cls,
        filepath: str,
        override_args: Optional[dict] = None,
    ):
        with fsspec.open(filepath, "r") as f:
            featurizer = cls.from_state_yaml(f.read(), override_args=override_args)  # type: ignore
        return featurizer

    @classmethod
    def from_state_file(
        cls,
        state_path: str,
        override_args: Optional[dict] = None,
    ):
        if state_path.endswith("yaml") or state_path.endswith("yml"):
            return cls.from_state_yaml_file(filepath=state_path, override_args=override_args)
        elif state_path.endswith("json"):
            return cls.from_state_json_file(filepath=state_path, override_args=override_args)
        elif state_path.endswith("pkl"):
            with fsspec.open(state_path, "rb") as IN:
                return joblib.load(IN)
        raise ValueError(
            "Only files with 'yaml' or 'json' format are allowed. "
            "The filename must be ending with `yaml`, 'yml' or 'json'."
        )
