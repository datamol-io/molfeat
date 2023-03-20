from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing_extensions import Literal

import inspect
import types
import importlib


def import_from_string(import_path: str) -> Any:
    """Import a class from a string."""

    import_classname = import_path.split(".")[-1]
    import_module = ".".join(import_path.split(".")[:-1])

    module = importlib.import_module(import_module)
    return getattr(module, import_classname)


def get_input_args():
    frame = None
    current_frame = inspect.currentframe()

    if current_frame:
        frame = current_frame.f_back

    if not isinstance(frame, types.FrameType):
        raise AttributeError("There is no `frame` available while being required.")

    init_args = {}
    for local_args in collect_init_args(frame, []):
        init_args.update(local_args)

    return init_args


# NOTE(hadim): the below is coming from PL.
# https://github.com/PyTorchLightning/pytorch-lightning/blob/87bd54aedfcd990b935860d020ca89d192e2ba40/pytorch_lightning/utilities/parsing.py#L161


def parse_class_init_keys(cls) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse key words for standard ``self``, ``*args`` and ``**kwargs``.
    Examples:
        >>> class Model():
        ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
        ...         pass
        >>> parse_class_init_keys(Model)
        ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: List[inspect.Parameter],
        param_type: Literal[
            inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD
        ],
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def get_init_args(frame: types.FrameType) -> Dict[str, Any]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return {}
    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters.keys()}
    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    return local_args


def collect_init_args(
    frame: types.FrameType, path_args: List[Dict[str, Any]], inside: bool = False
) -> List[Dict[str, Any]]:
    """Recursively collects the arguments passed to the child constructors in the inheritance tree.
    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon
    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)
    # frame.f_back must be of a type types.FrameType for get_init_args/collect_init_args due to mypy
    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    if "__class__" in local_vars:
        local_args = get_init_args(frame)
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True)
    if not inside:
        return collect_init_args(frame.f_back, path_args, inside)
    return path_args
