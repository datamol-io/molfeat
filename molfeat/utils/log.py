from typing import Union
from typing import List
from loguru import logger


class loguru_off:
    def __init__(self, modules: Union[str, List[str]]):
        """Disable loguru logging for a list of modules

        Args:
            modules: list of modules to temporary disable
        """
        if isinstance(modules, str):
            modules = [modules]
        self.modules = modules
        self._status = logger._core.enabled.copy()

    def __enter__(self):
        for mod in self.modules:
            logger.disable(mod)

    def __exit__(self, *args, **kwargs):
        for mod in self.modules:
            matching_mods = [x for x in self._status.keys() if x.startswith(mod)]
            if len(matching_mods) > 0:
                for mod_match in matching_mods:
                    logger._change_activation(mod_match, self._status[mod_match])
            else:
                # if it was not in a known state before, we will enable it
                logger.enable(mod)
