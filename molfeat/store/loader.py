import abc
from typing import Optional

import datamol as dm
import platformdirs

from molfeat.store.modelstore import ModelStore, ModelStoreError
from typing import Any


class PretrainedModel(abc.ABC):
    """
    Base class for loading pretrained models
    """

    @abc.abstractmethod
    def _artifact_load(self, **kwargs) -> str:
        """Load an artifact based on its name

        Args:
            name: name of the model to load
        """
        ...

    @abc.abstractmethod
    def load(self) -> Any:
        """Load the model"""
        ...


class PretrainedStoreModel(PretrainedModel):
    """
    Class for loading pretrained models from the model zoo
    """

    def __init__(
        self,
        name: str,
        cache_path: Optional[str] = None,
        store: Optional[ModelStore] = None,
    ):
        """Interface for pretrained model from the default modelstore

        Args:
            name: name of the pretrained transformer in the model store
        """
        self.name = name
        self.cache_path = cache_path or dm.fs.join(platformdirs.user_cache_dir("molfeat"), name)
        self.store = store if store is not None else ModelStore()

    def _artifact_load(self, **kwargs) -> str:
        """Load internal artifact from the model store

        Args:
            name: name of the model to load
            download_path: path to a directory where to save the downloaded files
        """
        try:
            matches = self.store.search(name=self.name)
            if not matches:
                raise ModelStoreError(f"Model {self.name} is not registered in the model store.")
            # ``download`` validates both files and checksums, while avoiding
            # network transfers for an already complete cache. Calling it even
            # when the cache directory exists also repairs interrupted downloads.
            self.store.download(matches[0], self.cache_path, **kwargs)
        except ModelStoreError:
            raise
        except Exception as e:
            raise ModelStoreError(f"Can't retrieve model {self.name} from the store !") from e
        return self.cache_path
