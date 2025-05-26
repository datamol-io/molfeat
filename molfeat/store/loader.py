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
        self.store = store or ModelStore()

    def _artifact_load(self, **kwargs) -> str:
        """Load internal artifact from the model store

        Args:
            name: name of the model to load
            download_path: path to a directory where to save the downloaded files
        """
        if not dm.fs.exists(self.cache_path):
            try:
                modelcard = self.store.search(name=self.name)[0]
                self.store.download(modelcard, self.cache_path, **kwargs)
            except Exception as e:
                raise ModelStoreError(f"Can't retrieve model {self.name} from the store !") from e
        return self.cache_path
