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
    def _artifact_load(self, name: str, download_path: Optional[str] = None, **kwargs) -> str:
        """Load an artifact based on its name

        Args:
            name: name of the model to load
            download_path: path to a directory where to save the downloaded files
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
            cache_path: optional local cache path.
            store: ModelStore to use for loading the pretrained model
        """
        if store is None:
            store = ModelStore()

        self.name = name
        self.cache_path = cache_path if cache_path else platformdirs.user_cache_dir("molfeat")
        self.store = store

    def _artifact_load(self, name: str, download_path: Optional[str] = None, **kwargs) -> str:
        """Load internal artifact from the model store

        Args:
            name: name of the model to load
            download_path: path to a directory where to save the downloaded files
        """
        path = download_path if download_path else dm.fs.join(self.cache_path, name)

        if not dm.fs.exists(path):
            try:
                modelcard = self.store.search(name=name)[0]
                self.store.download(modelcard, download_path, **kwargs)
            except Exception as e:
                raise ModelStoreError(f"Can't retrieve model {name} from the store !") from e
        return path
