from typing import Optional

import abc
import os
import datamol as dm
from functools import lru_cache
from molfeat.store.modelstore import ModelStore
from molfeat.store.modelstore import ModelStoreError


class PretrainedModel(abc.ABC):
    """Base class for loading pretrained models"""

    @classmethod
    def _artifact_load(cls, name: str, download_path: Optional[os.PathLike] = None, **kwargs):
        """Load an artifact based on its name

        Args:
            name: name of the model to load
            download_path: path to a directory where to save the downloaded files
        """
        ...

    @classmethod
    def _load_or_raise(
        cls,
        name: str,
        download_path: Optional[os.PathLike] = None,
        **kwargs,
    ):
        """Load model or raise an exception

        Args:
            name: name of the model to load
            download_path: local download path of the model

        """
        ...

    @abc.abstractmethod
    def load(self):
        """Load the model"""
        ...


class PretrainedStoreModel(PretrainedModel):
    r"""
    Class for loading pretrained models from the model zoo
    """

    def __init__(
        self,
        name: str,
        cache_path: Optional[os.PathLike] = None,
        store: Optional[ModelStore] = None,
    ):
        """Interface for pretrained model from the default modelstore

        Args:
            name: name of the pretrained transformer in the model store
            cache_path: optional local cache path.
            store: ModelStore to use for loading the pretrained model
        """
        self.name = name
        self.cache_path = cache_path
        if store is None:
            store = ModelStore()
        self.store = store

    @classmethod
    def _artifact_load(cls, name: str, download_path: Optional[os.PathLike] = None, **kwargs):
        """Load internal artefact from the model store

        Args:
            name: name of the model to load
            download_path: path to a directory where to save the downloaded files
        """

        if not dm.fs.exists(download_path):
            cls._load_or_raise.cache_clear()
        return cls._load_or_raise(name, download_path, **kwargs)

    @classmethod
    @lru_cache(maxsize=100)
    def _load_or_raise(
        cls,
        name: str,
        download_path: Optional[os.PathLike] = None,
        store: Optional[ModelStore] = None,
        **kwargs,
    ):
        """Load when from ada or raise exception
        Args:
            name: name
        """
        if store is None:
            store = ModelStore()
        try:
            modelcard = store.search(name=name)[0]
            artifact_dir = store.download(modelcard, download_path, **kwargs)
        except Exception as e:
            mess = f"Can't retrieve model {name} from the store !"
            raise ModelStoreError(mess)
        return artifact_dir

    def load(self):
        """Load the model"""
        raise NotImplementedError
