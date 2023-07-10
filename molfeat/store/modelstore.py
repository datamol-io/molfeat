import os
import pathlib
import tempfile
from typing import Any, Callable, Optional, Union

import datamol as dm
import filelock
import fsspec
import joblib
import platformdirs
import yaml
from dotenv import load_dotenv
from loguru import logger

from molfeat.store.modelcard import ModelInfo
from molfeat.utils import commons

load_dotenv()


class ModelStoreError(Exception):
    pass


class ModelStore:
    """A class for artefact serializing from any url

    This class not only allow pretrained model serializing and loading,
    but also help in listing model availability and registering models.

    For simplicity.
        * There is no versioning.
        * Only one model should match a given name
        * Model deletion is not allowed (on the read-only default store)
        * Only a single store is supported per model store instance

    !!! note "Building a New Model Store"
        To create a new model store, you will mainly need a model store bucket path. The default model store bucket, located at `gs://molfeat-store-prod/artifacts/`, is **read-only**.

        To build your own model store bucket, follow the instructions below:

        1. Create a local or remote cloud directory that can be accessed by fsspec (and the corresponding filesystem).
        2. [Optional] Sync the default model store bucket to your new path if you want to access the default models.
        3. Set the environment variable `MOLFEAT_MODEL_STORE_BUCKET` to your new path. This variable will be used as the default model store bucket when creating a new model store instance without specifying a path.
            Note that setting up this path is necessary if you want to access models directly by their names, without manually loading them from your custom model store.


    """

    # EN: be careful not to recreate ada
    # EN: should we just use modelstore ?
    MODEL_STORE_BUCKET = "gs://molfeat-store-prod/artifacts/"
    MODEL_PATH_NAME = "model.save"
    METADATA_PATH_NAME = "metadata.json"

    def __init__(self, model_store_bucket: Optional[str] = None):
        if model_store_bucket is None:
            model_store_bucket = os.getenv("MOLFEAT_MODEL_STORE_BUCKET", self.MODEL_STORE_BUCKET)
        self.model_store_bucket = model_store_bucket
        self._available_models = []
        self._update_store()

    def _update_store(self):
        """Initialize the store with all available models"""
        all_metadata = dm.fs.glob(dm.fs.join(self.model_store_bucket, "**/metadata.json"))
        self._available_models = []
        for mtd_file in all_metadata:
            with fsspec.open(mtd_file, "r") as IN:
                mtd_content = yaml.safe_load(IN)
                model_info = ModelInfo(**mtd_content)
                self._available_models.append(model_info)

    @property
    def available_models(self):
        """Return a list of all models that have been serialized in molfeat"""
        return self._available_models

    def __len__(self):
        """Return the length of the model store"""
        return len(self.available_models)

    def register(
        self,
        modelcard: Union[ModelInfo, dict],
        model: Optional[Any] = None,
        chunk_size: int = 2048,
        save_fn: Optional[Callable] = None,
        save_fn_kwargs: Optional[dict] = None,
        force: bool = True,
    ):
        """
        Register a new model to the store

        !!! note `save_fn`
            You can pass additional kwargs for your `save_fn` through the `save_fn_kwargs` argument.
            It's expected that `save_fn` will be called as : `save_fn(model, <model_upload_path>, **save_fn_wargs)`,
            with `<model_upload_path>` being provided by the model store, and that it will return the path to the serialized model.
            If not provided, `joblib.dump` is used by default.

        Args:
            modelcard: Model information
            model: A path to the model artifact or any object that needs to be saved
            chunk_size: the chunk size for the upload
            save_fn: any custom function for serializing the model, that takes the model, the upload path and parameters `save_fn_kwargs` as inputs.
            save_fn_kwargs: any additional kwargs to pass to save_fn
            force: whether to force upload to the bucket

        """
        if not isinstance(modelcard, ModelInfo):
            modelcard = ModelInfo(**modelcard)
        # we save the model first
        if self.exists(card=modelcard):
            logger.warning(f"Model {modelcard.name} exists already ...")
            if not force:
                return

        model_root_dir = modelcard.path(self.model_store_bucket)
        model_path = model
        model_upload_path = dm.fs.join(model_root_dir, self.MODEL_PATH_NAME)
        model_metadata_upload_path = dm.fs.join(model_root_dir, self.METADATA_PATH_NAME)

        save_fn_kwargs = save_fn_kwargs or {}
        if save_fn is None:
            if not isinstance(model, (pathlib.Path, os.PathLike)):
                local_model_path = tempfile.NamedTemporaryFile(delete=False)
                with local_model_path as f:
                    joblib.dump(model, local_model_path)
                model_path = local_model_path.name
            # Upload the artifact to the bucket
            dm.fs.copy_file(
                model_path,
                model_upload_path,
                progress=True,
                leave_progress=False,
                chunk_size=chunk_size,
                force=force,
            )
        else:
            model_path = save_fn(model, model_upload_path, **save_fn_kwargs)
            # we reset to None if the save_fn has not returned anything
            model_path = model_path or model_upload_path
        modelcard.sha256sum = commons.sha256sum(model_path)
        # then we save the metadata as json
        with fsspec.open(model_metadata_upload_path, "w") as OUT:
            OUT.write(modelcard.json())
        self._update_store()
        logger.info(f"Successfuly registered model {modelcard.name} !")

    def _filelock(self, lock_name: str):
        """Create an empty lock file into `cache_dir_path/locks/lock_name`"""

        lock_path = dm.fs.join(
            str(platformdirs.user_cache_dir("molfeat")), "_lock_files", lock_name
        )
        mapper = dm.fs.get_mapper(lock_path)
        # ensure file is created
        # out = mapper.fs.touch(lock_path) # does not work  -_-
        with fsspec.open(lock_path, "w", auto_mkdir=True) as f:
            pass

        return filelock.FileLock(lock_path)

    def download(
        self,
        modelcard: ModelInfo,
        output_dir: Optional[Union[os.PathLike, pathlib.Path]] = None,
        chunk_size: int = 2048,
        force: bool = False,
    ):
        """Download an artifact locally

        Args:
            modelcard: information on the model to download
            output_dir: path where to save the downloaded artifact
            chunk_size: chunk size to use for download
            force: whether to force download even if the file exists already
        """

        remote_dir = modelcard.path(self.model_store_bucket)
        model_name = modelcard.name
        if not self.exists(modelcard, check_remote=True):
            raise ModelStoreError(f"Model {model_name} does not exist in the model store !")

        if output_dir is None:
            output_dir = dm.fs.join(platformdirs.user_cache_dir("molfeat"), model_name)

        dm.fs.mkdir(output_dir, exist_ok=True)

        model_remote_path = dm.fs.join(remote_dir, self.MODEL_PATH_NAME)
        model_dest_path = dm.fs.join(output_dir, self.MODEL_PATH_NAME)
        metadata_remote_path = dm.fs.join(remote_dir, self.METADATA_PATH_NAME)
        metadata_dest_path = dm.fs.join(output_dir, self.METADATA_PATH_NAME)

        # avoid downloading if the file exists already
        if (
            not (
                dm.fs.exists(metadata_dest_path)
                and (dm.fs.exists(model_dest_path) == dm.fs.exists(model_remote_path))
            )
            or force
        ):
            # metadata should exists if the model exists
            with self._filelock(f"{model_name}.metadata.json.lock"):
                dm.fs.copy_file(
                    metadata_remote_path,
                    metadata_dest_path,
                    progress=True,
                    leave_progress=False,
                    force=True,
                )

            if dm.fs.exists(model_remote_path):
                with self._filelock(f"{model_name}.lock"):
                    if dm.fs.is_dir(model_remote_path):
                        # we copy the model dir
                        dm.fs.copy_dir(
                            model_remote_path,
                            model_dest_path,
                            progress=True,
                            leave_progress=False,
                            chunk_size=chunk_size,
                            force=force,
                        )
                    else:
                        # we copy the model dir
                        dm.fs.copy_file(
                            model_remote_path,
                            model_dest_path,
                            progress=True,
                            leave_progress=False,
                            chunk_size=chunk_size,
                            force=force,
                        )

        cache_sha256sum = commons.sha256sum(model_dest_path)
        if modelcard.sha256sum is not None and cache_sha256sum != modelcard.sha256sum:
            mapper = dm.fs.get_mapper(output_dir)
            mapper.fs.delete(output_dir, recursive=True)
            raise ModelStoreError(
                f"""The destination artifact at {model_dest_path} has a different sha256sum ({cache_sha256sum}) """
                f"""than the Remote artifact sha256sum ({modelcard.sha256sum}). The destination artifact has been removed !"""
            )

        return output_dir

    def load(
        self,
        model_name: Union[str, dict, ModelInfo],
        load_fn: Optional[Callable] = None,
        load_fn_kwargs: Optional[dict] = None,
        download_output_dir: Optional[Union[os.PathLike, pathlib.Path]] = None,
        chunk_size: int = 2048,
        force: bool = False,
    ):
        """
        Load a model by its name

        Args:
            model_name: name of the model to load
            load_fn: Custom loading function to load the model
            load_fn_kwargs: Optional dict of additional kwargs to provide to the loading function
            download_output_dir: Argument for download function to specify the download folder
            chunk_size: chunk size for download
            force: whether to reforce the download of the file

        Returns:
            model: Optional model, if the model requires download or loading weights
            model_info: model information card
        """
        if isinstance(model_name, str):
            # find the model with the same name
            modelcard = self.search(name=model_name)[0]
        else:
            modelcard = model_name
        output_dir = self.download(
            modelcard=modelcard,
            output_dir=download_output_dir,
            chunk_size=chunk_size,
            force=force,
        )
        if load_fn is None:
            load_fn = joblib.load
        model_path = dm.fs.join(output_dir, self.MODEL_PATH_NAME)
        metadata_path = dm.fs.join(output_dir, self.METADATA_PATH_NAME)

        # deal with non-pretrained models that might not have a serialized file
        model = None
        load_fn_kwargs = load_fn_kwargs or {}
        if dm.fs.exists(model_path):
            model = load_fn(model_path, **load_fn_kwargs)
        with fsspec.open(metadata_path, "r") as IN:
            model_info_dict = yaml.safe_load(IN)
        model_info = ModelInfo(**model_info_dict)
        return model, model_info

    def __contains__(self, card: Optional[ModelInfo] = None):
        return self.exists(card)

    def exists(
        self,
        card: Optional[ModelInfo] = None,
        check_remote: bool = False,
        **match_params,
    ) -> bool:
        """Returns True if a model is registered in the store

        Args:
            card: card of the model to check
            check_remote: whether to check if the remote path of the model exists
            match_params: parameters for matching as expected by `ModelInfo.match`
        """

        found = False
        for model_info in self.available_models:
            if model_info.match(card, **match_params):
                found = True
                break
        return found and (not check_remote or dm.fs.exists(card.path(self.model_store_bucket)))

    def search(self, modelcard: Optional[ModelInfo] = None, **search_kwargs):
        """ "Return all model card that match the required search parameters

        Args:
            modelcard: model card to search for
            search_kwargs: search parameters to use
        """
        search_infos = {}
        found = []
        if modelcard is not None:
            search_infos = modelcard.dict().copy()
        search_infos.update(search_kwargs)
        for model in self.available_models:
            if model.match(search_infos, match_only=list(search_infos.keys())):
                found.append(model)
        return found
