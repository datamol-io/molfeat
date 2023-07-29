from __future__ import annotations

from typing import List
from typing import Union
from typing import Optional

import os
import uuid
import torch
import tempfile
import numpy as np
import datamol as dm

from dataclasses import dataclass
from loguru import logger
from transformers import EncoderDecoderModel
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import MODEL_MAPPING
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from molfeat.trans.pretrained.base import PretrainedMolTransformer
from molfeat.store.loader import PretrainedStoreModel
from molfeat.store import ModelStore
from molfeat.store import ModelInfo
from molfeat.utils.converters import SmilesConverter
from molfeat.utils.pooler import get_default_hgf_pooler


@dataclass
class HFExperiment:
    model: PreTrainedModel
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    notation: str = "smiles"

    @classmethod
    def save(cls, model: HFExperiment, path: str, clean_up: bool = False):
        """Save a hugging face model to a specific path

        Args:
            model: model to save
            path: path to the folder root where to save the model
            clean_up: whether to clean up the local path after saving
        """
        local_path = tempfile.mkdtemp()
        # we can save both the tokenizer and the model to the same path
        model.model.save_pretrained(local_path)
        model.tokenizer.save_pretrained(local_path)
        dm.fs.copy_dir(local_path, path, force=True, progress=True, leave_progress=False)
        logger.info(f"Model saved to {path}")
        # clean up now
        if clean_up:
            mapper = dm.fs.get_mapper(local_path)
            mapper.fs.delete(local_path, recursive=True)
        return path

    @classmethod
    def load(cls, path: str, model_class=None):
        """Load a model from the given path
        Args:
            path: Path to the model to load
            model_class: optional model class to provide if the model should be loaded with a specific class
        """
        if not dm.fs.is_local_path(path):
            local_path = tempfile.mkdtemp()
            dm.fs.copy_dir(path, local_path, force=True, progress=True, leave_progress=False)
        else:
            local_path = path

        if model_class is None:
            model_config = AutoConfig.from_pretrained(local_path)
            architectures = getattr(model_config, "architectures", [])
            if len(architectures) > 0:
                model_class = MODEL_MAPPING._load_attr_from_module(
                    model_config.model_type, architectures[0]
                )
            else:
                model_class = AutoModel
        model = model_class.from_pretrained(local_path)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        return cls(model, tokenizer)


class HFModel(PretrainedStoreModel):
    """Transformer model loading model loading"""

    def __init__(
        self,
        name: str,
        cache_path: Optional[os.PathLike] = None,
        store: Optional[ModelStore] = None,
    ):
        """Model loader initializer

        Args:
            name (str, optional): Name of the model for ada.
            cache_path (os.PathLike, optional): Local cache path for faster loading. This is the cache_path parameter for ADA loading !
        """

        super().__init__(name, cache_path=cache_path, store=store)
        self._model = None

    @classmethod
    def _ensure_local(cls, object_path: Union[str, os.PathLike]):
        """Make sure the input path is a local path otherwise download it

        Args:
            object_path: Path to the object

        """
        if dm.fs.is_local_path(object_path):
            return object_path
        local_path = tempfile.mkdtemp()
        if dm.fs.is_file(object_path):
            local_path = os.path.join(local_path, os.path.basename(object_path))
            dm.fs.copy_file(object_path, local_path)
        else:
            dm.fs.copy_dir(object_path, local_path)
        return local_path

    @classmethod
    def from_pretrained(
        cls,
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        model_class=None,
        model_name: Optional[str] = None,
    ):
        """Load model using huggingface pretrained model loader hook

        Args:
            model: Model to load. Can also be the name on the hub or the path to the model
            tokenizer: Tokenizer to load. Can also be the name on the hub or the path to the tokenizer
            model_class: optional model class to provide if the model should be loaded with a specific class
            model_name: optional model name to give to this model.
        """

        # load the model
        if isinstance(model, PreTrainedModel):
            model_obj = model
        else:
            if dm.fs.exists(model):
                model = cls._ensure_local(model)
            if model_class is None:
                model_config = AutoConfig.from_pretrained(model)
                architectures = getattr(model_config, "architectures", [])
                if len(architectures) > 0:
                    model_class = MODEL_MAPPING._load_attr_from_module(
                        model_config.model_type, architectures[0]
                    )
                else:
                    model_class = AutoModel
            model_obj = model_class.from_pretrained(model)

        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            tokenizer_obj = tokenizer
        else:
            if dm.fs.exists(tokenizer):
                tokenizer = cls._ensure_local(tokenizer)
            tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer)
        name = model_name or f"hf_model_{uuid.uuid4().hex[:8]}"
        model = HFModel(name=name, store=ModelStore())
        model._model = HFExperiment(model=model_obj, tokenizer=tokenizer_obj)
        return model

    @classmethod
    def register_pretrained(
        cls,
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        model_card: ModelInfo,
        model_class=None,
    ):
        """Register a pretrained huggingface model to the model store
        Args:
            model: Model to load. Can also be the name on the hub or the path to the model
            tokenizer: Tokenizer to load. Can also be the name on the hub or the path to the tokenizer
            model_class: optional model class to provide if the model should be loaded with a specific class
            model_card: optional model card to provide for registering this model
        """
        model = cls.from_pretrained(model, tokenizer, model_class, model_name=model_card.name)
        model.store.register(model_card, model._model, save_fn=HFExperiment.save)
        return model

    def get_notation(self, default_notation: Optional[str] = None):
        """Get the notation of the model"""
        notation = default_notation
        try:
            modelcard = self.store.search(name=self.name)[0]
            notation = modelcard.inputs
        except Exception:
            pass
        return notation

    def load(self):
        """Load Transformer Pretrained featurizer model"""
        if self._model is not None:
            return self._model
        download_output_dir = self._artifact_load(
            name=self.name, download_path=self.cache_path, store=self.store
        )
        model_path = dm.fs.join(download_output_dir, self.store.MODEL_PATH_NAME)
        model = HFExperiment.load(model_path)
        return model


class PretrainedHFTransformer(PretrainedMolTransformer):
    """
    HuggingFace Transformer for feature extraction.

    !!! note
        For convenience and consistency, this featurizer only accepts as inputs
        smiles and molecules, then perform the internal conversion, based on the notation provided.
    """

    NEEDS_RANDOM_SEED = ["bert", "roberta"]

    def __init__(
        self,
        kind: Union[str, HFModel] = "ChemBERTa-77M-MLM",
        notation: Optional[str] = "none",
        pooling: str = "mean",
        concat_layers: Union[List[int], int] = -1,
        prefer_encoder: bool = True,
        dtype=np.float32,
        device="cpu",
        max_length: int = 128,
        ignore_padding: bool = True,
        preload: bool = False,
        n_jobs: int = 0,
        random_seed: Optional[int] = None,
        **params,
    ):
        """
        HuggingFace Transformer for featurizer extraction
        The default behaviour of this feature extractor is to return the last hidden state of the encoder
        similar to what is performed by the pipeline 'feature-extraction' in hugging face.

        !!! warning
            For bert models, the default pooling layers is a neural network. Therefore, do not use the default
            Or provide a random seed for reproducibility (in this case pooling will act as random projection to the same manifold)

        !!! note
            The pooling module of this featurizer is accessible through the `_pooling_obj` attribute.

        Args:
            kind: name of the featurizer as available in the model store
            notation: optional line notation to use. Only use if it cannot be found from the model card.
            pooling: type of pooling to use. One of ['default', 'mean', 'max', 'sum', 'clf', ]. The value "default" corresponds to the default litterature pooling for each model type.
                See `molfeat.utils.pooler.get_default_hf_pooler` for more details.
            concat_layers: Layer to concat to get the representation. By default the last hidden layer is returned.
            prefer_encoder: For an encoder-decoder model, prefer the embeddings provided by the encoder.
            dtype: Data type to output
            device: Torch device on which to run the featurizer.
            max_length: Maximum length of the input sequence to consider. Please update this for large sequences
            ignore_padding: Whether to ignore padding in the representation (default: True) to avoid effect of batching
            preload: Whether to preload the model into memory or not
            n_jobs: number of jobs to use
            random_seed: random seed to use for reproducibility whenever a DNN pooler is used (e.g bert/roberta)
        """

        super().__init__(
            dtype=dtype,
            device=device,
            n_jobs=n_jobs,
            **params,
        )
        if concat_layers is None:
            concat_layers = -1
        if not isinstance(concat_layers, list):
            concat_layers = [concat_layers]
        self.concat_layers = concat_layers
        self.max_length = max_length
        self.ignore_padding = ignore_padding
        self._require_mols = False
        self.random_seed = random_seed
        self.preload = preload
        self.pooling = pooling
        self.prefer_encoder = prefer_encoder
        self._pooling_obj = None
        if isinstance(kind, HFModel):
            self.kind = kind.name
            self.featurizer = kind
        else:
            self.kind = kind
            self.featurizer = HFModel(name=self.kind)
        self.notation = self.featurizer.get_notation(notation) or "none"
        self.converter = SmilesConverter(self.notation)
        if self.preload:
            self._preload()

    def _update_params(self):
        """Update the parameters of this model"""
        # pylint: disable=no-member
        super()._update_params()

        hf_model = HFModel(
            name=self.kind,
        )
        self.featurizer = hf_model.load()
        config = self.featurizer.model.config.to_dict()
        self._pooling_obj = self._pooling_obj = (
            get_default_hgf_pooler(self.pooling, config, random_seed=self.random_seed)
            if self._pooling_obj is None
            else self._pooling_obj
        )

    def _preload(self):
        """Perform preloading of the model from the store"""
        super()._preload()
        self.featurizer.max_length = self.max_length

        # we can be confident that the model has been loaded here
        if self._pooling_obj is not None and self.preload:
            return
        config = self.featurizer.model.config.to_dict()
        cur_tokenizer = self.featurizer.tokenizer
        for special_token_id_name in [
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
            "sep_token_id",
            "mask_token_id",
        ]:
            token_id = getattr(cur_tokenizer, special_token_id_name)
            if token_id is not None:
                config[special_token_id_name] = token_id

        self._pooling_obj = (
            get_default_hgf_pooler(self.pooling, config, random_seed=self.random_seed)
            if self._pooling_obj is None
            else self._pooling_obj
        )
        # pooling layer is still none, that means we could not fetch it properly
        if self._pooling_obj is None:
            logger.warning(
                "Cannot confidently find the pooling layer and therefore will not apply pooling"
            )

    def _convert(self, inputs: list, **kwargs):
        """Convert the list of molecules to the right format for embedding

        Args:
            inputs: inputs to preprocess

        Returns:
            processed: pre-processed input list
        """
        self._preload()

        if isinstance(inputs, (str, dm.Mol)):
            inputs = [inputs]

        def _to_smiles(x):
            return dm.to_smiles(x) if not isinstance(x, str) else x

        parallel_kwargs = getattr(self, "parallel_kwargs", {})

        if len(inputs) > 1:
            smiles = dm.utils.parallelized(
                _to_smiles,
                inputs,
                n_jobs=self.n_jobs,
                **parallel_kwargs,
            )
            inputs = dm.utils.parallelized(
                self.converter.encode,
                smiles,
                n_jobs=self.n_jobs,
                **parallel_kwargs,
            )
        else:
            inputs = self.converter.encode(_to_smiles(inputs[0]))
        # this check is necessary for some tokenizers
        if isinstance(inputs, str):
            inputs = [inputs]
        encoded = self.featurizer.tokenizer(
            list(inputs),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded

    def _embed(self, inputs, **kwargs):
        """
        Perform embedding of inputs using the pretrained model

        Args:
            inputs: smiles or seqs
            kwargs: any additional parameters
        """
        self._preload()

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None and self.ignore_padding:
            attention_mask = attention_mask.unsqueeze(-1)  # B, S, 1
        else:
            attention_mask = None
        with torch.no_grad():
            if (
                isinstance(self.featurizer.model, EncoderDecoderModel)
                or hasattr(self.featurizer.model, "encoder")
            ) and self.prefer_encoder:
                out_dict = self.featurizer.model.encoder(output_hidden_states=True, **inputs)
            else:
                out_dict = self.featurizer.model(output_hidden_states=True, **inputs)
            hidden_state = out_dict["hidden_states"]
            emb_layers = []
            for layer in self.concat_layers:
                emb = hidden_state[layer].detach().cpu()  # B, S, D
                emb = self._pooling_obj(
                    emb,
                    inputs["input_ids"],
                    mask=attention_mask,
                    ignore_padding=self.ignore_padding,
                )
                emb_layers.append(emb)
            emb = torch.cat(emb_layers, dim=1)
        return emb.numpy()

    def set_max_length(self, max_length: int):
        """Set the maximum length for this featurizer"""
        self.max_length = max_length
        self._preload()
