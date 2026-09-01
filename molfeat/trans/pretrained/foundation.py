"""Adapters for recent molecular foundation models.

The adapters in this module deliberately keep model-specific dependencies out
of Molfeat's core runtime. CheMeleon is implemented with the RDKit and PyTorch
primitives Molfeat already depends on. Mol-JEPA reuses the existing
``transformer`` and ``pyg`` extras and never executes Hub code unless the caller
explicitly opts in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Optional

import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid

import datamol as dm
import filelock
import numpy as np
import platformdirs
import torch

from rdkit.Chem.rdchem import BondType, HybridizationType
from torch import nn

from molfeat.trans.pretrained.base import PretrainedMolTransformer
from molfeat.utils import requires

_CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
_CHEMELEON_MD5 = "6a80b54fdb7de37ef0374d302f01e8ce"
_MOLJEPA_REPOSITORY = "Flogrammer/Mol-JEPA"
_MOLJEPA_REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"


_ATOM_FEATURE_CHOICES = (
    {value: index for index, value in enumerate(list(range(1, 37)) + [53])},
    {value: value for value in range(6)},
    {value: index for index, value in enumerate([-1, -2, 1, 2, 0])},
    {value: value for value in range(4)},
    {value: value for value in range(5)},
    {
        value: index
        for index, value in enumerate(
            [
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ]
        )
    },
)
_BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def _chemeleon_atom_features(atom) -> np.ndarray:
    """Return the 72 features used by the released CheMeleon checkpoint."""
    features = np.zeros(72, dtype=np.float32)
    values = (
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        int(atom.GetTotalNumHs()),
        atom.GetHybridization(),
    )

    offset = 0
    for value, choices in zip(values, _ATOM_FEATURE_CHOICES):
        features[offset + choices.get(value, len(choices))] = 1
        offset += len(choices) + 1

    features[offset] = int(atom.GetIsAromatic())
    features[offset + 1] = 0.01 * atom.GetMass()
    return features


def _chemeleon_bond_features(bond) -> np.ndarray:
    """Return the 14 features used by the released CheMeleon checkpoint."""
    features = np.zeros(14, dtype=np.float32)

    try:
        bond_type_index = _BOND_TYPES.index(bond.GetBondType())
    except ValueError:
        bond_type_index = len(_BOND_TYPES)
    if bond_type_index < len(_BOND_TYPES):
        features[1 + bond_type_index] = 1

    features[5] = int(bond.GetIsConjugated())
    features[6] = int(bond.IsInRing())
    stereo = int(bond.GetStereo())
    features[7 + (stereo if stereo in range(6) else 6)] = 1
    return features


def _batch_chemeleon_graphs(molecules: Sequence[dm.Mol]):
    """Build the directed molecular graph batch consumed by CheMeleon."""
    atom_features = []
    bond_features = []
    edge_indices = []
    reverse_edge_indices = []
    batch_indices = []
    atom_offset = 0
    bond_offset = 0

    for batch_index, molecule in enumerate(molecules):
        if molecule is None:
            raise ValueError("CheMeleon cannot featurize an invalid molecule.")
        if molecule.GetNumAtoms() == 0:
            raise ValueError("CheMeleon cannot featurize an empty molecule.")
        current_atom_features = np.asarray(
            [_chemeleon_atom_features(atom) for atom in molecule.GetAtoms()],
            dtype=np.float32,
        )

        current_bond_features = []
        sources = []
        targets = []
        for bond in molecule.GetBonds():
            features = _chemeleon_bond_features(bond)
            current_bond_features.extend((features, features))
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            sources.extend((begin, end))
            targets.extend((end, begin))

        current_bond_features = np.asarray(current_bond_features, dtype=np.float32).reshape(-1, 14)
        current_edge_indices = np.asarray((sources, targets), dtype=np.int64)
        current_reverse_indices = (
            np.arange(len(current_bond_features)).reshape(-1, 2)[:, ::-1].ravel()
        )

        atom_features.append(current_atom_features)
        bond_features.append(current_bond_features)
        edge_indices.append(current_edge_indices + atom_offset)
        reverse_edge_indices.append(current_reverse_indices + bond_offset)
        batch_indices.extend([batch_index] * len(current_atom_features))
        atom_offset += len(current_atom_features)
        bond_offset += len(current_bond_features)

    return (
        torch.from_numpy(np.concatenate(atom_features)).float(),
        torch.from_numpy(np.concatenate(bond_features)).float(),
        torch.from_numpy(np.hstack(edge_indices)).long(),
        torch.from_numpy(np.concatenate(reverse_edge_indices)).long(),
        torch.tensor(batch_indices, dtype=torch.long),
    )


class _CheMeleonDMPNN(nn.Module):
    """Minimal inference-only D-MPNN used by CheMeleon."""

    def __init__(self, hidden_dim: int = 2048, depth: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.W_i = nn.Linear(72 + 14, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(72 + hidden_dim, hidden_dim)

    @staticmethod
    def _sum_messages(values: torch.Tensor, indices: torch.Tensor, n_atoms: int):
        expanded_indices = indices.unsqueeze(1).repeat(1, values.shape[1])
        return values.new_zeros((n_atoms, values.shape[1])).scatter_reduce_(
            0,
            expanded_indices,
            values,
            reduce="sum",
            include_self=False,
        )

    def forward(self, atom_features, bond_features, edge_index, reverse_edge_index, batch):
        initial = self.W_i(torch.cat((atom_features[edge_index[0]], bond_features), dim=1))
        hidden = torch.relu(initial)

        for _ in range(1, self.depth):
            incoming = self._sum_messages(hidden, edge_index[1], len(atom_features))
            messages = incoming[edge_index[0]] - hidden[reverse_edge_index]
            hidden = torch.relu(initial + self.W_h(messages))

        incoming = self._sum_messages(hidden, edge_index[1], len(atom_features))
        atom_hidden = torch.relu(self.W_o(torch.cat((atom_features, incoming), dim=1)))

        expanded_batch = batch.unsqueeze(1).repeat(1, atom_hidden.shape[1])
        n_molecules = int(batch.max().item()) + 1
        return atom_hidden.new_zeros((n_molecules, atom_hidden.shape[1])).scatter_reduce_(
            0,
            expanded_batch,
            atom_hidden,
            reduce="mean",
            include_self=False,
        )


def _md5sum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheMeleonTransformer(PretrainedMolTransformer):
    """CheMeleon descriptor-foundation-model fingerprints.

    This inference-only implementation reproduces the official Chemprop 2.2+
    fingerprint path using only Molfeat's existing RDKit and PyTorch runtime.
    The 2,048-dimensional model weights are downloaded from the authors'
    Zenodo record and checksum-verified before loading.

    References:
        https://arxiv.org/abs/2506.15792
        https://doi.org/10.5281/zenodo.15460715

    Args:
        checkpoint_path: Optional local checkpoint or cache destination.
        download_url: Checkpoint URL used when ``checkpoint_path`` is absent.
        checksum: Expected MD5 checksum. Set to ``None`` only for a trusted custom checkpoint.
        device: Torch device used for inference.
        preload: Whether to load the checkpoint during construction.
        dtype: Output data type.
        n_jobs: Number of jobs used by Molfeat's molecule conversion.
    """

    def __init__(
        self,
        checkpoint_path: Optional[os.PathLike] = None,
        download_url: Optional[str] = _CHEMELEON_URL,
        checksum: Optional[str] = _CHEMELEON_MD5,
        device: str = "cpu",
        preload: bool = False,
        dtype=np.float32,
        n_jobs: int = 0,
        **params,
    ):
        super().__init__(dtype=dtype, preload=preload, n_jobs=n_jobs, **params)
        default_path = (
            Path(platformdirs.user_cache_dir("molfeat")) / "chemeleon" / "chemeleon_mp.pt"
        )
        self.checkpoint_path = str(checkpoint_path or default_path)
        self.download_url = download_url
        self.checksum = checksum
        self.device = torch.device(device)
        self.featurizer = None
        self._require_mols = True
        self._feat_length = 2048
        if self.preload:
            self._preload()

    def _checkpoint_is_valid(self, path: Path) -> bool:
        return path.is_file() and (self.checksum is None or _md5sum(path) == self.checksum)

    def _ensure_checkpoint(self) -> Path:
        checkpoint = Path(self.checkpoint_path).expanduser()
        if self._checkpoint_is_valid(checkpoint):
            return checkpoint
        if not self.download_url:
            raise FileNotFoundError(f"CheMeleon checkpoint not found or invalid: {checkpoint}")

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        lock_path = checkpoint.with_suffix(f"{checkpoint.suffix}.lock")
        with filelock.FileLock(str(lock_path)):
            if self._checkpoint_is_valid(checkpoint):
                return checkpoint

            for attempt in range(3):
                temporary = checkpoint.with_name(f".{checkpoint.name}.{uuid.uuid4().hex}.part")
                try:
                    with urllib.request.urlopen(self.download_url, timeout=120) as source:
                        with temporary.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                    if not self._checkpoint_is_valid(temporary):
                        raise ValueError(
                            "Downloaded CheMeleon checkpoint failed checksum validation."
                        )
                    temporary.replace(checkpoint)
                    break
                except (TimeoutError, urllib.error.URLError, ValueError) as error:
                    retryable_http_error = (
                        not isinstance(error, urllib.error.HTTPError)
                        or error.code == 429
                        or error.code >= 500
                    )
                    if attempt == 2 or not retryable_http_error:
                        raise
                    time.sleep(2**attempt)
                finally:
                    temporary.unlink(missing_ok=True)
        return checkpoint

    def _preload(self):
        if self.featurizer is not None:
            return

        checkpoint = torch.load(
            self._ensure_checkpoint(),
            map_location="cpu",
            weights_only=True,
        )
        hyperparameters = checkpoint.get("hyper_parameters", {})
        expected = {
            "d_v": 72,
            "d_e": 14,
            "d_h": 2048,
            "depth": 6,
            "activation": "relu",
            "undirected": False,
        }
        if any(hyperparameters.get(key) != value for key, value in expected.items()):
            raise ValueError("The checkpoint is not the supported CheMeleon D-MPNN architecture.")

        model = _CheMeleonDMPNN(
            hidden_dim=hyperparameters["d_h"],
            depth=hyperparameters["depth"],
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.featurizer = model.to(self.device).eval()

    def _convert(self, inputs: list, **kwargs):
        self._preload()
        return inputs

    def _embed(self, inputs, **kwargs):
        self._preload()
        batch = tuple(value.to(self.device) for value in _batch_chemeleon_graphs(inputs))
        with torch.no_grad():
            return self.featurizer(*batch).cpu().numpy()

    def _update_params(self):
        super()._update_params()
        self.featurizer = None
        if self.preload:
            self._preload()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("featurizer", None)
        return state


class MolJEPATransformer(PretrainedMolTransformer):
    """Mol-JEPA molecular embeddings from the authors' Hugging Face checkpoint.

    Mol-JEPA is external, non-commercially licensed software whose Hugging Face
    model contains custom Python code. Molfeat therefore pins the default
    revision and requires two explicit acknowledgements before loading it. No
    Mol-JEPA code or weights are redistributed by Molfeat.

    References:
        https://arxiv.org/abs/2608.22642
        https://github.com/Boehringer-Ingelheim/mol-jepa

    Args:
        model_name: Hugging Face model repository.
        revision: Immutable model and code revision.
        output: Output field to return: ``cls``, ``predictions``, or ``embeddings``.
            Multi-modal outputs are flattened to one vector per molecule.
        trust_remote_code: Explicit permission to execute the repository's custom model code.
        accept_noncommercial_license: Confirm that the caller accepts the external CC BY-NC 4.0 terms.
        model_kwargs: Additional arguments for ``AutoModel.from_pretrained``.
        device: Torch device used for inference.
        preload: Whether to load the checkpoint during construction.
        dtype: Output data type.
        n_jobs: Number of jobs used by Molfeat's molecule conversion.
    """

    _OUTPUT_LENGTHS = {"cls": 512, "predictions": 12 * 512, "embeddings": 13 * 512}

    def __init__(
        self,
        model_name: str = _MOLJEPA_REPOSITORY,
        revision: str = _MOLJEPA_REVISION,
        output: Literal["cls", "predictions", "embeddings"] = "cls",
        trust_remote_code: bool = False,
        accept_noncommercial_license: bool = False,
        model_kwargs: Optional[dict] = None,
        device: str = "cpu",
        preload: bool = False,
        dtype=np.float32,
        n_jobs: int = 0,
        **params,
    ):
        if output not in self._OUTPUT_LENGTHS:
            raise ValueError(f"Unsupported Mol-JEPA output: {output}")
        super().__init__(dtype=dtype, preload=preload, n_jobs=n_jobs, **params)
        self.model_name = model_name
        self.revision = revision
        self.output = output
        self.trust_remote_code = trust_remote_code
        self.accept_noncommercial_license = accept_noncommercial_license
        self.model_kwargs = dict(model_kwargs or {})
        self.device = torch.device(device)
        self.featurizer = None
        self._require_mols = False
        self._feat_length = self._OUTPUT_LENGTHS[output]
        if self.preload:
            self._preload()

    def _preload(self):
        if self.featurizer is not None:
            return
        if not self.trust_remote_code:
            raise ValueError(
                "Mol-JEPA requires custom Hugging Face code. Inspect the pinned repository and set "
                "trust_remote_code=True to permit its execution."
            )
        if not self.accept_noncommercial_license:
            raise ValueError(
                "Mol-JEPA is licensed CC BY-NC 4.0. Set accept_noncommercial_license=True only "
                "when the intended use complies with those terms."
            )
        if not requires.check("transformers") or not requires.check("torch_geometric"):
            raise ValueError(
                "Mol-JEPA requires the existing Molfeat transformer and pyg extras: "
                'python -m pip install "molfeat[transformer,pyg]"'
            )

        from transformers import AutoConfig, AutoModel

        reserved = {"trust_remote_code", "revision", "code_revision"}.intersection(
            self.model_kwargs
        )
        if reserved:
            raise ValueError(f"Mol-JEPA model_kwargs cannot override: {sorted(reserved)}")
        load_kwargs = dict(self.model_kwargs)
        load_kwargs.setdefault("low_cpu_mem_usage", False)
        try:
            model = AutoModel.from_pretrained(
                self.model_name,
                revision=self.revision,
                code_revision=self.revision,
                trust_remote_code=True,
                **load_kwargs,
            )
        except RuntimeError as error:
            if "expected device meta" in str(error):
                model = self._load_without_meta_initialization(AutoConfig, AutoModel)
            else:
                raise
        self.featurizer = model.to(self.device).eval()

    def _load_without_meta_initialization(self, auto_config, auto_model):
        """Load the pinned checkpoint after constructing the custom model on CPU.

        Transformers 5 initializes ``from_pretrained`` models on the meta device.
        Mol-JEPA's pinned custom constructor currently combines meta and CPU
        buffers, so this narrowly scoped fallback constructs the model normally
        and then performs a strict safetensors load.
        """
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_model

        snapshot = Path(
            snapshot_download(
                repo_id=self.model_name,
                revision=self.revision,
                allow_patterns=["*.json", "*.py", "model.safetensors"],
            )
        )
        checkpoint = snapshot / "model.safetensors"
        if not checkpoint.is_file():
            raise RuntimeError(
                f"The pinned Mol-JEPA snapshot does not contain {checkpoint.name!r}."
            )

        config = auto_config.from_pretrained(
            self.model_name,
            revision=self.revision,
            code_revision=self.revision,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = auto_model.from_config(config, trust_remote_code=True)
        missing, unexpected = load_model(model, checkpoint, strict=True, device="cpu")
        if missing or unexpected:
            raise RuntimeError(
                "The pinned Mol-JEPA checkpoint does not match its model definition: "
                f"missing={missing}, unexpected={unexpected}."
            )
        return model

    def _embed(self, inputs, **kwargs):
        self._preload()
        with torch.no_grad():
            output = self.featurizer(list(inputs))
        representation = (
            output.get(self.output) if isinstance(output, Mapping) else getattr(output, self.output)
        )
        if not torch.is_tensor(representation) or representation.ndim < 2:
            raise ValueError(f"Mol-JEPA returned an invalid {self.output!r} representation.")
        return representation.detach().reshape(len(inputs), -1).cpu().numpy()

    def _update_params(self):
        super()._update_params()
        self.featurizer = None
        if self.preload:
            self._preload()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("featurizer", None)
        return state
