from types import SimpleNamespace

import hashlib
import sys

import datamol as dm
import numpy as np
import pytest
import torch

from molfeat.trans.pretrained.foundation import (
    CheMeleonTransformer,
    MolJEPATransformer,
    _CheMeleonDMPNN,
    _batch_chemeleon_graphs,
)


def test_chemeleon_graph_and_inference_shapes():
    molecules = [dm.to_mol("CCO"), dm.to_mol("[Na+]")]
    atom_features, bond_features, edge_index, reverse_edge_index, batch = (
        _batch_chemeleon_graphs(molecules)
    )

    assert atom_features.shape == (4, 72)
    assert bond_features.shape == (4, 14)
    assert edge_index.shape == (2, 4)
    assert reverse_edge_index.tolist() == [1, 0, 3, 2]
    assert batch.tolist() == [0, 0, 0, 1]

    model = _CheMeleonDMPNN(hidden_dim=8, depth=3)
    embeddings = model(atom_features, bond_features, edge_index, reverse_edge_index, batch)
    assert embeddings.shape == (2, 8)
    assert torch.isfinite(embeddings).all()


def test_chemeleon_rejects_empty_molecules():
    with pytest.raises(ValueError, match="empty molecule"):
        _batch_chemeleon_graphs([dm.to_mol("")])


def test_chemeleon_download_is_checksum_verified(tmp_path):
    content = b"test checkpoint contents"
    source = tmp_path / "source.pt"
    source.write_bytes(content)
    destination = tmp_path / "cache" / "chemeleon.pt"
    checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()

    transformer = CheMeleonTransformer(
        checkpoint_path=destination,
        download_url=source.as_uri(),
        checksum=checksum,
    )
    assert transformer._ensure_checkpoint() == destination
    assert destination.read_bytes() == content

    destination.write_bytes(b"corrupt")
    assert transformer._ensure_checkpoint() == destination
    assert destination.read_bytes() == content


def test_moljepa_requires_explicit_security_and_license_acknowledgements():
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        MolJEPATransformer()._preload()

    with pytest.raises(ValueError, match="accept_noncommercial_license=True"):
        MolJEPATransformer(trust_remote_code=True)._preload()


def test_moljepa_pins_remote_code_and_flattens_output(monkeypatch):
    calls = {}

    class FakeModel(torch.nn.Module):
        def forward(self, smiles):
            calls["smiles"] = smiles
            values = torch.arange(len(smiles) * 13 * 512, dtype=torch.float32)
            return SimpleNamespace(embeddings=values.reshape(len(smiles), 13, 512))

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        "molfeat.trans.pretrained.foundation.requires.check",
        lambda dependency: dependency in {"transformers", "torch_geometric"},
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    transformer = MolJEPATransformer(
        output="embeddings",
        trust_remote_code=True,
        accept_noncommercial_license=True,
    )
    embeddings = transformer.transform(["CCO", "c1ccccc1"])

    assert embeddings.shape == (2, 13 * 512)
    assert embeddings.dtype == np.float32
    assert calls["smiles"] == ["CCO", "c1ccccc1"]
    assert calls["kwargs"]["revision"] == transformer.revision
    assert calls["kwargs"]["code_revision"] == transformer.revision
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["low_cpu_mem_usage"] is False
