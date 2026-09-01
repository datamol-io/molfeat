import hashlib

import datamol as dm
import fsspec
import pytest

from molfeat.store import ModelInfo, ModelStore, ModelStoreError
from molfeat.store.modelcard import get_model_init
from molfeat.store.loader import PretrainedStoreModel


class StubStoreModel(PretrainedStoreModel):
    def load(self):
        return self._artifact_load()


def test_model_store_loads_index_lazily(monkeypatch):
    calls = []

    def empty_store(path):
        calls.append(path)
        return []

    monkeypatch.setattr(dm.fs, "glob", empty_store)
    store = ModelStore("memory://models")
    assert calls == []

    assert store.available_models == []
    assert len(calls) == 1


@pytest.mark.parametrize("group", ["dgllife", "graphormer"])
def test_legacy_model_cards_fail_with_migration_guidance(group):
    card = _model_card()
    card.group = group

    with pytest.raises(ValueError, match="not supported by Molfeat 1.x"):
        get_model_init(card)


def _model_card(name="test-model", sha256sum=None):
    return ModelInfo(
        name=name,
        type="pretrained",
        group="test",
        submitter="Molfeat",
        description="Test model",
        representation="vector",
        authors=["Molfeat"],
        sha256sum=sha256sum,
    )


def test_pretrained_loader_repairs_partial_cache(tmp_path, monkeypatch):
    card = _model_card()
    store = ModelStore("memory://models")
    store._available_models = [card]
    cache_path = tmp_path / "cached-model"
    cache_path.mkdir()
    calls = []

    def download(modelcard, output_dir, **kwargs):
        calls.append((modelcard, output_dir))

    monkeypatch.setattr(store, "download", download)
    loader = StubStoreModel(card.name, cache_path=str(cache_path), store=store)

    assert loader._artifact_load() == str(cache_path)
    assert calls == [(card, str(cache_path))]


def test_pretrained_loader_reports_missing_model():
    store = ModelStore("memory://models")
    store._available_models = []
    loader = StubStoreModel("missing", store=store)

    with pytest.raises(ModelStoreError, match="not registered"):
        loader._artifact_load()


def test_download_allows_legacy_empty_directory_checksum(tmp_path, monkeypatch):
    card = _model_card(sha256sum=hashlib.sha256(b"").hexdigest())
    store = ModelStore("memory://models")
    store._available_models = [card]
    remote_root = card.path(store.model_store_root)
    remote_model = dm.fs.join(remote_root, store.MODEL_PATH_NAME)
    remote_metadata = dm.fs.join(remote_root, store.METADATA_PATH_NAME)
    dm.fs.mkdir(remote_model, exist_ok=True)
    with fsspec.open(dm.fs.join(remote_model, "weights.bin"), "wb") as stream:
        stream.write(b"weights")
    with fsspec.open(remote_metadata, "w") as stream:
        stream.write(card.model_dump_json())

    output = store.download(card, tmp_path / "download")

    assert dm.fs.is_file(dm.fs.join(output, store.MODEL_PATH_NAME, "weights.bin"))
