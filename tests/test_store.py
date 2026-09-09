import hashlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import datamol as dm
import fsspec
import pytest

from molfeat.store import ModelInfo, ModelStore, ModelStoreError
from molfeat.store.modelcard import get_model_init
from molfeat.store.loader import PretrainedStoreModel
from molfeat.store.modelstore import _download_directory
from molfeat.utils.commons import sha256sum


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "platformdirs.user_cache_dir", lambda *args, **kwargs: str(tmp_path / "cache")
    )


@pytest.fixture
def http_store(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        sha256sum=sha256sum,
    )


def test_pretrained_registration_requires_declared_license():
    card = _model_card()
    card.license = None
    store = ModelStore("memory://models")

    with pytest.raises(ModelStoreError, match="must declare the checkpoint license"):
        store.register(card, model={"weights": []})


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


def test_http_model_directory_download_and_force(http_store, tmp_path):
    root, url = http_store
    card = _model_card()
    remote = Path(card.path(str(root)))
    model = remote / ModelStore.MODEL_PATH_NAME
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "nested" / "weights.bin").write_bytes(b"weights")
    (remote / ModelStore.METADATA_PATH_NAME).write_text(card.model_dump_json())
    store = ModelStore(url)
    store._available_models = [card]
    output = tmp_path / "download"

    store.download(card, output)
    cached = output / ModelStore.MODEL_PATH_NAME
    assert (cached / "config.json").read_text() == "{}"
    assert (cached / "nested" / "weights.bin").read_bytes() == b"weights"
    assert sorted(p.relative_to(cached).as_posix() for p in cached.rglob("*") if p.is_file()) == [
        "config.json",
        "nested/weights.bin",
    ]

    (model / "nested" / "weights.bin").write_bytes(b"updated")
    store.download(card, output)
    assert (cached / "nested" / "weights.bin").read_bytes() == b"weights"
    store.download(card, output, force=True)
    assert (cached / "nested" / "weights.bin").read_bytes() == b"updated"


def test_http_model_directory_checksum_still_enforced(http_store, tmp_path):
    root, url = http_store
    card = _model_card(sha256sum="wrong-checksum")
    remote = Path(card.path(str(root)))
    (remote / ModelStore.MODEL_PATH_NAME).mkdir(parents=True)
    (remote / ModelStore.MODEL_PATH_NAME / "weights.bin").write_bytes(b"weights")
    (remote / ModelStore.METADATA_PATH_NAME).write_text(card.model_dump_json())
    store = ModelStore(url)
    store._available_models = [card]
    with pytest.raises(ModelStoreError, match="different sha256sum"):
        store.download(card, tmp_path / "download")


@pytest.mark.parametrize("path", ["/model/../escape", "/model//escape"])
def test_model_directory_rejects_path_traversal(tmp_path, monkeypatch, path):
    fs = fsspec.filesystem("memory")
    monkeypatch.setattr(fs, "find", lambda *args, **kwargs: {path: {"type": "file"}})
    with pytest.raises(ValueError, match="escapes its directory"):
        _download_directory("memory://model", tmp_path / "download")


def test_directory_checksum_preserves_sorted_content_hash(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "empty").mkdir()
    first = b"a" * (1024 * 1024 + 1)
    (tmp_path / "a.bin").write_bytes(first)
    (tmp_path / "nested" / "b.bin").write_bytes(b"b")
    assert sha256sum(tmp_path) == hashlib.sha256(first + b"b").hexdigest()
    assert sha256sum(tmp_path / "a.bin") == hashlib.sha256(first).hexdigest()
    assert sha256sum(tmp_path / "empty") == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("loader", ["experiment", "ensure_local"])
def test_hf_http_safetensors_round_trip(http_store, loader):
    transformers = pytest.importorskip("transformers")
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from molfeat.trans.pretrained.hf_transformers import HFExperiment, HFModel

    root, url = http_store
    config = transformers.BertConfig(
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    model = transformers.BertModel(config).eval()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[PAD]": 1, "CCO": 2, "C": 3}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    model.save_pretrained(root / "model")
    tokenizer.save_pretrained(root / "model")
    source = url + "/model"
    if loader == "ensure_local":
        source = HFModel._ensure_local(source)
        assert Path(source, "model.safetensors").is_file()
    restored = HFExperiment.load(source)
    inputs = tokenizer(["CCO"], return_tensors="pt")
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state
        actual = restored.model(
            **restored.tokenizer(["CCO"], return_tensors="pt")
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
