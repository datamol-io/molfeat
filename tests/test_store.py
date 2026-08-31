import datamol as dm

from molfeat.store import ModelStore


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

    assert store.available_models == []
    assert len(calls) == 1
