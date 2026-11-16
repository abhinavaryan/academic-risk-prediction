import shutil
import threading

import pytest

from riskml.serving.registry import ArtifactIntegrityError, ModelRegistry, UnknownCutoffError


def test_loads_model_schema_and_threshold(artifacts_dir):
    loaded = ModelRegistry(artifacts_dir).get(60)
    assert loaded.cutoff == 60
    assert loaded.threshold == 0.45
    assert [f["name"] for f in loaded.schema["features"]] == ["clicks", "score", "gender"]


def test_available_cutoffs(artifacts_dir):
    assert ModelRegistry(artifacts_dir).cutoffs == [0, 60]


def test_unknown_cutoff_raises(artifacts_dir):
    with pytest.raises(UnknownCutoffError):
        ModelRegistry(artifacts_dir).get(45)


def test_tampered_artifact_refused(artifacts_dir, tmp_path):
    copy = tmp_path / "art"
    shutil.copytree(artifacts_dir, copy)
    model = copy / "models" / "hgb_day60.skops"
    model.write_bytes(model.read_bytes() + b"\x00")
    with pytest.raises(ArtifactIntegrityError):
        ModelRegistry(copy).get(60)


def test_file_missing_from_manifest_refused(artifacts_dir, tmp_path):
    copy = tmp_path / "art2"
    shutil.copytree(artifacts_dir, copy)
    (copy / "models" / "hgb_day0.skops").rename(copy / "models" / "evil.skops")
    with pytest.raises(ArtifactIntegrityError):
        ModelRegistry(copy).get(0)


def test_concurrent_get_loads_once(artifacts_dir, monkeypatch):
    reg = ModelRegistry(artifacts_dir)
    calls = []
    original = reg._load

    def counting(cutoff):
        calls.append(cutoff)
        return original(cutoff)

    monkeypatch.setattr(reg, "_load", counting)
    barrier = threading.Barrier(32)
    results = []

    def worker():
        barrier.wait()
        results.append(reg.get(60))

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [60]
    assert len({id(r) for r in results}) == 1
