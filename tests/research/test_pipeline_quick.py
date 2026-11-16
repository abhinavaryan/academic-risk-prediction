"""Quick end-to-end run of the research pipeline on the real OULAD (skipped if not downloaded)."""

import json

import pytest

from riskml.config import DEFAULT_DATA_DIR

pytestmark = [pytest.mark.oulad, pytest.mark.slow]


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "studentVle.csv").exists(),
                    reason="OULAD not downloaded (run `make data`)")
def test_quick_pipeline_writes_consistent_outputs(tmp_path):
    from riskml.research.pipeline import run

    run(DEFAULT_DATA_DIR, out_root=tmp_path, quick=True)
    gen = tmp_path / "web" / "src" / "generated"
    lb = json.loads((gen / "leaderboard.json").read_text())
    assert lb["champion"] in {m["name"] for m in lb["models"]}
    for m in lb["models"]:
        assert 0.5 <= m["test"]["roc_auc"] <= 1.0
    manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text())
    assert manifest["files"], "manifest lists served model files"
    overview = json.loads((gen / "overview.json").read_text())
    assert overview["dataset"]["registrations"] == 32593
