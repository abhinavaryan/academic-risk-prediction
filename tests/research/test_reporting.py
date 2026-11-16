"""Figures and the Markdown report regenerate from the committed dashboard JSON."""

import pytest

from riskml.config import GENERATED_DIR
from riskml.research.report import write_report


def test_report_has_every_model(tmp_path):
    import json

    out = write_report(GENERATED_DIR, tmp_path / "results.md")
    text = out.read_text()
    for m in json.loads((GENERATED_DIR / "leaderboard.json").read_text())["models"]:
        assert m["name"] in text
    assert "nan" not in text.lower()


def test_figures_render(tmp_path):
    pytest.importorskip("matplotlib")
    from riskml.research.figures import make_figures

    files = make_figures(GENERATED_DIR, tmp_path)
    assert len(files) >= 10
    assert all(f.stat().st_size > 10_000 for f in files)


def test_readme_results_block_is_generated(tmp_path):
    from riskml.research.report import update_readme

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    update_readme(GENERATED_DIR, readme)
    text = readme.read_text()
    assert "old" not in text and text.startswith("intro") and text.endswith("outro\n")
    assert "ROC-AUC" in text and "<!-- RESULTS:END -->" in text
