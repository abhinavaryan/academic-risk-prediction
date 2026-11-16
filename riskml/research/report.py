"""Markdown results report (tables ready to paste into the project report / synopsis).

python -m riskml.research.report        # regenerate reports/results.md from web/src/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskml.config import GENERATED_DIR, REPORTS_DIR


def _f(v, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _auc_ci(m: dict) -> str:
    ci = (m.get("ci95") or {}).get("roc_auc")
    base = _f(m["test"]["roc_auc"])
    return f"{base} [{_f(ci[0])}–{_f(ci[1])}]" if ci else base


def _p(test: dict | None) -> str:
    if not test:
        return "—"
    return "<0.001" if test["p_value"] < 0.001 else f"{test['p_value']:.3f}"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(gen: Path = GENERATED_DIR, out: Path = REPORTS_DIR / "results.md") -> Path:
    load = lambda n: json.loads((gen / f"{n}.json").read_text())  # noqa: E731
    ov, lb, tl, ab, imp, fr = (
        load(n)
        for n in ("overview", "leaderboard", "timeline", "ablation", "importance", "fairness")
    )
    d = ov["dataset"]
    parts = [
        "# Results — early academic-risk prediction on OULAD",
        f"_Generated {ov['generated_at']} by `python -m riskml.research.pipeline` "
        f"(pipeline v{ov['version']}, runtime {ov['runtime_minutes']} min). "
        "Every number below is computed from the public dataset._",
        "## Dataset",
        _table(
            ["Item", "Value"],
            [
                ["Course registrations", f"{d['registrations']:,}"],
                ["Unique students", f"{d['students']:,}"],
                ["Modules / course runs", f"{d['modules']} / {d['module_presentations']}"],
                ["VLE activity records", f"{d['vle_interactions']:,} ({d['vle_clicks']:,} clicks)"],
                ["Assessment submissions", f"{d['assessment_submissions']:,}"],
            ],
        ),
        "## Protocol",
        f"- Train: {', '.join(ov['split']['train_presentations'])} "
        f"({ov['split']['n_train']:,} registrations at day {ov['primary_cutoff']}); "
        f"test: {ov['split']['test_presentation']} ({ov['split']['n_test']:,}).",
        f"- Model selection: {ov['split']['cv_folds']}-fold {ov['split']['cv']}, metric = CV "
        "ROC-AUC. Decision threshold maximises F1 on out-of-fold predictions.",
        "- 95% confidence intervals: percentile bootstrap (1,000 resamples) of the test set.",
        f"## Model leaderboard (day {lb['cutoff']})",
        _table(
            [
                "#",
                "Model",
                "CV ROC-AUC",
                "Test ROC-AUC [95% CI]",
                "F1",
                "Precision",
                "Recall",
                "Accuracy",
                "MCC",
                "McNemar p vs best",
            ],
            [
                [
                    m["rank"],
                    m["name"] + (" **(deployed)**" if m["name"] == lb["deployed"] else ""),
                    f"{_f(m['cv']['roc_auc'])} ± {_f(m['cv']['roc_auc_std'])}",
                    _auc_ci(m),
                    _f(m["test"]["f1"]),
                    _pct(m["test"]["precision"]),
                    _pct(m["test"]["recall"]),
                    _pct(m["test"]["accuracy"]),
                    _f(m["test"]["mcc"]),
                    _p(m.get("mcnemar_vs_champion")),
                ]
                for m in sorted(lb["models"], key=lambda x: x["rank"])
            ],
        ),
        f"Best by cross-validation: **{lb['champion']}**. Deployed: **{lb['deployed']}** — "
        f"{ov['deployed']['reason']}",
        "## Early-warning timeline (test ROC-AUC / F1 by prediction day)",
        _table(
            [
                "Day",
                "Students still enrolled",
                "At-risk share",
                *[f"{m} AUC / F1" for m in tl["models"]],
            ],
            [
                [
                    c["cutoff"],
                    f"{c['n_test']:,}",
                    _pct(c["at_risk_rate_test"]),
                    *[
                        f"{_f(c['models'][m]['roc_auc'])} / {_f(c['models'][m]['f1'])}"
                        for m in tl["models"]
                    ],
                ]
                for c in tl["cutoffs"]
            ],
        ),
        f"## Feature-group ablation ({ab['model']})",
        _table(
            ["Feature set", "Features", "ROC-AUC", "F1", "Recall"],
            [
                [v["variant"], v["n_features"], _f(v["roc_auc"]), _f(v["f1"]), _pct(v["recall"])]
                for v in ab["variants"]
            ],
        ),
        "## Most important features (permutation importance, deployed model)",
        _table(
            ["Feature", "Drop in test ROC-AUC"],
            [[r["label"], f"{r['mean']:.4f} ± {r['std']:.4f}"] for r in imp["permutation"][:15]],
        ),
        "## Fairness audit (deployed model, test semester)",
        _table(
            [
                "Attribute",
                "Recall gap",
                "False-alarm gap",
                "Flag-rate gap",
                "Recall gap without protected attributes",
            ],
            [
                [
                    a["label"],
                    _f(a["gaps"]["tpr_gap"]),
                    _f(a["gaps"]["fpr_gap"]),
                    _f(a["gaps"]["positive_rate_gap"]),
                    _f((a.get("gaps_without_protected") or {}).get("tpr_gap")),
                ]
                for a in fr["attributes"]
            ],
        ),
        "## Figures",
        "See `reports/figures/` (PNG, 200 dpi).",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n")
    return out


README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(gen: Path = GENERATED_DIR, readme: Path | None = None) -> Path:
    """Rewrite the headline-results block of README.md from the generated JSON."""
    from riskml.config import ROOT  # noqa: PLC0415

    readme = readme or ROOT / "README.md"
    ov = json.loads((gen / "overview.json").read_text())
    lb = json.loads((gen / "leaderboard.json").read_text())
    tl = json.loads((gen / "timeline.json").read_text())
    dep, champ, ew = ov["deployed"], ov["champion"], ov["early_warning"]
    by_name = {m["name"]: m for m in lb["models"]}
    top = sorted(lb["models"], key=lambda m: m["rank"])[:5]
    dep_base = tl["deployed"]
    block = "\n\n".join(
        [
            README_START,
            f"### Headline results (day {ov['primary_cutoff']}, unseen "
            f"{ov['split']['test_presentation']} semester)",
            f"- **{len(lb['models'])} models compared**; best by cross-validation: "
            f"**{champ['name']}** (test ROC-AUC {_f(champ['test']['roc_auc'])}).",
            f"- **Deployed: {dep['name']}**: ROC-AUC {_auc_ci(by_name[dep['name']])}, F1 "
            f"{_f(dep['test']['f1'])}, recall {_pct(ew['recall'])} — it flags "
            f"{ew['tp']:,} of the {ew['at_risk_remaining_test']:,} students who went on to fail "
            "or withdraw.",
            "- Early warning (deployed model, test ROC-AUC by prediction day): "
            + ", ".join(
                f"day {c['cutoff']}: {_f(c['models'][dep_base]['roc_auc'])}" for c in tl["cutoffs"]
            )
            + ".",
            _table(
                ["Model", "CV ROC-AUC", "Test ROC-AUC [95% CI]", "Test F1"],
                [[m["name"], _f(m["cv"]["roc_auc"]), _auc_ci(m), _f(m["test"]["f1"])] for m in top],
            ),
            f"_Auto-generated by the pipeline on {ov['generated_at'][:10]}; full tables in "
            "[reports/results.md](reports/results.md)._",
            README_END,
        ]
    )
    text = readme.read_text()
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + block + text[end:])
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "results.md")
    args = parser.parse_args()
    print(write_report(args.gen_dir, args.out))
    if args.out == REPORTS_DIR / "results.md":
        print(update_readme(args.gen_dir))


if __name__ == "__main__":
    main()
