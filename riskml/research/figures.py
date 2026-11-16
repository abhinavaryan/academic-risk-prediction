"""Publication figures (PNG, 200 dpi) for the project report, drawn from the generated JSON.

python -m riskml.research.figures            # regenerate reports/figures from web/src/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riskml.config import GENERATED_DIR, REPORTS_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MUTED = "#c3c2b7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)


def _load(gen: Path, name: str):
    return json.loads((gen / f"{name}.json").read_text())


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", facecolor="white")
    plt.close(fig)


def leaderboard_chart(gen: Path, out: Path) -> None:
    lb = _load(gen, "leaderboard")
    models = sorted(lb["models"], key=lambda m: m["test"]["roc_auc"])
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(models) + 1))
    for i, m in enumerate(models):
        hl = m["name"] in (lb["deployed"], lb["champion"])
        ax.barh(i, m["test"]["roc_auc"], height=0.55, color=SERIES[0] if hl else MUTED)
        ci = (m.get("ci95") or {}).get("roc_auc")
        if ci:
            ax.plot(ci, [i, i], color=INK, linewidth=0.8)
        ax.text(
            m["test"]["roc_auc"] + 0.004,
            i,
            f"{m['test']['roc_auc']:.3f}",
            va="center",
            color=INK2,
            fontsize=7,
        )
    ax.set_yticks(range(len(models)), [m["name"] for m in models])
    ax.set_xlim(0.45, 1)
    ax.set_xlabel("Test ROC-AUC (95% bootstrap CI)")
    ax.set_title(f"Model comparison on the unseen test semester (day {lb['cutoff']})")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "01_leaderboard_test_auc")


def roc_chart(gen: Path, out: Path) -> None:
    lb, curves = _load(gen, "leaderboard"), _load(gen, "curves")
    ranked = sorted(lb["models"], key=lambda m: m["rank"])
    names = list(dict.fromkeys([lb["champion"], lb["deployed"], *[m["name"] for m in ranked]]))
    names = [n for n in names if n != "Baseline (majority)" and n in curves][:4]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8)
    auc = {m["name"]: m["test"]["roc_auc"] for m in lb["models"]}
    for i, n in enumerate(names):
        xs, ys = zip(*curves[n]["roc"], strict=True)
        ax.plot(xs, ys, color=SERIES[i], linewidth=1.6, label=f"{n} (AUC {auc[n]:.3f})")
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="ROC curves (test semester)",
    )
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, out, "02_roc_curves")


def confusion_chart(gen: Path, out: Path) -> None:
    lb = _load(gen, "leaderboard")
    m = next(x for x in lb["models"] if x["name"] == lb["deployed"])["test"]
    cells = [[m["tp"], m["fn"]], [m["fp"], m["tn"]]]
    total = sum(sum(r) for r in cells)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.imshow([[c / total for c in r] for r in cells], cmap="Blues", vmin=0, vmax=0.6)
    for i in range(2):
        for j in range(2):
            v = cells[i][j]
            ax.text(
                j,
                i,
                f"{v:,}\n{v / total:.1%}",
                ha="center",
                va="center",
                color="white" if v / total > 0.3 else INK,
                fontsize=9,
            )
    ax.set_xticks([0, 1], ["Predicted at risk", "Predicted not"])
    ax.set_yticks([0, 1], ["Actually at risk", "Actually not"])
    ax.grid(False)
    ax.set_title(f"Confusion matrix — {lb['deployed']}")
    _save(fig, out, "03_confusion_matrix")


def timeline_chart(gen: Path, out: Path) -> None:
    tl = _load(gen, "timeline")
    days = [c["cutoff"] for c in tl["cutoffs"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, metric, label in ((axes[0], "roc_auc", "ROC-AUC"), (axes[1], "f1", "F1-score")):
        for i, name in enumerate(tl["models"]):
            ax.plot(
                days,
                [c["models"].get(name, {}).get(metric) for c in tl["cutoffs"]],
                marker="o",
                markersize=3,
                color=SERIES[i],
                linewidth=1.6,
                label=name,
            )
        ax.set(xlabel="Prediction day", ylabel=label, title=f"{label} by prediction day")
    axes[0].legend(fontsize=7)
    _save(fig, out, "04_early_warning_timeline")


def ablation_chart(gen: Path, out: Path) -> None:
    ab = _load(gen, "ablation")
    v = ab["variants"]
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(v) + 1))
    ax.barh(range(len(v)), [x["roc_auc"] for x in v], height=0.55, color=SERIES[0])
    for i, x in enumerate(v):
        ax.text(x["roc_auc"] + 0.004, i, f"{x['roc_auc']:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(v)), [x["variant"] for x in v])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Test ROC-AUC", title=f"Feature-group ablation ({ab['model']})")
    _save(fig, out, "05_ablation")


def importance_charts(gen: Path, out: Path) -> None:
    imp = _load(gen, "importance")
    rows = imp["permutation"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(
        range(len(rows)),
        [max(0, r["mean"]) for r in rows],
        xerr=[r["std"] for r in rows],
        height=0.6,
        color=SERIES[0],
        error_kw={"linewidth": 0.7},
    )
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Drop in test ROC-AUC when shuffled", title="Permutation importance (deployed)")
    _save(fig, out, "06_permutation_importance")
    if imp.get("shap"):
        rows = imp["shap"]["mean_abs"][:15][::-1]
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.barh(range(len(rows)), [r["value"] for r in rows], height=0.6, color=SERIES[1])
        ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="Mean |SHAP value| (log-odds)", title=f"SHAP — {imp['shap_model']}")
        _save(fig, out, "07_shap_importance")


def fairness_chart(gen: Path, out: Path) -> None:
    fr = _load(gen, "fairness")
    attrs = fr["attributes"]
    fig, axes = plt.subplots(1, len(attrs), figsize=(2.6 * len(attrs), 3.2), sharey=True)
    for ax, a in zip(axes, attrs, strict=True):
        groups = a["groups"]
        ax.bar(range(len(groups)), [g["tpr"] or 0 for g in groups], color=SERIES[0], width=0.6)
        ax.set_xticks(
            range(len(groups)), [g["group"] for g in groups], rotation=60, ha="right", fontsize=7
        )
        ax.set_title(a["label"], fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall (true-positive rate)")
    fig.suptitle("Fairness audit: recall by group (deployed model)", fontsize=10, fontweight="bold")
    _save(fig, out, "08_fairness_recall")


def eda_charts(gen: Path, out: Path) -> None:
    eda = _load(gen, "eda")
    outcomes = [o["outcome"] for o in eda["outcomes"]]
    weeks = eda["engagement_by_week"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, o in enumerate(outcomes):
        ax.plot(
            [w["week"] for w in weeks],
            [w.get(o, 0) for w in weeks],
            color=SERIES[i],
            linewidth=1.6,
            label=o,
        )
    ax.set(
        xlabel="Course week",
        ylabel="Avg. clicks per student",
        title="Weekly LMS engagement by final result",
    )
    ax.legend(fontsize=7)
    _save(fig, out, "09_engagement_by_week")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(outcomes, [o["n"] for o in eda["outcomes"]], color=SERIES[: len(outcomes)], width=0.6)
    for i, o in enumerate(eda["outcomes"]):
        ax.text(i, o["n"], f"{o['n']:,}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Registrations", title="Final results in OULAD")
    _save(fig, out, "10_outcome_distribution")


def calibration_chart(gen: Path, out: Path) -> None:
    lb, curves = _load(gen, "leaderboard"), _load(gen, "curves")
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8)
    for i, n in enumerate(dict.fromkeys([lb["deployed"], lb["champion"]])):
        pts = curves[n]["calibration"]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="o",
            markersize=3,
            color=SERIES[i],
            linewidth=1.6,
            label=n,
        )
    ax.set(
        xlabel="Mean predicted risk",
        ylabel="Observed at-risk share",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Calibration (test semester)",
    )
    ax.legend(fontsize=7)
    _save(fig, out, "11_calibration")


def tuning_chart(gen: Path, out: Path) -> None:
    tuning = _load(gen, "leaderboard").get("tuning") or {}
    if not tuning:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, (n, t) in enumerate(tuning.items()):
        ax.plot(
            [h["trial"] + 1 for h in t["history"]],
            [h["best"] for h in t["history"]],
            color=SERIES[i],
            linewidth=1.6,
            label=n,
        )
    ax.set(
        xlabel="Optuna trial",
        ylabel="Best CV ROC-AUC so far",
        title="Bayesian hyper-parameter tuning",
    )
    ax.legend(fontsize=7)
    _save(fig, out, "12_tuning_history")


def make_figures(gen: Path = GENERATED_DIR, out: Path = REPORTS_DIR / "figures") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    for fn in (
        leaderboard_chart,
        roc_chart,
        confusion_chart,
        timeline_chart,
        ablation_chart,
        importance_charts,
        fairness_chart,
        eda_charts,
        calibration_chart,
        tuning_chart,
    ):
        fn(gen, out)
    return sorted(out.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "figures")
    args = parser.parse_args()
    for p in make_figures(args.gen_dir, args.out):
        print(p)


if __name__ == "__main__":
    main()
