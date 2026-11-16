import numpy as np
import pandas as pd
import pytest

from riskml.custom.trainer import TrainConfig, TrainingError, train_custom

UCI_CFG = TrainConfig(target="Target", positive_values=["Dropout"],
                      negative_values=["Graduate"], unknown_values=["Enrolled"])


@pytest.fixture(scope="module")
def uci_result(uci_df):
    return train_custom(uci_df, UCI_CFG, time_budget_s=120)


@pytest.mark.slow
def test_trains_on_uci_and_scores_enrolled(uci_result):
    r = uci_result
    assert r["summary"]["n_labeled"] == 3630
    assert r["summary"]["n_scored"] == 794
    scored = [p for p in r["predictions"] if p["source"] == "scored"]
    assert len(scored) == 794
    assert all(p["actual"] is None for p in scored)
    champ = next(m for m in r["leaderboard"] if m["name"] == r["champion"])
    assert champ["roc_auc"] > 0.85
    assert len(r["leaderboard"]) >= 5
    assert r["leaderboard"] == sorted(r["leaderboard"], key=lambda m: m["roc_auc"], reverse=True)


@pytest.mark.slow
def test_reasons_present_for_scored_rows(uci_result):
    scored = [p for p in uci_result["predictions"] if p["source"] == "scored"]
    high = [p for p in scored if p["band"] == "high"]
    assert high, "some enrolled students should be high risk"
    assert all(1 <= len(p["reasons"]) <= 3 for p in high)
    assert all(r["impact"] > 0 for p in high for r in p["reasons"])
    assert uci_result["importance"][0]["mean_abs_impact"] > 0


@pytest.mark.slow
def test_row_numbers_match_spreadsheet(uci_result, uci_df):
    first_scored = next(p for p in uci_result["predictions"] if p["source"] == "scored")
    assert uci_df.loc[first_scored["row"] - 2, "Target"] == "Enrolled"


def test_leaky_column_excluded_with_warning(uci_df):
    df = uci_df.sample(800, random_state=0).reset_index(drop=True)
    df["Final status code"] = np.where(df["Target"] == "Dropout", "D", "G")
    df.loc[df["Target"] == "Enrolled", "Final status code"] = "E"
    r = train_custom(df, UCI_CFG, time_budget_s=60)
    excluded = {e["column"]: e["reason"] for e in r["summary"]["excluded"]}
    assert "Final status code" in excluded
    assert "leak" in excluded["Final status code"]
    assert any("Final status code" in w for w in r["warnings"])


def test_single_class_rejected(uci_df):
    df = uci_df[uci_df["Target"] != "Dropout"]
    with pytest.raises(TrainingError) as e:
        train_custom(df, UCI_CFG)
    assert e.value.code == "single_class"


def test_too_few_labeled_rows(uci_df):
    with pytest.raises(TrainingError) as e:
        train_custom(uci_df.head(20), UCI_CFG)
    assert e.value.code == "too_few_rows"


def test_unknown_target_column(uci_df):
    with pytest.raises(TrainingError) as e:
        train_custom(uci_df, TrainConfig(target="Nope", positive_values=["x"]))
    assert e.value.code == "bad_config"


def test_deterministic(uci_df):
    df = uci_df.sample(600, random_state=1).reset_index(drop=True)
    cfg = TrainConfig(target="Target", positive_values=["Dropout"], unknown_values=["Enrolled"],
                      models=["Logistic Regression", "Random Forest"])
    a = train_custom(df, cfg)
    b = train_custom(df, cfg)
    assert a["leaderboard"] == b["leaderboard"]
    assert [p["probability"] for p in a["predictions"]] == [p["probability"]
                                                            for p in b["predictions"]]


def test_time_budget_keeps_first_model_and_warns(uci_df):
    df = uci_df.sample(500, random_state=2).reset_index(drop=True)
    r = train_custom(df, UCI_CFG, time_budget_s=0.0)
    assert len(r["leaderboard"]) == 1
    assert any("time budget" in w for w in r["warnings"])


def test_numeric_threshold_target_and_percent_strings():
    rng = np.random.default_rng(0)
    n = 300
    att = rng.uniform(30, 100, n)
    marks = np.clip(att * 0.7 + rng.normal(0, 8, n), 0, 100)
    df = pd.DataFrame({"Roll No": [f"R{i}" for i in range(n)],
                       "Attendance": [f"{a:.0f}%" for a in att],
                       "Final Marks": [f"{m:.1f}" for m in marks]})
    df.loc[:19, "Final Marks"] = None  # current students: result not out yet
    cfg = TrainConfig(target="Final Marks", positive_below=40.0, id_column="Roll No")
    r = train_custom(df, cfg)
    assert r["summary"]["n_scored"] == 20
    assert r["summary"]["features_used"] == ["Attendance"]
    scored = [p for p in r["predictions"] if p["source"] == "scored"]
    assert scored[0]["label"].startswith("R")


def test_explanations_shrink_when_over_budget(uci_df):
    r = train_custom(uci_df, UCI_CFG, time_budget_s=0.0)
    assert r["summary"]["n_explained"] <= 300
    assert r["summary"]["n_explained"] >= 1
