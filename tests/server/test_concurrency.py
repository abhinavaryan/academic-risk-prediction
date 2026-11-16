import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from riskml.config import SAMPLE_DATA_DIR

UCI = (SAMPLE_DATA_DIR / "uci_student_dropout.csv").read_bytes()


def test_parallel_predict_consistent(client):
    payload = {"cutoff": 60, "features": {"clicks": 12, "score": 41.5, "gender": "F"}}
    app = client.app

    def call(_):
        with TestClient(app) as c:
            return c.post("/api/predict", json=payload).json()

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(call, range(50)))
    assert all(r == results[0] for r in results)


@pytest.mark.slow
def test_parallel_custom_training_isolated(make_client):
    client = make_client(max_trainings=2)
    app = client.app
    configs = [
        {
            "target": "Target",
            "positive_values": ["Dropout"],
            "unknown_values": ["Enrolled"],
            "models": ["Logistic Regression", "Decision Tree"],
        },
        {
            "target": "Target",
            "positive_values": ["Dropout", "Enrolled"],
            "models": ["Logistic Regression", "Naive Bayes"],
        },
    ]

    def call(cfg):
        with TestClient(app) as c:
            r = c.post(
                "/api/custom/train",
                files={"file": ("uci.csv", UCI, "text/csv")},
                data={"config": json.dumps(cfg)},
            )
            return r.status_code, r.json()

    sequential = [call(c) for c in configs]
    with ThreadPoolExecutor(max_workers=2) as ex:
        parallel = list(ex.map(call, configs))
    for (s_code, s_body), (p_code, p_body) in zip(sequential, parallel, strict=True):
        assert s_code == p_code == 200
        assert s_body["leaderboard"] == p_body["leaderboard"] or all(
            {k: v for k, v in a.items() if k != "seconds"}
            == {k: v for k, v in b.items() if k != "seconds"}
            for a, b in zip(s_body["leaderboard"], p_body["leaderboard"], strict=True)
        )
        assert s_body["summary"] == p_body["summary"]
    assert parallel[0][1]["summary"]["n_scored"] == 794
    assert parallel[1][1]["summary"]["n_scored"] == 0
