import json

import pytest

from riskml.config import SAMPLE_DATA_DIR

UCI = (SAMPLE_DATA_DIR / "uci_student_dropout.csv").read_bytes()
CFG = {
    "target": "Target",
    "positive_values": ["Dropout"],
    "negative_values": ["Graduate"],
    "unknown_values": ["Enrolled"],
}


def test_profile_uci(client):
    r = client.post("/api/custom/profile", files={"file": ("uci.csv", UCI, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["suggested_target"] == "Target"
    assert body["n_rows"] == 4424


@pytest.mark.slow
def test_profile_and_train_uci_end_to_end(client):
    r = client.post(
        "/api/custom/train",
        files={"file": ("uci.csv", UCI, "text/csv")},
        data={"config": json.dumps(CFG)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["n_scored"] == 794
    assert len(body["predictions"]) == 4424
    assert body["leaderboard"][0]["roc_auc"] > 0.85


def test_training_error_is_422_with_code(client):
    r = client.post(
        "/api/custom/train",
        files={"file": ("a.csv", b"a,y\n1,Fail\n2,Fail\n", "text/csv")},
        data={"config": json.dumps({"target": "y", "positive_values": ["Fail"]})},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] in {"too_few_rows", "single_class"}
