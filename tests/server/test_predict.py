def _post(client, features, cutoff=60, **extra):
    return client.post("/api/predict", json={"cutoff": cutoff, "features": features, **extra})


def test_valid_prediction(client):
    r = _post(client, {"clicks": 0, "score": 10, "gender": "M"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["band"] in {"low", "moderate", "high"}
    assert body["factors"] and {"name", "label", "impact", "direction"} <= set(body["factors"][0])


def test_unknown_category_rejected(client):
    r = _post(client, {"gender": "X"})
    assert r.status_code == 422
    assert "gender" in r.json()["error"]["fields"]


def test_negative_clicks_rejected(client):
    r = _post(client, {"clicks": -5})
    assert r.status_code == 422
    assert "clicks" in r.json()["error"]["fields"]


def test_extra_field_rejected(client):
    assert _post(client, {"clicks": 5}, admin=True).status_code == 422


def test_wrong_types_rejected(client):
    assert client.post("/api/predict", json={"cutoff": "sixty", "features": {}}).status_code == 422
    assert client.post("/api/predict", json={"cutoff": 60, "features": [1, 2]}).status_code == 422
    assert client.post("/api/predict", content=b"not json",
                       headers={"content-type": "application/json"}).status_code == 422


def test_too_many_features_rejected(client):
    r = _post(client, {f"f{i}": 1 for i in range(300)})
    assert r.status_code == 422


def test_unknown_cutoff_predict_404(client):
    assert _post(client, {"clicks": 1}, cutoff=45).status_code == 404
    assert _post(client, {"clicks": 1}, cutoff=999).status_code == 422  # outside 0..400
