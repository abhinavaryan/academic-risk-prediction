import pytest

from riskml.config import MAX_UPLOAD_BYTES


def test_security_headers_present(client):
    r = client.get("/api/health")
    h = r.headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert "default-src 'none'" in h["content-security-policy"]
    assert h["referrer-policy"] == "strict-origin-when-cross-origin"
    assert len(h["x-request-id"]) >= 16
    assert h["cache-control"] == "no-store"


def test_request_id_echoed_when_safe(client):
    r = client.get("/api/health", headers={"x-request-id": "abc123DEF456ghi789"})
    assert r.headers["x-request-id"] == "abc123DEF456ghi789"
    r = client.get("/api/health", headers={"x-request-id": "<script>alert(1)</script>"})
    assert "<" not in r.headers["x-request-id"]


def test_errors_do_not_leak_tracebacks(make_client, monkeypatch):
    c = make_client()
    reg = c.app.state.registry

    def boom(_cutoff):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(reg, "get", boom)
    r = c.post("/api/predict", json={"cutoff": 60, "features": {}})
    assert r.status_code == 500
    text = r.text
    assert "secret internal detail" not in text and "Traceback" not in text
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]


def test_body_too_large_413(client):
    big = b"a,b\n" + b"1,2\n" * (MAX_UPLOAD_BYTES // 4 + 50_000)
    r = client.post("/api/custom/profile", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "too_large"


def test_rate_limit_429(make_client):
    c = make_client(rate_per_min=3)
    data = {"file": ("a.csv", b"a,b\n1,2\n", "text/csv")}
    codes = [c.post("/api/custom/profile", files=data).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_rate_limit_does_not_affect_predict(make_client):
    c = make_client(rate_per_min=1)
    for _ in range(5):
        assert c.post("/api/predict", json={"cutoff": 60, "features": {}}).status_code == 200


def test_busy_returns_503(make_client):
    c = make_client(max_trainings=1)
    slots = c.app.state.training_slots
    assert slots.acquire(blocking=False)
    try:
        r = c.post(
            "/api/custom/train",
            files={"file": ("a.csv", b"a,y\n1,Fail\n", "text/csv")},
            data={"config": '{"target": "y", "positive_values": ["Fail"]}'},
        )
        assert r.status_code == 503
        assert r.headers["retry-after"]
    finally:
        slots.release()


@pytest.mark.parametrize("config", ["not json", '{"target": 5}', '{"target": "y", "evil": 1}'])
def test_bad_config_rejected(client, config):
    r = client.post(
        "/api/custom/train",
        files={"file": ("a.csv", b"a,y\n1,Fail\n", "text/csv")},
        data={"config": config},
    )
    assert r.status_code == 422


def test_unreadable_upload_400(client):
    r = client.post(
        "/api/custom/profile", files={"file": ("x.csv", bytes(range(256)) * 20, "text/csv")}
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unreadable"


def test_cors_only_for_allowed_origins(make_client):
    c = make_client(cors_origins=["https://good.example"])
    ok = c.options(
        "/api/health",
        headers={"origin": "https://good.example", "access-control-request-method": "GET"},
    )
    bad = c.options(
        "/api/health",
        headers={"origin": "https://evil.example", "access-control-request-method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://good.example"
    assert "access-control-allow-origin" not in bad.headers
