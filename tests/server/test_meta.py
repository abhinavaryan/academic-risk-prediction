def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"]["available"] == [0, 60]


def test_meta_lists_limits_and_models(client):
    body = client.get("/api/meta").json()
    assert body["cutoffs"] == [0, 60]
    assert body["primary_cutoff"] == 60
    assert body["limits"]["max_upload_bytes"] == 4_000_000
    assert "Logistic Regression" in body["custom_models"]


def test_schema_and_samples(client):
    s = client.get("/api/schema/60")
    assert s.status_code == 200
    assert s.json()["threshold"] == 0.45
    samples = client.get("/api/samples/60").json()
    assert samples[0]["id"] == "AAA-2014J-1"


def test_unknown_cutoff_404(client):
    r = client.get("/api/schema/45")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_cutoff"


def test_openapi_docs_available(client):
    assert client.get("/api/openapi.json").status_code == 200
    assert client.get("/api/docs").status_code == 200


def test_sample_dataset_download(client):
    r = client.get("/api/custom/sample")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert r.content.startswith(b"\xef\xbb\xbfMarital status;")


def test_static_site_served_when_enabled(make_client, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>Agrim</body></html>")
    c = make_client(static_dir=tmp_path)
    assert "Agrim" in c.get("/").text
    assert c.get("/api/health").status_code == 200
