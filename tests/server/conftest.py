import pytest
from fastapi.testclient import TestClient

from riskml.server.main import create_app
from tests.serving.conftest import write_artifacts


@pytest.fixture(scope="session")
def server_artifacts(tmp_path_factory):
    return write_artifacts(tmp_path_factory.mktemp("server_artifacts"))


@pytest.fixture
def make_client(server_artifacts):
    def factory(**kwargs):
        kwargs.setdefault("artifacts_dir", server_artifacts)
        return TestClient(create_app(**kwargs), raise_server_exceptions=False)
    return factory


@pytest.fixture
def client(make_client):
    return make_client()
