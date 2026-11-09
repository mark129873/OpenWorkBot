import pytest

from app.config import settings
from app.user_context import bind_user


TEST_USER = "a" * 32


@pytest.fixture(autouse=True)
def isolated_user(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path)
    monkeypatch.setattr(settings, "user_skills_dir", tmp_path / "user_skills")
    with bind_user(TEST_USER):
        yield


def install_runtime(app, runtime):
    class Gateway:
        async def runtime(self, user_id):
            assert user_id == TEST_USER
            return runtime

    app.state.gateway = Gateway()

    @app.middleware("http")
    async def bind_principal(request, call_next):
        request.state.principal = (TEST_USER, "test", "csrf")
        with bind_user(TEST_USER):
            return await call_next(request)
