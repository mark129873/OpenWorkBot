import pytest

from app.config import settings
from app.user_context import bind_user


TEST_USER = "a" * 32


@pytest.fixture(autouse=True)
def isolated_user(tmp_path, monkeypatch):
    from app.models import User
    from app.sessions import chat_store
    from app.user_context import require_user

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "database_backend", "sqlite")
    monkeypatch.setattr(settings, "database_url", "")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path)
    monkeypatch.setattr(settings, "user_skills_dir", tmp_path / "user_skills")
    initialize = chat_store.initialize

    def initialize_test_user(session, data_dir):
        user_id = require_user()
        if session.get(User, user_id) is None:
            session.add(User(id=user_id, username=f"test-{user_id}", password="test"))
            session.flush()
        initialize(session, data_dir)

    monkeypatch.setattr(chat_store, "initialize", initialize_test_user)
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
