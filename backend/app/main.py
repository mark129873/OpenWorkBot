from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.api import sessions as sessions_api
from app.config import settings
from app.obs import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "workspace").mkdir(parents=True, exist_ok=True)
    setup_logging(settings.data_dir, settings.log_level)
    db.init_db(settings.data_dir)
    from app.agent import hooks

    hooks.register_default_hooks()

    import logging

    logging.getLogger(__name__).info(
        "后端启动完成：data_dir=%s model=%s", settings.data_dir, settings.model_id or "(未配置)"
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="个人助手 harness", lifespan=lifespan)
    app.include_router(sessions_api.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/permission/state")
    def permission_state() -> dict:
        from app.sessions import execution_state

        return execution_state.load_execution_state(settings.data_dir)

    return app


app = create_app()
