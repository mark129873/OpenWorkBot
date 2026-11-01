from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.api import chat as chat_api
from app.config import settings
from app.obs import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    from app.process_lock import ProcessLock

    lock = ProcessLock(settings.data_dir)
    lock.acquire()
    runtime = None
    try:
        (settings.data_dir / "workspace").mkdir(parents=True, exist_ok=True)
        setup_logging(settings.data_dir, settings.log_level)
        db.init_db(settings.data_dir)
        from app.agent import hooks
        from app.agent.runtime import Runtime

        hooks.register_default_hooks()
        runtime = Runtime()
        app.state.runtime = runtime
        await runtime.start()
        yield
    finally:
        if runtime is not None:
            await runtime.close()
        db.dispose_engine()
        lock.release()


def create_app() -> FastAPI:
    app = FastAPI(title="个人助手 harness", lifespan=lifespan)
    app.include_router(chat_api.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/permission/state")
    def permission_state() -> dict:
        from app.sessions import execution_state

        return execution_state.load_execution_state(settings.data_dir)

    return app


app = create_app()
