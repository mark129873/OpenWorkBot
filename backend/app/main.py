from contextlib import asynccontextmanager
import ipaddress

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.agent.checkpoint import SqliteSaver

from app import db
from app.api import auth as auth_api
from app.api import chat as chat_api
from app.config import settings
from app.obs import setup_logging
from app.user_context import bind_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    from app.process_lock import ProcessLock

    lock = ProcessLock(settings.data_dir)
    lock.acquire()
    gateway = None
    try:
        setup_logging(settings.data_dir, settings.log_level)
        db.init_db(settings.data_dir)
        from app.agent import hooks
        from app.gateway import Gateway

        hooks.register_default_hooks()
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.conn.execute("PRAGMA journal_mode=WAL")
            await saver.conn.execute("PRAGMA busy_timeout=5000")
            await saver.setup()
            gateway = Gateway(saver)
            app.state.gateway = gateway
            await gateway.start()
            try:
                yield
            finally:
                await gateway.close()
    finally:
        db.dispose_engine()
        lock.release()


def create_app() -> FastAPI:
    app = FastAPI(title="个人助手 harness", lifespan=lifespan)
    app.include_router(auth_api.router)
    app.include_router(chat_api.router)

    @app.middleware("http")
    async def authenticate(request, call_next):
        try:
            local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
        except ValueError:
            local = False
        if not local or request.url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse({"detail": "仅允许本机访问"}, status_code=403)
        if request.url.path == "/api/health":
            return await call_next(request)
        principal = auth_api.lookup(request.cookies.get(auth_api.SESSION_COOKIE))
        request.state.principal = principal
        public = request.url.path in {"/api/auth/session", "/api/auth/register", "/api/auth/login"}
        if not principal and not public:
            return JSONResponse({"detail": "请先登录"}, status_code=401)
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not auth_api.valid_csrf(request, principal):
            return JSONResponse({"detail": "请求校验失败"}, status_code=403)
        if principal:
            with bind_user(principal[0]):
                return await call_next(request)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/permission/state")
    def permission_state() -> dict:
        from app.sessions import execution_state

        return execution_state.load_execution_state(settings.data_dir)

    return app


app = create_app()
