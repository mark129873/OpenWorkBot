import hashlib
import hmac
import secrets
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app import db
from app.agent.skills import initialize_user_skills
from app.models import LoginSession, User


router = APIRouter(prefix="/api/auth")
SESSION_COOKIE = "openworkbot_session"
CSRF_COOKIE = "openworkbot_csrf"
SESSION_SECONDS = 7 * 24 * 60 * 60


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


def token_id(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def lookup(token: str | None) -> tuple[str, str, str] | None:
    if not token or len(token) > 256:
        return None
    with db.new_session() as session:
        login = session.get(LoginSession, token_id(token))
        if not login or login.expires_at <= int(time.time()):
            return None
        user = session.get(User, login.user_id)
        if not user or user.disabled:
            return None
        return user.id, user.username, login.csrf


def valid_csrf(request: Request, principal: tuple[str, str, str] | None) -> bool:
    token = request.headers.get("x-csrf-token", "")
    cookie = request.cookies.get(CSRF_COOKIE, "")
    if not token or not cookie or len(token) > 256 or len(cookie) > 256:
        return False
    token_bytes, cookie_bytes = token.encode("utf-8"), cookie.encode("utf-8")
    return hmac.compare_digest(token_bytes, cookie_bytes) and (
        principal is None or hmac.compare_digest(token_bytes, principal[2].encode("utf-8")))


def _csrf_cookie(response: JSONResponse, csrf: str) -> None:
    response.set_cookie(CSRF_COOKIE, csrf, samesite="strict", httponly=False, path="/")


@router.get("/session")
def current_session(request: Request):
    principal = request.state.principal
    csrf = principal[2] if principal else secrets.token_urlsafe(32)
    response = JSONResponse({"username": principal[1] if principal else None, "csrf": csrf})
    _csrf_cookie(response, csrf)
    return response


def _login_response(session, user: User, request: Request) -> JSONResponse:
    previous = request.cookies.get(SESSION_COOKIE)
    if previous:
        session.execute(delete(LoginSession).where(LoginSession.id == token_id(previous)))
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    session.add(LoginSession(id=token_id(token), user_id=user.id, csrf=csrf,
                             expires_at=int(time.time()) + SESSION_SECONDS))
    session.commit()
    response = JSONResponse({"username": user.username, "csrf": csrf})
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict",
                        max_age=SESSION_SECONDS, path="/")
    _csrf_cookie(response, csrf)
    return response


@router.post("/register", status_code=201)
def register(body: Credentials, request: Request):
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="账号名不能为空")
    with db.new_session() as session:
        user = User(username=username, password=body.password)
        session.add(user)
        try:
            session.flush()
            initialize_user_skills(user.id)
            response = _login_response(session, user, request)
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail="账号名已存在") from exc
    response.status_code = 201
    return response


@router.post("/login")
def login(body: Credentials, request: Request):
    with db.new_session() as session:
        user = session.scalar(select(User).where(User.username == body.username.strip()))
        if (not user or user.disabled or not hmac.compare_digest(
                user.password.encode("utf-8"), body.password.encode("utf-8"))):
            raise HTTPException(status_code=401, detail="账号或密码错误")
        return _login_response(session, user, request)


@router.post("/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with db.new_session() as session:
            session.execute(delete(LoginSession).where(LoginSession.id == token_id(token)))
            session.commit()
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/", samesite="strict")
    response.delete_cookie(CSRF_COOKIE, path="/", samesite="strict")
    return response
