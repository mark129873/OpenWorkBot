import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.runtime import Conflict

router = APIRouter(prefix="/api/chat")


async def get_runtime(request: Request):
    return await request.app.state.gateway.runtime(request.state.principal[0])


class MessageInput(BaseModel):
    assistant_id: str = Field(min_length=1, max_length=128)
    client_message_id: str = Field(min_length=1, max_length=128)
    text: str = Field(default="", max_length=100000)
    permission_mode: str | None = None
    attachments: list[str] = Field(default_factory=list)


class ApprovalInput(BaseModel):
    assistant_id: str = Field(min_length=1, max_length=128)
    option_id: str
    feedback: str | None = Field(default=None, max_length=4096)


class ResetInput(BaseModel):
    assistant_id: str = Field(min_length=1, max_length=128)
    reset_id: str = Field(min_length=1, max_length=128)
    confirm_stopped: bool = False


@router.get("")
async def chat(request: Request, limit: int | None = Query(default=None, ge=1, le=200),
               before: int | None = Query(default=None, ge=0)):
    if before is not None and limit is None:
        raise HTTPException(status_code=400, detail="分页需要指定数量")
    return await (await get_runtime(request)).snapshot(limit, before)


@router.post("/messages", status_code=202)
async def send_message(body: MessageInput, request: Request):
    try:
        return await (await get_runtime(request)).accept(body.client_message_id, body.text, body.permission_mode,
                                                         assistant_id=body.assistant_id, attachments=body.attachments)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/uploads")
async def upload_file(request: Request, assistant_id: str = Form(), upload_id: str = Form(),
                      file: UploadFile = File()):
    from app.api.uploads import upload
    try:
        return await upload(await get_runtime(request), assistant_id, upload_id, file)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/approvals/{request_id}")
async def approve(request_id: str, body: ApprovalInput, request: Request):
    try:
        return await (await get_runtime(request)).decide(request_id, body.option_id, body.feedback,
                                                         assistant_id=body.assistant_id)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset", status_code=202)
async def reset(body: ResetInput, request: Request):
    try:
        return await (await get_runtime(request)).reset(body.assistant_id, body.reset_id, body.confirm_stopped)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset/cancel")
async def cancel_reset(body: ResetInput, request: Request):
    try:
        return await (await get_runtime(request)).cancel_reset(body.assistant_id, body.reset_id)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/events")
async def events(request: Request, limit: int | None = Query(default=None, ge=1, le=200)):
    from app.api.auth import SESSION_COOKIE, lookup

    runtime = await get_runtime(request)
    token = request.cookies.get(SESSION_COOKIE)
    queue, snapshot = await runtime.subscribe(limit)

    async def stream():
        try:
            initial = {"revision": snapshot["revision"], "type": "snapshot", "data": snapshot}
            yield f"data: {json.dumps(initial, ensure_ascii=False)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    if not lookup(token):
                        return
                    yield ": keepalive\n\n"
                    continue
                if event is None or not lookup(token):
                    return
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            runtime.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
