import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.runtime import Conflict

router = APIRouter(prefix="/api/chat")


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
async def chat(request: Request):
    return await request.app.state.runtime.snapshot()


@router.post("/messages", status_code=202)
async def send_message(body: MessageInput, request: Request):
    try:
        return await request.app.state.runtime.accept(body.client_message_id, body.text, body.permission_mode,
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
        return await upload(request.app.state.runtime, assistant_id, upload_id, file)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/approvals/{request_id}")
async def approve(request_id: str, body: ApprovalInput, request: Request):
    try:
        return await request.app.state.runtime.decide(request_id, body.option_id, body.feedback,
                                                      assistant_id=body.assistant_id)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset", status_code=202)
async def reset(body: ResetInput, request: Request):
    try:
        return await request.app.state.runtime.reset(body.assistant_id, body.reset_id, body.confirm_stopped)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset/cancel")
async def cancel_reset(body: ResetInput, request: Request):
    try:
        return await request.app.state.runtime.cancel_reset(body.assistant_id, body.reset_id)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/events")
async def events(request: Request):
    runtime = request.app.state.runtime
    queue, snapshot = await runtime.subscribe()

    async def stream():
        try:
            initial = {"revision": snapshot["revision"], "type": "snapshot", "data": snapshot}
            yield f"data: {json.dumps(initial, ensure_ascii=False)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    return
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            runtime.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
