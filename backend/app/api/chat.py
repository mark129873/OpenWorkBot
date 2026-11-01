import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.runtime import Conflict

router = APIRouter(prefix="/api/chat")


class MessageInput(BaseModel):
    client_message_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=100000)
    permission_mode: str | None = None


class ApprovalInput(BaseModel):
    option_id: str
    feedback: str | None = Field(default=None, max_length=4096)


@router.get("")
async def chat(request: Request):
    return await request.app.state.runtime.snapshot()


@router.post("/messages", status_code=202)
async def send_message(body: MessageInput, request: Request):
    try:
        return await request.app.state.runtime.accept(body.client_message_id, body.text, body.permission_mode)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/approvals/{request_id}")
async def approve(request_id: str, body: ApprovalInput, request: Request):
    try:
        return await request.app.state.runtime.decide(request_id, body.option_id, body.feedback)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
