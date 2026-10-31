"""SSE 断开只停止读流，后台轮次继续执行并落盘。"""

import asyncio
import json
import logging
from pathlib import Path
from contextlib import contextmanager

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DbSession

from app import db
from app.agent.llm import build_client
from app.agent.loop import TurnDeps, run_turn
from app.config import settings, validate_llm_config
from app.models import new_id
from app.sessions import approvals, execution_state, replay, store, turn_manager
from app.sessions.approvals import InteractiveApprover
from app.sessions.recorder import TurnRecorder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# 心跳保活，避免空闲连接被代理关闭。
_KEEPALIVE_SECONDS = 15

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "agent" / "system_prompt.md"


class TurnIn(BaseModel):
    text: str
    mode: str | None = None


def get_llm_client() -> object:
    return build_client(settings)


def _load_system_prompt() -> str:
    """每轮重读系统提示词，修改下一轮生效。"""
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _sse_frame(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("")
def create_session() -> dict:
    """空会话只分配 ID，首条消息才落库。"""
    return {"id": new_id()}


@router.get("")
def list_sessions(db_session: DbSession = Depends(db.get_db)) -> list[dict]:
    items = []
    for row in store.list_sessions(db_session):
        items.append(
            {
                **store.session_info(db_session, row),
                "running": turn_manager.is_running(row.id),
            }
        )
    return items


@router.get("/{session_id}")
def get_session_detail(session_id: str, db_session: DbSession = Depends(db.get_db)) -> dict:
    result = replay.load_replay(
        db_session, session_id, running_turn_ids=turn_manager.running_turn_ids()
    )
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return result


@router.delete("/{session_id}")
async def delete_session(session_id: str, db_session: DbSession = Depends(db.get_db)) -> dict:
    # 检查至归档不得让出执行，避免与轮次登记交错。
    if turn_manager.is_running(session_id):
        raise HTTPException(status_code=409, detail="会话正在生成中，请先停止")
    if not store.soft_delete_session(db_session, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    approvals.forget_session(session_id)
    return {"ok": True}


@router.post("/{session_id}/turn")
async def start_turn(
    session_id: str,
    body: TurnIn,
    client: object = Depends(get_llm_client),
) -> StreamingResponse:
    _check_llm_config()
    _check_not_running(session_id)

    with _prepare_turn(session_id) as (recorder, stop_event):
        # 先拒绝已删会话，避免失败请求改变项目权限。
        if body.mode:
            execution_state.save_execution_state(settings.data_dir, body.mode)
        history = replay.load_history(recorder.db, session_id)
        history.append({"role": "user", "content": body.text})
        return _turn_sse_response(client, recorder, history, stop_event)


@router.post("/{session_id}/regenerate")
async def regenerate_turn(
    session_id: str,
    client: object = Depends(get_llm_client),
) -> StreamingResponse:
    _check_llm_config()
    _check_not_running(session_id)

    with _prepare_turn(session_id) as (recorder, stop_event):
        turn_db = recorder.db
        last_user = store.find_last_user_message(turn_db, session_id)
        if last_user is None:
            raise HTTPException(status_code=404, detail="没有可重新生成的轮次")

        # 保留用户消息，装配失败后仍可重跑。
        store.rollback_turn(turn_db, session_id, store.turn_id(last_user), last_user.sequence)
        store.retag_message(turn_db, last_user, recorder.turn_id)
        history = replay.load_history(turn_db, session_id)
        return _turn_sse_response(
            client, recorder, history, stop_event, user_sequence=last_user.sequence
        )


@contextmanager
def _prepare_turn(session_id: str):
    """装配成功移交 Session；失败或取消须释放连接及轮次登记。"""
    turn_db = db.new_session()
    recorder = None
    try:
        # 未落库草稿可发送，已归档会话不可恢复执行。
        if store.is_session_archived(turn_db, session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        recorder = TurnRecorder(turn_db, session_id, model=settings.model_id, max_tokens=settings.max_tokens)
        stop_event = turn_manager.register(session_id, recorder.turn_id)
        yield recorder, stop_event
    except BaseException:
        # 关闭连接并回滚，释放单轮槽位。
        try:
            if recorder is not None:
                turn_manager.unregister(session_id, recorder.turn_id)
        finally:
            turn_db.close()
        raise


def _check_llm_config() -> None:
    missing = validate_llm_config(settings)
    if missing:
        raise HTTPException(status_code=400, detail=f"缺少 LLM 配置：{','.join(missing)}")


def _check_not_running(session_id: str) -> None:
    if turn_manager.is_running(session_id):
        raise HTTPException(status_code=409, detail="该会话正在生成中")


def _turn_sse_response(
    client, recorder: TurnRecorder, history: list, stop_event: asyncio.Event,
    *, user_sequence: int | None = None,
) -> StreamingResponse:
    session_id, turn_db = recorder.session_id, recorder.db
    queue: asyncio.Queue = asyncio.Queue()
    # 审批与循环共用队列，否则等待事件无法送达。
    deps = TurnDeps(
        client=client,
        recorder=recorder,
        system_prompt=_load_system_prompt(),
        history=history,
        stop_flag=stop_event.is_set,
        approver=InteractiveApprover(turn_db, session_id, recorder, queue, stop_event),
        settings=settings,
        queue=queue,
        persist_user_message=user_sequence is None,
        user_sequence=user_sequence,
    )

    async def pump() -> None:
        try:
            async for event in run_turn(deps):
                await queue.put(event)
        except Exception as exc:
            logger.exception("turn 泵异常")
            await queue.put({"type": "error", "message": str(exc)})
            await queue.put(
                {"type": "turn_completed", "turn_id": recorder.turn_id,
                 "ended_at": 0, "active_ms": 0, "state": "failed"}
            )
        finally:
            await queue.put(None)
            turn_manager.unregister(session_id, recorder.turn_id)
            turn_db.close()

    async def event_stream():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield _sse_frame(event)

    # 响应装配成功后才启动后台任务。
    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    coroutine = pump()
    try:
        asyncio.create_task(coroutine)
    except BaseException:
        coroutine.close()
        raise
    return response


@router.post("/{session_id}/stop")
async def stop_turn(session_id: str) -> dict:
    if not turn_manager.request_stop(session_id):
        raise HTTPException(status_code=404, detail="该会话没有进行中的 turn")
    return {"ok": True}


class ApprovalIn(BaseModel):
    """旧 approved 字段仅兼容单次决定，不授予会话权限。"""

    request_id: str
    option_id: str | None = None
    feedback: str | None = None
    approved: bool | None = None


@router.post("/{session_id}/approval")
async def submit_approval(session_id: str, body: ApprovalIn) -> dict:
    info = approvals.get_request(body.request_id)
    if info is None or info["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="审批请求不存在或已处理")
    # 空字符串也是未知选项，不回退到布尔批准。
    option_id = body.option_id
    if option_id is None and body.approved is not None:
        option_id = "allowOnce" if body.approved else "deny"
    if option_id not in {o["option_id"] for o in info["options"]}:
        raise HTTPException(status_code=400, detail="不支持的审批选项")
    if option_id == "allowSession" and info["scope"] is None:
        raise HTTPException(status_code=400, detail="无法确定会话授权范围")
    feedback = (body.feedback or "").strip()[:4096] or None
    if not approvals.resolve(body.request_id, option_id, session_id=session_id, feedback=feedback):
        raise HTTPException(status_code=404, detail="审批请求不存在或已处理")
    return {"ok": True}
