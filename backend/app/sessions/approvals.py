from app.agent.permission_service import evaluate
from app.sessions import session_permissions


def _options_for_scope(scope: tuple[str, str] | None) -> list[dict]:
    """展示与授权共用路径快照，避免符号链接变化扩权。"""
    return [
        {"option_id": "allowOnce", "label": "仅本次允许"},
        {"option_id": "allowSession", "label": "本会话内始终允许",
         "content": session_permissions.describe(scope)},
        {"option_id": "deny", "label": "拒绝"},
    ]


def build_options(tool_name: str, tool_input: dict, *, canonical_path: str | None = None) -> list[dict]:
    options = _options_for_scope(session_permissions.scope_for(tool_name, tool_input, canonical_path=canonical_path))
    return [option for option in options if tool_name != "bash" or option["option_id"] != "allowSession"]


async def authorize_tool(deps, tool_name: str, tool_input: dict, *, data_dir=None, sandboxed: bool = False,
                         canonical_path: str | None = None) -> str | None:
    """会话授权仅覆盖 ask，不能覆盖硬拒或 deny。"""
    if deps.stop_flag():
        return "Error: 操作已停止"
    result = evaluate(data_dir, tool_name, tool_input, sandboxed=sandboxed)
    decision, reason = result["decision"], result["reason"]
    if decision == "deny":
        return f"Error: 已被安全策略拒绝：{reason}"
    if decision == "allow" or (not (tool_name == "bash" and sandboxed)
                                and session_permissions.allows(deps.recorder.session_id, tool_name, tool_input,
                                                               canonical_path=canonical_path)):
        return None
    outcome = await deps.approver(tool_name, tool_input, reason or "该操作需要确认")
    if deps.stop_flag():
        return "Error: 操作已停止"
    if not outcome["approved"]:
        return f"Error: 用户拒绝了该操作：{outcome['denial_reason'] or reason}"
    # 等待期间规则可能变化，执行前重查拒绝规则。
    latest = evaluate(data_dir, tool_name, tool_input, sandboxed=sandboxed)
    if latest["decision"] == "deny":
        return f"Error: 已被安全策略拒绝：{latest['reason']}"
    return None
