import re

from app.agent.tools import safe_path
from app.config import settings


_DENY_PATTERNS: list[str] = [
    "rm -rf /",
    "remove-item -recurse -force c:\\",
    "format ",
    "diskpart",
    "shutdown /s",
    "shutdown -s",
    "rd /s /q c:\\",
]

# 仅接受无需 shell 求值的字面量。
_READONLY_ROOTS = frozenset({
    "ls", "dir", "cat", "type", "echo", "pwd", "head", "tail", "wc",
    "which", "where", "get-childitem", "get-content", "select-string",
})
_GIT_READONLY_OPTIONS = {
    "status": {"--short", "-s", "--branch", "-b", "--porcelain", "--porcelain=v1", "--porcelain=v2"},
    "log": {"--oneline", "--all", "--decorate", "--no-decorate", "--no-patch"},
    "branch": {"--list", "--all", "-a", "--remotes", "-r", "-v", "-vv", "--show-current", "--no-color"},
    "remote": {"-v", "--verbose"},
}


TOOL_SPECS: dict[str, dict] = {
    "bash":        {"permission": "bash",    "riskLevel": "high",   "sideEffectScope": "system",    "needsApproval": True,  "destructive": True,  "readOnly": False},
    "read_file":   {"permission": "read",    "riskLevel": "low",    "sideEffectScope": "none",      "needsApproval": False, "destructive": False, "readOnly": True},
    "glob":        {"permission": "read",    "riskLevel": "low",    "sideEffectScope": "none",      "needsApproval": False, "destructive": False, "readOnly": True},
    "ls":          {"permission": "read",    "riskLevel": "low",    "sideEffectScope": "none",      "needsApproval": False, "destructive": False, "readOnly": True},
    "grep":        {"permission": "read",    "riskLevel": "low",    "sideEffectScope": "none",      "needsApproval": False, "destructive": False, "readOnly": True},
    "write_file":  {"permission": "edit",    "riskLevel": "medium", "sideEffectScope": "workspace", "needsApproval": True,  "destructive": False, "readOnly": False},
    "str_replace": {"permission": "edit",    "riskLevel": "medium", "sideEffectScope": "workspace", "needsApproval": True,  "destructive": False, "readOnly": False},
    "write_todos": {"permission": "todo",    "riskLevel": "low",    "sideEffectScope": "session",   "needsApproval": False, "destructive": False, "readOnly": True},
    "load_skill":  {"permission": "read",    "riskLevel": "low",    "sideEffectScope": "none",      "needsApproval": False, "destructive": False, "readOnly": True},
}

_DEFAULT_SPEC = {"permission": "unknown", "riskLevel": "medium", "sideEffectScope": "workspace",
                 "needsApproval": True, "destructive": False, "readOnly": False}


def _is_readonly_bash(command: str) -> bool:
    """仅可证明的简单只读命令免批，不套用其他 shell 的解析规则。"""
    # 控制字符须在 strip 前拒绝；复杂 shell 语法转审批。
    if any(ord(ch) < 32 and ch != "\t" for ch in command):
        return False
    if not re.fullmatch(r"[\w ./\\:*?=,@+\-\t]+", command):
        return False
    words = command.strip().split()
    if not words:
        return False
    root = words[0].lower()
    if root in _READONLY_ROOTS:
        return True
    if root in {"python", "node"}:
        return words[1:] == ["--version"]
    if root != "git" or len(words) < 2:
        return False
    # Git 参数区分大小写，禁止 lower 后误放行写操作。
    # diff/show 可调用外部程序，未列出参数须审批。
    subcommand = words[1]
    options = _GIT_READONLY_OPTIONS.get(subcommand)
    if options is None:
        return False
    return all(arg in options or (subcommand == "log" and re.fullmatch(r"-\d+", arg))
               for arg in words[2:])


def _capability(tool_name: str, tool_input: dict) -> dict:
    spec = dict(TOOL_SPECS.get(tool_name, _DEFAULT_SPEC))
    if tool_name == "bash":
        command = str(tool_input.get("command", ""))
        if _is_readonly_bash(command):
            spec.update({"riskLevel": "low", "sideEffectScope": "none",
                         "needsApproval": False, "destructive": False, "readOnly": True})
    return spec


def hard_deny_reason(tool_name: str, tool_input: dict) -> str | None:
    if tool_name == "bash":
        return "当前沙箱禁止宿主命令"
    if tool_name == "rag_ingest":
        try:
            safe_path(str(tool_input.get("path", "")))
        except ValueError as exc:
            return str(exc)
    if tool_name in {"write_file", "str_replace", "read_file", "glob", "ls", "grep"}:
        return "文件工具必须在沙箱内执行"
    return None


_SUBJECT_KEYS = ("command", "url", "file_path", "path", "pattern", "document_id")


def _subject(tool_input: dict) -> str:
    for key in _SUBJECT_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def _match_content(content: str, subject: str) -> bool:
    """cmd:* 按词边界匹配；仅 * 是通配符。"""
    if content.endswith(":*"):
        prefix = content[:-2]
        return subject == prefix or subject.startswith(prefix + " ") or subject.startswith(prefix + "\t")
    if "*" in content:
        # 完整匹配，避免 .md 规则放行 .md.exe。
        pattern = ".*".join(re.escape(part) for part in content.split("*"))
        return re.fullmatch(pattern, subject, flags=re.DOTALL) is not None
    return subject == content


def match_rule(rules: dict, behavior: str, tool_name: str, tool_input: dict) -> dict | None:
    subject = _subject(tool_input)
    for rule in rules.get(behavior, []):
        if rule.get("toolName") != tool_name:
            continue
        content = rule.get("ruleContent")
        if not content or _match_content(content, subject):
            return rule
    return None


def _deny(rule_id: str, reason: str) -> dict:
    return {"decision": "deny", "rule_id": rule_id, "reason": reason}


def _ask(rule_id: str, reason: str) -> dict:
    return {"decision": "ask", "rule_id": rule_id, "reason": reason}


def _allow(rule_id: str, reason: str) -> dict:
    return {"decision": "allow", "rule_id": rule_id, "reason": reason}


def check_permission(
    mode: str, rules: dict, tool_name: str, tool_input: dict, *, sandboxed: bool = False,
) -> dict:
    """优先级：硬拒、yolo、deny、ask、allow、edit、build。"""
    cap = _capability(tool_name, tool_input)

    # 硬拒优先于所有模式和规则。
    sandbox_files = {"read_file", "write_file", "str_replace", "glob", "ls", "grep"}
    if sandboxed and tool_name in sandbox_files:
        path = tool_input.get("path")
        hard = None if isinstance(path, str) and path.startswith("/") and "\0" not in path else "路径必须是容器内绝对路径"
    else:
        hard = None if tool_name == "bash" and sandboxed else hard_deny_reason(tool_name, tool_input)
    if hard:
        return _deny("hard.deny", hard)

    if mode == "yolo":
        return _allow("mode.yolo", "完全访问模式放行")

    rule = match_rule(rules, "deny", tool_name, tool_input)
    if rule:
        return _deny("rule.deny", f"命中拒绝规则：{rule.get('ruleContent') or rule.get('toolName')}")

    if tool_name == "bash" and sandboxed:
        return _ask("sandbox.bash", "容器命令每次执行均需确认")

    rule = match_rule(rules, "ask", tool_name, tool_input)
    if rule:
        return _ask("rule.ask", "命中确认规则")

    rule = match_rule(rules, "allow", tool_name, tool_input)
    if rule:
        return _allow("rule.allow", f"命中允许规则：{rule.get('ruleContent') or rule.get('toolName')}")

    if tool_name in {"rag_search", "rag_list", "rag_ingest", "rag_remove"}:
        return _allow("mode.knowledge", "知识库操作按用户明确要求执行")

    if mode == "edit":
        if cap["permission"] == "edit" and cap["sideEffectScope"] == "workspace":
            return _allow("mode.edit.fileEdit", "自动编辑模式放行文件编辑工具")

    if cap["readOnly"] and not cap["destructive"] and not cap["needsApproval"]:
        return _allow("mode.build.readOnly", "只读工具放行")
    if cap["riskLevel"] == "critical":
        return _ask("mode.build.criticalRisk", "极高风险操作需要确认")
    if cap["riskLevel"] == "high":
        return _ask("mode.build.highRisk", "高风险操作需要确认")
    if (cap["sideEffectScope"] == "session" and cap["riskLevel"] == "low"
            and not cap["destructive"] and not cap["needsApproval"]):
        return _allow("mode.build.sessionState", "低风险会话态更新放行")
    if cap["needsApproval"] or cap["destructive"] or cap["sideEffectScope"] != "none":
        return _ask("mode.build.sideEffect", "该操作有副作用，需要确认")
    return _allow("mode.build.lowRisk", "低风险操作放行")


def evaluate(data_dir, tool_name: str, tool_input: dict, *, sandboxed: bool = False) -> dict:
    from app.sessions import execution_state

    state = execution_state.load_execution_state(data_dir or settings.data_dir)
    rules = execution_state.load_permission_rules(data_dir or settings.data_dir)
    return check_permission(state["mode"], rules, tool_name, tool_input, sandboxed=sandboxed)
