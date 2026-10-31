import re

from app.config import BACKEND_ROOT

SKILLS_DIR = BACKEND_ROOT / "skills"
_NAME_RE = re.compile(r"^[\w-]+$")


def tool_load_skill(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        return f"Error: 非法技能名：{name}"
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir())) if SKILLS_DIR.exists() else ""
        return f"Error: 技能不存在：{name}" + (f"（可用：{available}）" if available else "")
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error: 读取技能失败：{exc}"


def list_skill_names() -> list[str]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(p.name for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())
