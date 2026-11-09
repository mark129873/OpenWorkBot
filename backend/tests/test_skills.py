import contextvars
import re

import pytest

from app.agent import skills
from app.user_context import bind_user, require_user


def test_skill_catalog_reads_only_valid_metadata(tmp_path):
    root = skills.user_skills_path(require_user())
    good = root / "code-review"
    good.mkdir(parents=True)
    (good / "SKILL.md").write_text(
        "---\nname: code-review\ndescription: >-\n  审查代码，\n  在请求审查时使用。\n"
        "license: internal\n---\n正文不进入目录", encoding="utf-8")
    for name, content in {
        "mismatch": "---\nname: other\ndescription: 说明\n---\n正文",
        "missing": "---\nname: missing\n---\n正文",
        "invalid": "---\nname: invalid\ndescription: [\n---\n正文",
        "bad_name": "---\nname: bad_name\ndescription: 说明\n---\n正文",
        "代码审查": "---\nname: 代码审查\ndescription: 中文技能\n---\n正文",
    }.items():
        directory = root / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(content, encoding="utf-8")

    assert skills.list_skills() == [{"name": "code-review", "description": "审查代码， 在请求审查时使用。"},
                                    {"name": "代码审查", "description": "中文技能"}]
    prompt = skills.skill_prompt()
    assert "code-review" in prompt and "审查代码" in prompt
    assert "正文不进入目录" not in prompt and "internal" not in prompt
    assert "正文" in skills.tool_load_skill("代码审查")


def test_skill_body_and_resources_are_loaded_on_demand(tmp_path):
    directory = skills.user_skills_path(require_user()) / "code-review"
    references = directory / "references"
    references.mkdir(parents=True)
    manifest = "---\nname: code-review\ndescription: 审查代码\n---\n完整说明"
    (directory / "SKILL.md").write_text(manifest, encoding="utf-8")
    content = "资料行\r\n" * 8000
    (references / "guide.md").write_bytes(content.encode("utf-8"))
    outside = tmp_path / "outside.txt"
    outside.write_text("不应读取", encoding="utf-8")

    assert skills.tool_load_skill("code-review").endswith(manifest)
    first = skills.tool_load_skill("code-review", "references/guide.md")
    match = re.search(r"\n\[继续读取：offset=(\d+)\]$", first)
    assert match
    second = skills.tool_load_skill("code-review", "references/guide.md", int(match.group(1)))
    assert first[:match.start()] + second == content
    assert skills.tool_load_skill("code-review", "../../outside.txt").startswith("Error:")
    assert skills.tool_load_skill("code-review", str(outside)).startswith("Error:")
    assert skills.tool_load_skill("../code-review").startswith("Error:")
    assert skills.tool_load_skill("code-review", "references/guide.md", -1).startswith("Error:")
    assert skills.tool_load_skill("code-review", "references/guide.md", 10 ** 100).startswith("Error:")

    try:
        (references / "link.txt").symlink_to(outside)
    except OSError:
        pass
    else:
        assert skills.tool_load_skill("code-review", "references/link.txt").startswith("Error:")


def test_user_skill_templates_and_loading_are_isolated():
    first, second = "a" * 32, "b" * 32
    skills.initialize_user_skills(first)
    first_root = skills.user_skills_path(first)
    original = (first_root / "code-review" / "SKILL.md").read_text(encoding="utf-8")
    (first_root / "code-review" / "SKILL.md").write_text(original + "\n甲的修改", encoding="utf-8")
    (first_root / "private").mkdir()
    (first_root / "private" / "SKILL.md").write_text(
        "---\nname: private\ndescription: 甲的技能\n---\n私有正文", encoding="utf-8")

    with bind_user(second):
        assert skills.list_skills() == []
        assert skills.tool_load_skill("code-review").startswith("Error:")
        skills.initialize_user_skills(second)
        assert skills.tool_load_skill("code-review").endswith(original)
        assert skills.tool_load_skill("private").startswith("Error:")
        assert skills.tool_load_skill("code-review", f"../../{first}/private/SKILL.md").startswith("Error:")
        assert "甲的技能" not in skills.skill_prompt()

    assert "甲的修改" in skills.tool_load_skill("code-review")
    assert "私有正文" in skills.tool_load_skill("private")
    with pytest.raises(FileExistsError):
        skills.initialize_user_skills(first)
    assert "甲的修改" in skills.tool_load_skill("code-review")


def test_skill_access_requires_user_identity():
    context = contextvars.Context()
    with pytest.raises(RuntimeError, match="缺少用户身份"):
        context.run(skills.list_skills)
    assert context.run(skills.tool_load_skill, "code-review").startswith("Error:")


def test_user_skill_directory_cannot_link_to_another_user():
    first, second = "a" * 32, "b" * 32
    skills.initialize_user_skills(first)
    other = skills.user_skills_path(second)
    try:
        other.symlink_to(skills.user_skills_path(first), target_is_directory=True)
    except OSError:
        pytest.skip("无法创建目录符号链接")
    with bind_user(second), pytest.raises(ValueError, match="不能是链接"):
        skills.list_skills()
