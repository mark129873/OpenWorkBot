import re

from app.agent import skills


def test_skill_catalog_reads_only_valid_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path)
    good = tmp_path / "code-review"
    good.mkdir()
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
        directory = tmp_path / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(content, encoding="utf-8")

    assert skills.list_skills() == [{"name": "code-review", "description": "审查代码， 在请求审查时使用。"},
                                    {"name": "代码审查", "description": "中文技能"}]
    prompt = skills.skill_prompt()
    assert "code-review" in prompt and "审查代码" in prompt
    assert "正文不进入目录" not in prompt and "internal" not in prompt
    assert "正文" in skills.tool_load_skill("代码审查")


def test_skill_body_and_resources_are_loaded_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path / "skills")
    directory = skills.SKILLS_DIR / "code-review"
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
