import re
from pathlib import Path

import pytest

from app import models, prompts, rag_parameters as params, tracing
from app.chunking import chunks


def test_chunk_parameters_control_actual_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "CHUNK_SIZE", 4)
    monkeypatch.setattr(params, "CHUNK_OVERLAP", 1)
    document = {"tree": [{"id": "p1", "type": "paragraph", "text": "abcdefghi"}]}
    assert [chunk["content"] for chunk in chunks(document)] == ["abcd", "defg", "ghi"]


def test_extracted_prompts_include_question_and_evidence():
    question, evidence = "测试问题", "测试依据"
    plan = prompts.PLAN_USER.format(instruction=prompts.PLAN_INITIAL, query=question, missing=[])
    variant = prompts.VARIANT_USER.format(query=question, missing=[], limit=7)
    grade = prompts.GRADE_USER.format(query=question, evidence=evidence)
    answer = prompts.ANSWER_USER.format(query=question, sufficient=False, evidence=evidence)
    assert all(question in text for text in (plan, variant, grade, answer))
    assert all(evidence in text for text in (grade, answer))


def test_prompt_version_changes_when_a_template_changes(monkeypatch):
    previous = tracing.prompt_version()
    monkeypatch.setattr(prompts, "ANSWER_SYSTEM", prompts.ANSWER_SYSTEM + "补充要求")
    assert len(previous) == 64 and tracing.prompt_version() != previous


@pytest.mark.asyncio
async def test_configured_json_retries_are_used(monkeypatch):
    monkeypatch.setattr(params, "JSON_RETRIES", 2)
    calls = []

    class FakeModel:
        async def text(self, system, prompt):
            calls.append(prompt)
            return ("invalid", "[]", '{"ok":true}')[len(calls) - 1]

    assert await models.Models.structured(FakeModel(), "系统", "问题", {}) == {"ok": True}
    assert len(calls) == 3 and calls[0] == "问题"
    assert calls[1:] == ["问题" + prompts.JSON_RETRY_SUFFIX] * 2


def test_invalid_chunk_overlap_is_rejected():
    source = params.__file__
    code = re.sub(r"^CHUNK_OVERLAP\s*=.*$", "CHUNK_OVERLAP = CHUNK_SIZE",
                  Path(source).read_text(encoding="utf-8"), flags=re.MULTILINE)
    with pytest.raises(ValueError, match="切片长度必须大于重叠"):
        exec(compile(code, source, "exec"), {})
