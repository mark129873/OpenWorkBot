import pytest

from app.graph import answer_body, answer_evidence


class Models:
    def __init__(self):
        self.calls = []

    async def text(self, system, prompt):
        self.calls.append((system, prompt))
        return "根据原文直接回答。"


@pytest.mark.asyncio
async def test_answer_uses_evidence_without_adding_sources():
    evidence = [{"document_id": "1", "filename": "a.pdf", "content": "资料原文"}]
    models = Models()
    result = await answer_evidence(models, {"query": "问题", "evidence": evidence, "sufficient": True})
    assert result == {"status": "success", "answer": "根据原文直接回答。", "sources": []}
    system, prompt = models.calls[0]
    assert "不附引用编号或来源列表" in system
    assert "除非问题明确询问文件名或制度编号" in system
    assert "资料原文" in prompt and "source_number" not in prompt


@pytest.mark.asyncio
async def test_insufficient_answer_preserves_search_warning():
    result = await answer_evidence(Models(), {"query": "问题", "sufficient": False, "warning": "失败",
        "evidence": [{"filename": "a.pdf", "content": "资料原文"}]})
    assert result["status"] == "insufficient" and result["sources"] == []
    assert result["answer"] == "根据原文直接回答。\n\n提示：部分查询失败，结果可能不完整。"


@pytest.mark.asyncio
async def test_no_evidence_returns_insufficient_without_calling_model():
    models = Models()
    result = await answer_evidence(models, {"query": "问题", "evidence": []})
    assert result == {"status": "insufficient", "answer": "知识库中暂无足够的相关依据。", "sources": []}
    assert not models.calls


def test_citation_cleanup_preserves_facts_warnings_and_business_numbers():
    answer = ("单笔1000万元【1】，期限T+3【2】。\n\n参考来源：【1】A.pdf、【2】B.pdf"
              "\n\n提示：部分查询失败，结果可能不完整。")
    assert answer_body(answer, []) == ("单笔1000万元，期限T+3。\n\n"
                                       "提示：部分查询失败，结果可能不完整。")
    assert answer_body("结论【4】。另一结论[4]。", []) == "结论。另一结论。"
    assert answer_body("第【4】条，范围[0,1]，数组a[4]。", [], "解释第【4】条") == "第【4】条，范围[0,1]，数组a[4]。"
    assert answer_body("【4】是必要条件。", [], "【4】是什么意思") == "【4】是必要条件。"
    assert answer_body("保留\n\n\n空行\n\n参考来源：A.pdf\n\n提示：部分查询失败。", []) == (
        "保留\n\n\n空行\n\n提示：部分查询失败。")


def test_citation_cleanup_preserves_facts_inside_mixed_parentheses():
    evidence = [{"filename": "A.pdf"}, {"filename": "B.pdf"}]
    answer = "第一项（《A.pdf》）。第二项【A.pdf；B.pdf】。第三项（来源：A.pdf；适用属地细则，来源：B.pdf）。"
    assert answer_body(answer, evidence) == "第一项。第二项。第三项（适用属地细则）。"
    assert answer_body("适用文件为（《A.pdf》）。", evidence, "适用哪个文件？") == "适用文件为（《A.pdf》）。"
    assert answer_body(answer_body(answer, evidence), evidence) == answer_body(answer, evidence)
    assert answer_body("事实。**来源：**《A.pdf》。", evidence) == "事实。"
    assert answer_body("事实来源：A.pdf。", evidence) == "事实来源：A.pdf。"
    assert answer_body("来源：alpha是合法资金。", [{"filename": "a"}]) == "来源：alpha是合法资金。"


@pytest.mark.asyncio
async def test_citation_only_model_output_is_not_treated_as_answer():
    class CitationOnly:
        async def text(self, system, prompt):
            return "【1】\n\n参考来源：【1】A.pdf"

    with pytest.raises(ValueError, match="清除引用后回答为空"):
        await answer_evidence(CitationOnly(), {"query": "问题", "sufficient": True,
            "evidence": [{"filename": "A.pdf", "content": "资料"}]})
