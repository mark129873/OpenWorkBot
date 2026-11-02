import asyncio
import json
import logging
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app import prompts, rag_parameters as params, tracing

logger = logging.getLogger(__name__)


class State(TypedDict, total=False):
    query: str
    attempt: int
    graded: bool
    plan: dict
    limits: dict
    rewrites: list[str]
    multi_queries: list[str]
    subqueries: list[str]
    expansions: list[str]
    queries: list[str]
    seen: list[str]
    candidates: list[dict]
    rerank_candidates: list[dict]
    evidence: list[dict]
    sufficient: bool
    can_retry: bool
    missing: list[str]
    warning: str
    status: str
    answer: str
    sources: list[dict]


def query_limits(plan: dict) -> dict[str, int]:
    remaining = params.QUERIES_PER_ROUND
    limits = {}
    if plan.get("rewrite") is True:
        limits["rewrite"] = min(params.REWRITE_QUERY_LIMIT, remaining)
        remaining -= limits["rewrite"]
    selected = [name for name in ("multi_query", "expand", "subquery") if plan.get(name) is True]
    if selected:
        share, extra = divmod(remaining, len(selected))
        limits.update({name: share + int(index < extra) for index, name in enumerate(selected)})
    return limits


async def grade_evidence(models, state: State) -> dict:
    evidence = state.get("evidence", [])
    if not evidence:
        return {"graded": True, "sufficient": False, "can_retry": True, "missing": ["相关依据"]}
    result = await models.structured(
        prompts.GRADE_SYSTEM,
        prompts.GRADE_USER.format(query=state["query"], evidence=json.dumps(evidence, ensure_ascii=False)),
        {"sufficient": False, "can_retry": True, "missing": []},
    )
    missing = result.get("missing", [])
    return {"graded": True, "sufficient": result.get("sufficient") is True,
            "can_retry": result.get("can_retry") is True,
            "missing": [x for x in missing if isinstance(x, str)] if isinstance(missing, list) else []}


def answer_body(answer: str, evidence: list[dict], question: str = "") -> str:
    names = {item.get("filename") for item in evidence if isinstance(item.get("filename"), str)}
    footers = re.findall(r"(?m)^[ \t]*参考来源[：:][^\r\n]*", answer)
    for footer in footers:
        names.update(re.sub(r"^【\d+】", "", name.strip())
                     for name in re.split(r"[、]", re.split(r"[：:]", footer, maxsplit=1)[1]))
    answer = re.sub(r"(?m)^[ \t]*参考来源[：:][^\r\n]*(?:\r?\n)?", "", answer)
    # shortcut: 无上下文的数字括号按引用处理，若业务编号误删需改为结构化引用。
    answer = re.sub(r"(?<!第)(?:【[0-9、，,\s]+】|(?<![A-Za-z0-9_])\[\d+\])(?![章节条款项号])",
                    lambda match: match.group() if match.group() in question else "", answer)
    variants = set(filter(None, names))
    for name in names:
        if "_" in name:
            prefix, title = name.split("_", 1)
            variants.add(f"{prefix}《{title}》")
    asks_for_file = re.search(r"文件名|文件名称|制度编号|文件编号|规程名称|哪(?:个|些|份)(?:文件|制度|规程|文档)", question)
    if variants and not asks_for_file:
        files = "|".join(re.escape(name) for name in sorted(variants, key=len, reverse=True))
        item = rf"\**`?《?(?:{files})》?`?\**(?=[ \t\r\n、；;，,。.!?）)】《]|$)"
        items = rf"{item}(?:[ \t]*[、；;，,]?[ \t]*{item})*"
        answer = re.sub(rf"[（(【][ \t]*(?:来源[：:][ \t]*)?{items}[ \t]*[）)】]", "", answer)
        def inline_source(match):
            punctuation = match.group("punctuation")
            previous = answer[:match.start()].rstrip()
            return "" if previous.endswith(("。", "！", "？", ".", "!", "?")) else punctuation

        answer, count = re.subn(rf"(?<!\w)(?:\*\*)?来源[：:](?:\*\*)?[ \t]*{items}(?P<punctuation>[。.])?",
                                inline_source, answer)
        if count:
            answer = re.sub(r"([（(])[ \t]*[；;，,、][ \t]*", r"\1", answer)
            answer = re.sub(r"[；;，,、][ \t]*([）)])", r"\1", answer)
            answer = re.sub(r"[（(][ \t]*[）)]", "", answer)
    if footers:
        answer = re.sub(r"\n{3,}(?=提示：部分查询失败)", "\n\n", answer)
    return answer.strip()


async def answer_evidence(models, state: State) -> dict:
    evidence = state.get("evidence", [])
    if not evidence:
        return {"status": "insufficient", "answer": "知识库中暂无足够的相关依据。", "sources": []}
    response = await models.text(
        prompts.ANSWER_SYSTEM,
        prompts.ANSWER_USER.format(query=state["query"], sufficient=state.get("sufficient", False),
                                   evidence=json.dumps(evidence, ensure_ascii=False)),
    )
    response = answer_body(response, evidence, state["query"])
    if not response:
        raise ValueError("清除引用后回答为空")
    if state.get("warning"):
        response += "\n\n提示：部分查询失败，结果可能不完整。"
    return {"status": "success" if state.get("sufficient") else "insufficient",
            "answer": response, "sources": []}


def build_graph(knowledge):
    models, vectors = knowledge.models, knowledge.vectors

    async def plan(state: State) -> dict:
        attempt = state.get("attempt", 0) + (1 if state.get("graded") else 0)
        instruction = prompts.PLAN_SUPPLEMENT if attempt else prompts.PLAN_INITIAL
        result = await models.structured(
            prompts.PLAN_SYSTEM,
            prompts.PLAN_USER.format(instruction=instruction, query=state["query"], missing=state.get("missing", [])),
            {"rewrite": False, "multi_query": False, "subquery": False, "expand": False},
        )
        return {"plan": result, "limits": query_limits(result),
                "attempt": attempt, "rewrites": [], "multi_queries": [], "subqueries": [], "expansions": []}

    async def variant(state: State, name: str, field: str, instruction: str) -> dict:
        limit = state.get("limits", {}).get(name, 0)
        if not limit:
            return {field: []}
        try:
            result = await models.structured(
                instruction,
                prompts.VARIANT_USER.format(query=state["query"], missing=state.get("missing", []),
                                            limit=limit), {"queries": []})
            values = result.get("queries", [])
        except Exception as exc:
            logger.warning("查询变体 %s 生成失败：%s", field, type(exc).__name__)
            return {field: []}
        return {field: [item.strip() for item in values if isinstance(item, str) and item.strip()][:limit]
                if isinstance(values, list) else []}

    async def rewrite(state: State) -> dict:
        return await variant(state, "rewrite", "rewrites", prompts.REWRITE_SYSTEM)

    async def multi_query(state: State) -> dict:
        return await variant(state, "multi_query", "multi_queries", prompts.MULTI_QUERY_SYSTEM)

    async def subquery(state: State) -> dict:
        return await variant(state, "subquery", "subqueries", prompts.SUBQUERY_SYSTEM)

    async def expand(state: State) -> dict:
        return await variant(state, "expand", "expansions", prompts.EXPAND_SYSTEM)

    def route(state: State) -> list[str]:
        plan = state.get("plan", {})
        return [name for name in ("rewrite", "multi_query", "expand", "subquery")
                if plan.get(name) is True] or ["retrieve"]

    async def retrieve(state: State) -> dict:
        plan = state.get("plan", {})
        sources = {}
        for name, field in (("rewrite", "rewrites"), ("multi_query", "multi_queries"),
                            ("expand", "expansions"), ("subquery", "subqueries")):
            if plan.get(name) is True:
                for query in state.get(field, []):
                    sources.setdefault(query, name)
        if not sources and state["query"].strip():
            sources[state["query"]] = "original"
        seen = set(state.get("seen", []))
        queries = [query for query in sources if query not in seen][:params.QUERIES_PER_ROUND]
        if not queries and not seen:
            raise RuntimeError("查询转换未生成有效查询")
        candidates = list(state.get("candidates", []))
        if queries:
            async with knowledge.catalog_lock:
                versions = knowledge.catalog.versions()

                async def search_one(query: str, strategy: str):
                    with tracing.observation("search_query", as_type="chain", input={"query": query},
                                             metadata={"round": state.get("attempt", 0) + 1,
                                                       "strategy": strategy}) as span:
                        for attempt in range(params.SEARCH_RETRIES + 1):
                            try:
                                vector = (await models.embed([query]))[0]
                                hits = await vectors.search(query, vector, versions)
                                tracing.update(span, output={"hits": len(hits), "attempts": attempt + 1})
                                return hits
                            except Exception:
                                if attempt >= params.SEARCH_RETRIES:
                                    raise

                results = await asyncio.gather(*(search_one(query, sources[query]) for query in queries),
                                               return_exceptions=True)
            failed = sum(isinstance(result, BaseException) for result in results)
            for result in results:
                if isinstance(result, list):
                    candidates.extend(result)
            if failed and not candidates:
                raise RuntimeError("知识库检索失败")
            warning = "部分查询失败" if failed else state.get("warning", "")
        else:
            warning = state.get("warning", "")
        unique = {}
        for item in candidates:
            if item["id"] not in unique or item.get("score", 0) > unique[item["id"]].get("score", 0):
                unique[item["id"]] = item
        ranked = sorted(unique.values(), key=lambda item: item.get("score", 0), reverse=True)[:params.RERANK_CANDIDATES]
        for index, item in enumerate(ranked):
            item["_rank_index"] = index
        return {"queries": queries, "seen": [*seen, *queries],
                "candidates": candidates, "rerank_candidates": ranked, "warning": warning}

    async def rerank_stage(state: State) -> dict:
        ranked = await models.rerank(state["query"], state.get("rerank_candidates", []))
        return {"evidence": await knowledge.parent_documents(ranked)}

    async def grade(state: State) -> dict:
        return await grade_evidence(models, state)

    async def answer(state: State) -> dict:
        return await answer_evidence(models, state)

    def grade_route(state: State) -> str:
        return ("plan" if not state.get("sufficient") and state.get("can_retry")
                and state.get("attempt", 0) < params.SUPPLEMENTAL_ROUNDS else "answer")

    graph = StateGraph(State)
    for name, node in (("plan", plan), ("rewrite", rewrite), ("multi_query", multi_query),
                       ("expand", expand), ("subquery", subquery), ("retrieve", retrieve), ("rerank_stage", rerank_stage),
                       ("grade", grade), ("answer", answer)):
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route, ["rewrite", "multi_query", "expand", "subquery", "retrieve"])
    for name in ("rewrite", "multi_query", "expand", "subquery"):
        graph.add_edge(name, "retrieve")
    graph.add_edge("retrieve", "rerank_stage")
    graph.add_edge("rerank_stage", "grade")
    graph.add_conditional_edges("grade", grade_route, {"plan": "plan", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()
