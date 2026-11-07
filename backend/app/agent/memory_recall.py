import html
import math
from collections import Counter
from itertools import islice

import jieba
import tiktoken


_MAX_TOKENS = 2000
_CORRECTION_TOKENS = 500


def _tokens(text: str) -> list[str]:
    return list(islice((part for part in jieba.cut(text[:4096].strip().lower())
                        if any(character.isalnum() for character in part)), 128))


def _rank(facts: list[dict], query: str) -> list[dict]:
    if not query.strip():
        return sorted(facts, key=lambda fact: fact["confidence"], reverse=True)
    query = query[:4096].strip().lower()
    query_tokens = list(dict.fromkeys(_tokens(query)))
    corpus = [_tokens(fact["content"]) for fact in facts]
    frequency = Counter(token for tokens in corpus for token in set(tokens))
    idf = {token: math.log((len(corpus) + 1) / (count + 1)) + 1 for token, count in frequency.items()}

    def score(item):
        fact, tokens = item
        content = fact["content"][:4096].strip().lower()
        containment = bool(content and (query in content or content in query))
        matched = total = 1.0 if containment else 0.0
        for token in query_tokens:
            weight = idf.get(token, 1.0) ** 2
            total += weight
            if token in tokens or (len(token) >= 4 and any(len(other) >= 4 and
                    (token.startswith(other) or other.startswith(token)) for other in tokens)):
                matched += weight
        relevance = math.sqrt(matched / total) if total else 0.0
        return (relevance + fact["confidence"]) / 2

    return [fact for fact, _ in sorted(zip(facts, corpus), key=score, reverse=True)]


def _snapshot_data(summary: dict, facts: list[dict], query: str) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    def count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    sections = []
    for section, heading, fields in (
        ("user", "User Context", (("workContext", "Work"), ("personalContext", "Personal"),
                                  ("topOfMind", "Current Focus"), ("cognitiveStyle", "Thinking Style"))),
        ("history", "History", (("recentMonths", "Recent"), ("earlierContext", "Earlier"),
                                ("longTermBackground", "Background"))),
    ):
        lines = [f"- {label}: {html.escape(summary[section][key]['summary'], quote=False)}"
                 for key, label in fields if summary[section][key]["summary"]]
        if lines:
            sections.append(f"{heading}:\n" + "\n".join(lines))

    def line(fact):
        category = html.escape(str(fact.get("category") or "context"), quote=False)
        content = html.escape(fact["content"].strip(), quote=False)
        result = f"- [{category} | {fact['confidence']:.2f}] {content}"
        if category == "correction" and fact.get("sourceError"):
            result += f" (avoid: {html.escape(str(fact['sourceError']).strip(), quote=False)})"
        return result

    ranked = _rank(facts, query)
    corrections = [fact for fact in ranked if fact.get("category") == "correction"]
    regular = [fact for fact in ranked if fact.get("category") != "correction"]
    guaranteed = []
    for fact in corrections:
        candidate = "\n".join([*guaranteed, line(fact)])
        if count(candidate) > _CORRECTION_TOKENS:
            break
        guaranteed.append(line(fact))
    guaranteed_cost = count("\n".join(guaranteed)) if guaranteed else 0
    fact_lines = list(guaranteed)
    for fact in regular:
        candidate = "\n\n".join([*sections, "Facts:\n" + "\n".join([*fact_lines, line(fact)])])
        if count(candidate) > _MAX_TOKENS:
            break
        fact_lines.append(line(fact))
    facts_block = "Facts:\n" + "\n".join(fact_lines) if fact_lines else ""
    text = "\n\n".join([*sections, *([facts_block] if facts_block else [])])
    limit = _MAX_TOKENS + guaranteed_cost
    if count(text) > limit:
        prefix = "\n\n".join(sections)
        low, high = 0, len(prefix)
        text = facts_block
        while low <= high:
            middle = (low + high) // 2
            candidate = prefix[:middle].rstrip() + "\n..."
            if facts_block:
                candidate += "\n\n" + facts_block
            if count(candidate) <= limit:
                text = candidate
                low = middle + 1
            else:
                high = middle - 1
    return text


def _snapshot(user_id: str, query: str) -> str:
    from app.agent.memory_store import _read_files

    summary, facts = _read_files(user_id)
    if not facts and not any(item["summary"] for section in ("user", "history") for item in summary[section].values()):
        return ""
    return _snapshot_data(summary, facts, query)
