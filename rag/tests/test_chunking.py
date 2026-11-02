import pytest

from app import rag_parameters as params
from app.chunking import chunks


def test_chunks_follow_heading_hierarchy_and_preserve_pages():
    document = {"tree": [
        {"id": "h1", "type": "heading", "text": "制度", "page": 1},
        {"id": "h2", "type": "heading", "text": "申请材料", "parent": "h1", "page": 1},
        {"id": "p1", "type": "paragraph", "text": "材料清单", "parent": "h2", "page": 1},
        {"id": "p2", "type": "paragraph", "text": "补充材料", "parent": "h2", "page": 2},
        {"id": "h3", "type": "heading", "text": "限额", "parent": "h1", "page": 2},
        {"id": "t1", "type": "table", "cells": [["项目", "限额"], ["单笔", "300万元"]],
         "parent": "h3", "page": 2},
    ]}
    result = list(chunks(document))
    assert len(result) == 2
    assert result[0] == {"content": "材料清单\n补充材料", "heading_path": ["制度", "申请材料"],
                         "pages": [1, 2], "search_text": "制度\n申请材料\n材料清单\n补充材料"}
    assert result[1]["content"] == "项目\t限额\n单笔\t300万元"
    assert result[1]["heading_path"] == ["制度", "限额"]
    assert result[1]["pages"] == [2]


def test_long_content_keeps_overlap_and_includes_titles_in_limit(monkeypatch):
    monkeypatch.setattr(params, "CHUNK_SIZE", 100)
    monkeypatch.setattr(params, "CHUNK_OVERLAP", 10)
    text = "资料甲乙丙丁戊己庚辛壬癸" * 30
    document = {"tree": [
        {"id": "h1", "type": "heading", "text": "制度标题", "page": 1},
        {"id": "p1", "type": "paragraph", "text": text, "parent": "h1", "page": 1},
    ]}
    result = list(chunks(document))
    assert len(result) > 2
    assert all(len(item["search_text"]) <= 100 for item in result)
    assert all(item["heading_path"] == ["制度标题"] for item in result)
    assert all(left["content"][-10:] == right["content"][:10]
               for left, right in zip(result, result[1:]))
    assert result[0]["content"] + "".join(item["content"][10:] for item in result[1:]) == text


def test_invalid_heading_relationship_is_rejected():
    document = {"tree": [
        {"id": "h1", "type": "heading", "text": "标题", "parent": "h1"},
        {"id": "p1", "type": "paragraph", "text": "内容", "parent": "h1"},
    ]}
    with pytest.raises(ValueError, match="标题层级"):
        list(chunks(document))
