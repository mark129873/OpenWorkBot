import json
import uuid

import pytest

from app.documents import read_document, remove_structures, save_structure


def test_structures_are_saved_by_document_and_version(tmp_path):
    document_id = uuid.uuid4().hex
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    data = {"doc_id": document_id, "source": {"filename": "资料.pdf"}, "tree": []}

    first_path = save_structure(tmp_path, document_id, first, data)
    second_path = save_structure(tmp_path, document_id, second, data)
    assert first_path != second_path
    assert json.loads(first_path.read_text(encoding="utf-8")) == data
    assert json.loads(second_path.read_text(encoding="utf-8")) == data

    other = tmp_path / "unrelated.json"
    other.write_text("keep", encoding="utf-8")
    remove_structures(tmp_path, document_id)
    assert not first_path.exists() and not second_path.exists()
    assert other.read_text(encoding="utf-8") == "keep"


def test_structure_write_failure_does_not_leave_partial_file(tmp_path, monkeypatch):
    document_id, version = uuid.uuid4().hex, uuid.uuid4().hex

    def fail(*_args):
        raise OSError("injected failure")

    monkeypatch.setattr("app.documents.os.replace", fail)
    with pytest.raises(OSError):
        save_structure(tmp_path, document_id, version, {"tree": []})
    assert list((tmp_path / "documents" / document_id).iterdir()) == []
    with pytest.raises(ValueError, match="文档 ID"):
        remove_structures(tmp_path, "../outside")


def test_read_document_keeps_headings_tables_and_page_order(tmp_path):
    document_id, version = uuid.uuid4().hex, uuid.uuid4().hex
    document = {"doc_id": document_id, "source": {"filename": "资料.pdf"}, "tree": [
        {"type": "heading", "text": "材料", "page": 1},
        {"type": "paragraph", "text": "营业执照", "page": 1},
        {"type": "heading", "text": "额度", "page": 2},
        {"type": "table", "cells": [["单笔", "300万元"]], "page": 2},
    ]}
    save_structure(tmp_path, document_id, version, document)
    assert read_document(tmp_path, document_id, version) == "[第 1 页]\n材料\n营业执照\n[第 2 页]\n额度\n单笔\t300万元"
    with pytest.raises(ValueError, match="文档版本"):
        read_document(tmp_path, document_id, "../outside")
