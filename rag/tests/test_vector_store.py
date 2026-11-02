from app.catalog import Catalog
from app.config import Settings
from app.models import Models
from app.vector_store import VectorStore


async def test_new_index_stores_source_and_search_text_separately(tmp_path):
    model = tmp_path / "embedding.gguf"
    model.write_bytes(b"model")
    config = Settings(tmp_path, tmp_path, "http://milvus.invalid", "secret", "test_collection",
                      "http://embedding.invalid", str(model), "", "", "", "", "")
    catalog = Catalog(tmp_path)

    class Schema:
        def __init__(self):
            self.fields = []
            self.functions = []

        def add_field(self, name, datatype, **kwargs):
            self.fields.append((name, datatype, kwargs))

        def add_function(self, function):
            self.functions.append(function.to_dict())

    class Indexes:
        def add_index(self, **_kwargs):
            pass

    class Client:
        def __init__(self):
            self.exists = False
            self.schema = Schema()
            self.rows = []

        def has_collection(self, _name):
            return self.exists

        def create_schema(self, **_kwargs):
            return self.schema

        def prepare_index_params(self):
            return Indexes()

        def create_collection(self, *_args, **_kwargs):
            self.exists = True

        def insert(self, _name, rows):
            self.rows.extend(rows)

        def hybrid_search(self, _name, **kwargs):
            assert {"content", "search_text", "metadata"} <= set(kwargs["output_fields"])
            assert [request.limit for request in kwargs["reqs"]] == [20, 20]
            assert kwargs["limit"] == 10 and kwargs["ranker"].dict()["params"]["k"] == 60
            row = self.rows[0]
            return [[{"chunk_id": row["chunk_id"], "entity": {
                key: row[key] for key in kwargs["output_fields"]}, "distance": 0.8}]]

    store = VectorStore(config, catalog)
    store.client = Client()
    chunk = {"content": "原文", "search_text": "文档标题\n章节标题\n原文",
             "heading_path": ["文档标题", "章节标题"], "pages": [2]}
    await store.add("version", "doc.pdf", 0, [chunk], [[1.0, 0.0]])

    assert store.client.schema.functions[0]["input_field_names"] == ["search_text"]
    assert store.client.rows[0]["content"] == "原文"
    assert store.client.rows[0]["search_text"] == chunk["search_text"]
    result = await store.search("问题", [1.0, 0.0], ["version"])
    assert result[0]["content"] == "原文"
    assert result[0]["search_text"] == chunk["search_text"]
    assert result[0]["heading_path"] == chunk["heading_path"]
    assert result[0]["pages"] == [2]


async def test_rerank_uses_title_but_returns_original_content(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    request = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.8}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, _url, json):
            request.update(json)
            return Response()

    monkeypatch.setattr("app.models.httpx.AsyncClient", Client)
    config = Settings(tmp_path, tmp_path, "", "", "", "", "", "http://rerank", "rerank", "", "", "")
    result = await Models(config).rerank("问题", [{"_rank_index": 0, "content": "原文",
                                                    "search_text": "标题\n原文", "pages": [2]}])
    assert request["documents"] == ["标题\n原文"]
    assert result == [{"_rank_index": 0, "content": "原文", "pages": [2]}]


async def test_rerank_returns_all_candidates_before_document_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": index, "relevance_score": index} for index in range(12)]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, _url, json):
            assert json["top_n"] == 12
            return Response()

    monkeypatch.setattr("app.models.httpx.AsyncClient", Client)
    config = Settings(tmp_path, tmp_path, "", "", "", "", "", "http://rerank", "rerank", "", "", "")
    items = [{"_rank_index": index, "content": str(index), "search_text": str(index)} for index in range(12)]
    ranked = await Models(config).rerank("问题", items)
    assert [item["content"] for item in ranked] == [str(index) for index in reversed(range(12))]
