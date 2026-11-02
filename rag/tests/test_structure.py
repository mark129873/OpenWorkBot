from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from app.chunking import chunks
from app.config import Settings
from app.knowledge import Knowledge
from app.structure import clean_document, parse_document


def _pdf(path, *, blank=False):
    writer = PdfWriter()
    writer.add_metadata({"/Title": "Staff Policy"})
    font = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    }))
    bold = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
    }))
    image = DecodedStreamObject()
    image.set_data(b"\xff\x00\x00")
    image.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
                  NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1),
                  NameObject("/ColorSpace"): NameObject("/DeviceRGB"), NameObject("/BitsPerComponent"): NumberObject(8)})
    image_ref = writer._add_object(image)
    for number in (1, 2):
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font, NameObject("/F2"): bold}),
            NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): image_ref}),
        })
        text = (
            "BT /F1 8 Tf 70 785 Td (Policy header) Tj ET\n"
            + ("BT /F2 18 Tf 70 740 Td (Staff Policy) Tj ET\n" if number == 1 else "")
            + f"BT /F2 12 Tf 70 690 Td ({number}. Leave) Tj ET\n"
            + "BT /F1 10 Tf 70 670 Td (Leave is limited to 30 days.) Tj ET\n"
            + ("70 620 m 330 620 l S 70 590 m 330 590 l S 70 560 m 330 560 l S\n"
               "70 560 m 70 620 l S 200 560 m 200 620 l S 330 560 m 330 620 l S\n"
               "BT /F1 10 Tf 80 600 Td (Type) Tj ET BT /F1 10 Tf 210 600 Td (Days) Tj ET\n"
               "BT /F1 10 Tf 80 570 Td (Sick) Tj ET BT /F1 10 Tf 210 570 Td (30) Tj ET\n"
               "q 50 0 0 50 70 450 cm /Im0 Do Q\n" if number == 1 else "")
            + "BT /F1 8 Tf 70 20 Td (Policy warning) Tj ET\n"
        )
        stream = DecodedStreamObject()
        stream.set_data(text.encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if blank:
        writer.add_blank_page(width=600, height=800)
    writer.write(path)


def test_pdf_structure_preserves_title_table_image_and_order(tmp_path):
    path = tmp_path / "policy.pdf"
    _pdf(path)
    document, edges = parse_document(path, ".pdf", path.name, "doc-id")
    assert document["source"] == {"filename": "policy.pdf", "parser": "pdf", "pages": 2}
    nodes = document["tree"]
    assert [node["order"] for node in nodes] == list(range(1, len(nodes) + 1))
    root = next(node for node in nodes if node["type"] == "heading" and node["level"] == 1)
    assert root["text"] == "Staff Policy" and root["page"] == 1 and len(root["bbox"]) == 4
    section = next(node for node in nodes if node.get("text") == "1. Leave")
    assert section["parent"] == root["id"] and section["level"] == 2
    table = next(node for node in nodes if node["type"] == "table")
    assert table["parent"] == section["id"]
    assert table["cells"] == [["Type", "Days"], ["Sick", "30"]]
    figure = next(node for node in nodes if node["type"] == "figure")
    assert figure["parent"] == section["id"] and figure["page"] == 1
    assert figure["ocr_text"] is None and figure["description"] is None

    clean_document(document, edges)
    text = [node.get("text") for node in document["tree"]]
    assert text.count("Policy header") == 1
    assert text.count("Policy warning") == 1
    result = list(chunks(document))
    assert any(item["heading_path"] == ["Staff Policy", "1. Leave"]
               and "Leave is limited to 30 days." in item["content"] for item in result)


def test_markdown_keeps_heading_hierarchy_and_fenced_text(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Handbook\n\n## Leave\n\nDetails.\n\n```\n# Not a heading\n```\n", encoding="utf-8")
    document, edges = parse_document(path, ".md", path.name, "doc-id")
    assert document["source"]["pages"] is None
    assert [node["text"] for node in document["tree"] if node["type"] == "heading"] == ["Handbook", "Leave"]
    clean_document(document, edges)
    result = list(chunks(document))
    assert len(result) == 1
    assert result[0]["heading_path"] == ["Handbook", "Leave"]
    assert "# Not a heading" in result[0]["content"]
    assert result[0]["pages"] == []


async def test_partly_empty_pdf_ingests_without_page_warning(tmp_path):
    path = tmp_path / "policy.pdf"
    _pdf(path, blank=True)
    config = Settings(tmp_path, tmp_path / "data", "", "", "", "", "", "", "", "", "", "")

    class Models:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Vectors:
        async def add(self, *_args):
            pass

        async def delete(self, *_args):
            pass

    service = Knowledge(config, models=Models(), vectors=Vectors())
    result = await service.ingest("policy.pdf")
    assert result["status"] == "success" and result["warning"] == ""
    assert result["chunks"] > 0
