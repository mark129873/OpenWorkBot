import math

import pytest

from evaluations.utils.benchmark import retrieval_scores


def test_document_retrieval_scores_use_ranked_chunks_and_unique_documents():
    chunks = [{"filename": name} for name in ("other.pdf", "A02_file.pdf", "A02_file.pdf", "A01_file.pdf")]
    recall, ndcg = retrieval_scores(chunks, ["A01 file.pdf", "A02 file.pdf"])
    assert recall == 1
    assert ndcg == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))


def test_top_five_is_applied_after_document_deduplication():
    chunks = [{"filename": "A01_file.pdf"}] * 10 + [
        {"filename": f"A0{number}_file.pdf"} for number in range(2, 7)]
    recall, ndcg = retrieval_scores(chunks, ["A01 file.pdf", "A02 file.pdf", "A06 file.pdf"])
    assert recall == pytest.approx(2 / 3)
    assert ndcg == pytest.approx((1 + 1 / math.log2(3)) / (1 + 1 / math.log2(3) + 1 / math.log2(4)))
    assert retrieval_scores([], ["A01 file.pdf"]) == (0, 0)


@pytest.mark.parametrize("evidence,expected", [([], []), ([{}], ["A01 file.pdf"]), ([], ["invalid.pdf"])])
def test_unknown_labels_or_missing_evidence_metadata_are_not_scored(evidence, expected):
    with pytest.raises(ValueError):
        retrieval_scores(evidence, expected)
