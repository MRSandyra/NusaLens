from nusalens.retrieval.search_dense import DenseHit
from nusalens.retrieval.search_sparse import SparseHit
from nusalens.retrieval.hybrid_search import rrf_fusion


def test_rrf_scores_match_formula():
    dense = [DenseHit("c1", "d1", "text1", 0.9), DenseHit("c2", "d1", "text2", 0.8)]
    sparse = [SparseHit("c2", "d1", "text2", 5.0), SparseHit("c1", "d1", "text1", 3.0)]
    fused = rrf_fusion(dense, sparse, k=60)
    scores = {h.chunk_id: h.score for h in fused}
    # c1: dense rank 1, sparse rank 2 -> 1/61 + 1/62
    assert scores["c1"] == 1 / 61 + 1 / 62
    # c2: dense rank 2, sparse rank 1 -> 1/62 + 1/61
    assert scores["c2"] == 1 / 62 + 1 / 61


def test_rrf_result_sorted_descending_by_score():
    dense = [DenseHit("c1", "d1", "t1", 0.9), DenseHit("c2", "d1", "t2", 0.8), DenseHit("c3", "d1", "t3", 0.7)]
    sparse = [DenseHit("c1", "d1", "t1", 5.0)]
    fused = rrf_fusion(dense, sparse, k=60)
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0].chunk_id == "c1"  # appears in both lists, highest combined score


def test_rrf_hit_only_in_one_list_still_included():
    dense = [DenseHit("c1", "d1", "t1", 0.9)]
    sparse = [SparseHit("c2", "d2", "t2", 5.0)]
    fused = rrf_fusion(dense, sparse, k=60)
    assert {h.chunk_id for h in fused} == {"c1", "c2"}


def test_rrf_preserves_text_and_doc_id():
    dense = [DenseHit("c1", "doc-x", "hello world", 0.9)]
    fused = rrf_fusion(dense, [], k=60)
    assert fused[0].text == "hello world"
    assert fused[0].doc_id == "doc-x"


def test_rrf_empty_inputs_return_empty():
    assert rrf_fusion([], [], k=60) == []


def test_rrf_lower_k_increases_separation_between_ranks():
    dense = [DenseHit("c1", "d1", "t1", 0.9), DenseHit("c2", "d1", "t2", 0.8)]
    fused_low_k = rrf_fusion(dense, [], k=1)
    fused_high_k = rrf_fusion(dense, [], k=1000)
    gap_low_k = fused_low_k[0].score - fused_low_k[1].score
    gap_high_k = fused_high_k[0].score - fused_high_k[1].score
    assert gap_low_k > gap_high_k
