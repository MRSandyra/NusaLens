from nusalens.eval.retrieval_harness import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k_hit_within_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a", "b", "c"], {"d"}, k=3) == 0.0


def test_recall_at_k_no_gold_ids_is_nan():
    import math
    assert math.isnan(recall_at_k(["a"], set(), k=3))


def test_mrr_at_k_first_position():
    assert mrr_at_k(["a", "b"], {"a"}, k=10) == 1.0


def test_mrr_at_k_second_position():
    assert mrr_at_k(["a", "b"], {"b"}, k=10) == 0.5


def test_mrr_at_k_no_hit_is_zero():
    assert mrr_at_k(["a", "b"], {"c"}, k=10) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_at_k_worse_ranking_is_less_than_one():
    perfect = ndcg_at_k(["a", "b"], {"a"}, k=2)
    worse = ndcg_at_k(["b", "a"], {"a"}, k=2)
    assert worse < perfect


def test_ndcg_at_k_no_gold_ids_is_zero():
    assert ndcg_at_k(["a"], set(), k=3) == 0.0
