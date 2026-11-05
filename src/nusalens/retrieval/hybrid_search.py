import time
from dataclasses import dataclass

import psycopg
from sentence_transformers import CrossEncoder

from nusalens.config import settings
from nusalens.retrieval.query_expand import expand_query, load_acronyms
from nusalens.retrieval.search_dense import DenseSearcher
from nusalens.retrieval.search_sparse import SparseSearcher


@dataclass
class FusedHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


def rrf_fusion(dense_hits, sparse_hits, k=60):
    """Combine ranked lists using RRF: score(d) = sum of 1/(k + rank_i(d))."""
    scores = {}
    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    # Also preserve text and doc_id from one of the results
    text_map = {h.chunk_id: h.text for h in dense_hits + sparse_hits}
    doc_map = {h.chunk_id: h.doc_id for h in dense_hits + sparse_hits}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [FusedHit(cid, doc_map[cid], text_map[cid], score) for cid, score in ranked]


class HybridSearcher:
    """Combined dense + sparse + RRF + cross-encoder reranking."""
    def __init__(self, dsn, device="cpu", dense_model_name=None, dense_embedding_column="embedding", pool=None):
        kwargs = {"embedding_column": dense_embedding_column}
        if dense_model_name is not None:
            kwargs["model_name"] = dense_model_name
        self.dense = DenseSearcher(dsn, device=device, pool=pool, **kwargs)
        self.sparse = SparseSearcher(dsn, pool=pool)
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)
        self.acronyms = load_acronyms()

    def search_rrf(self, query, k=10, k_base=50, rrf_k=60):
        """RRF fusion of dense and sparse top 50."""
        dense_hits = self.dense.search(query, k=k_base)
        sparse_hits = self.sparse.search(query, k=k_base)
        fused = rrf_fusion(dense_hits, sparse_hits, k=rrf_k)
        return fused[:k]

    def search_rrf_rerank(self, query, k=8, k_base=50, k_rerank=15, rrf_k=60):
        """RRF + cross-encoder reranking over the top k_rerank candidates."""
        dense_hits = self.dense.search(query, k=k_base)
        sparse_hits = self.sparse.search(query, k=k_base)
        fused = rrf_fusion(dense_hits, sparse_hits, k=rrf_k)
        return [h for h, _ in self._rerank_scored(query, fused, k=k, k_rerank=k_rerank)]

    def search_rrf_rerank_exp(self, query, k=8, k_base=50, k_rerank=15, rrf_k=60):
        """Acronym-expanded query, then RRF + cross-encoder reranking."""
        hits_scored = self.search_rrf_rerank_exp_scored(query, k=k, k_base=k_base, k_rerank=k_rerank, rrf_k=rrf_k)
        return [h for h, _ in hits_scored]

    def search_rrf_rerank_exp_scored(self, query, k=8, k_base=50, k_rerank=15, rrf_k=60):
        """Same as search_rrf_rerank_exp but keeps the cross-encoder score per hit, for the pipeline's abstain gate."""
        expanded = expand_query(query, self.acronyms)
        dense_hits = self.dense.search(expanded, k=k_base)
        sparse_hits = self.sparse.search(expanded, k=k_base)
        fused = rrf_fusion(dense_hits, sparse_hits, k=rrf_k)
        return self._rerank_scored(expanded, fused, k=k, k_rerank=k_rerank)

    def search_rrf_rerank_exp_timed(self, query, k=8, k_base=50, k_rerank=15, rrf_k=60):
        """Same as search_rrf_rerank_exp_scored but also splits out (retrieve_ms, rerank_ms)."""
        t0 = time.perf_counter()
        expanded = expand_query(query, self.acronyms)
        dense_hits = self.dense.search(expanded, k=k_base)
        sparse_hits = self.sparse.search(expanded, k=k_base)
        fused = rrf_fusion(dense_hits, sparse_hits, k=rrf_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        scored = self._rerank_scored(expanded, fused, k=k, k_rerank=k_rerank)
        rerank_ms = (time.perf_counter() - t1) * 1000
        return scored, retrieve_ms, rerank_ms

    def _rerank_scored(self, query, fused, k, k_rerank):
        to_rerank = fused[:k_rerank]
        if not to_rerank:
            return to_rerank[:k]
        pairs = [[query, h.text] for h in to_rerank]
        scores = self.reranker.predict(pairs)
        reranked = sorted(
            zip(to_rerank, scores),
            key=lambda kv: kv[1],
            reverse=True
        )
        return [(h, float(s)) for h, s in reranked[:k]]
