"""BM25 lexical recall over the full candidate pool."""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ranker.io_stream import build_candidate_text, iter_candidates
from ranker.thesaurus import tokenize

K1 = 1.5
B = 0.75
RECALL_K = 1500


@dataclass
class RecallResult:
    candidate_id: str
    lexical_score: float


class BM25Index:
    """Two-pass streaming BM25: pass 1 collects df/lengths, pass 2 scores."""

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.doc_lens: list[int] = []
        self.candidate_ids: list[str] = []
        self.df: Counter[str] = Counter()

    def fit_pass1(self, candidates_path: str) -> None:
        for cand in iter_candidates(candidates_path):
            text = build_candidate_text(cand)
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens) or 1)
            self.candidate_ids.append(cand["candidate_id"])
            seen = set(tokens)
            for t in seen:
                self.df[t] += 1
            self.N += 1
        total = sum(self.doc_lens) or 1
        self.avgdl = total / max(self.N, 1)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score_doc(self, tokens: list[str], query_tf: Counter[str]) -> float:
        if not tokens:
            return 0.0
        dl = len(tokens)
        doc_tf = Counter(tokens)
        score = 0.0
        for term, qf in query_tf.items():
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom * qf
        return score

    def recall_top_k(
        self,
        candidates_path: str,
        query_text: str,
        k: int = RECALL_K,
    ) -> list[RecallResult]:
        query_tf = Counter(tokenize(query_text))
        heap: list[tuple[float, str]] = []

        idx = 0
        for cand in iter_candidates(candidates_path):
            tokens = tokenize(build_candidate_text(cand))
            score = self.score_doc(tokens, query_tf)
            cid = self.candidate_ids[idx]
            idx += 1
            if len(heap) < k:
                heapq.heappush(heap, (score, cid))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, cid))

        results = sorted(heap, key=lambda x: (-x[0], x[1]))
        return [RecallResult(candidate_id=cid, lexical_score=sc) for sc, cid in results]


def load_shortlist(
    candidates_path: str,
    shortlist_ids: set[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cand in iter_candidates(candidates_path):
        cid = cand["candidate_id"]
        if cid in shortlist_ids:
            out[cid] = cand
            if len(out) == len(shortlist_ids):
                break
    return out


def recall_candidates(
    candidates_path: str,
    query_text: str,
    k: int = RECALL_K,
) -> tuple[list[RecallResult], BM25Index]:
    index = BM25Index()
    index.fit_pass1(candidates_path)
    results = index.recall_top_k(candidates_path, query_text, k=k)
    return results, index