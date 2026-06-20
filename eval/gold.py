#!/usr/bin/env python3
"""Offline gold-set evaluation on sample_candidates.json."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranker.fuse import rank_top_n, score_shortlist
from ranker.io_stream import build_query_text, infer_current_year
from ranker.lexical import recall_candidates

# Hand-labeled relevance (3=strong, 2=good, 1=weak, 0=irrelevant) vs Senior AI Engineer JD
GOLD_LABELS: dict[str, int] = {
    "CAND_0000001": 2,  # backend/data hybrid, some ML skills, transitioning
    "CAND_0000002": 0,  # operations/marketing, no ML depth
    "CAND_0000003": 0,  # customer support, 1yr exp
    "CAND_0000004": 0,  # marketing manager
    "CAND_0000005": 0,  # accountant/marketing mismatch
    "CAND_0000006": 3,  # strong ML engineer profile (if present)
    "CAND_0000007": 2,
    "CAND_0000008": 1,
    "CAND_0000009": 0,
    "CAND_0000010": 1,
}


def _auto_label(candidate: dict) -> int:
    """Heuristic fallback for unlabeled sample rows."""
    from ranker.features import compute_features

    f = compute_features(candidate)
    score = f.structured
    title = (candidate.get("profile") or {}).get("current_title", "").lower()
    if score > 0.55 and any(t in title for t in ("ml", "ai", "machine learning", "data scientist")):
        return 3
    if score > 0.4:
        return 2
    if score > 0.25:
        return 1
    return 0


def dcg(relevances: list[float], k: int) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    gains = [float(labels.get(cid, 0)) for cid in ranked_ids[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = dcg([float(x) for x in ideal], k)
    if idcg == 0:
        return 0.0
    return dcg(gains, k) / idcg


def precision_at_k(ranked_ids: list[str], labels: dict[str, int], k: int, threshold: int = 2) -> float:
    hits = sum(1 for cid in ranked_ids[:k] if labels.get(cid, 0) >= threshold)
    return hits / k


def evaluate(sample_path: Path, k: int = 50) -> dict[str, float]:
    query = build_query_text(ROOT / "job_description.txt")
    current_year = infer_current_year(sample_path)
    recall, _ = recall_candidates(str(sample_path), query, k=len(json.load(open(sample_path))))
    lexical = {r.candidate_id: r.lexical_score for r in recall}

    import json as _json
    candidates = {c["candidate_id"]: c for c in _json.load(open(sample_path, encoding="utf-8"))}
    scored = score_shortlist(candidates, lexical, current_year)
    top = rank_top_n(scored, n=min(k, len(candidates)))
    ranked_ids = [sc.candidate_id for sc in top]

    labels = dict(GOLD_LABELS)
    for cid, cand in candidates.items():
        if cid not in labels:
            labels[cid] = _auto_label(cand)

    return {
        f"nDCG@{k}": ndcg_at_k(ranked_ids, labels, k),
        f"precision@{k}": precision_at_k(ranked_ids, labels, k),
    }


def main():
    sample = ROOT / "sample_candidates.json"
    if not sample.exists():
        print(f"Missing {sample}")
        sys.exit(1)
    metrics = evaluate(sample)
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")


if __name__ == "__main__":
    main()