#!/usr/bin/env python3
"""Ablation table: full vs -traps vs -trust vs -expansion."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gold import evaluate, ndcg_at_k
from eval.synth_honeypots import build_honeypots
from ranker.fuse import NEG_INF, rank_top_n, score_shortlist  # noqa: F401
from ranker.io_stream import build_query_text, infer_current_year
from ranker.lexical import recall_candidates
from ranker.traps import impossibility_flags

REFERENCE_YEAR = 2026


def honeypot_leak_at_100(
    candidates: dict,
    lexical: dict,
    current_year: int,
    honeypot_ids: set[str],
    **kwargs,
) -> int:
    scored = score_shortlist(candidates, lexical, current_year, **kwargs)
    top = rank_top_n(scored, n=100)
    return sum(
        1
        for sc in top
        if sc.candidate_id in honeypot_ids and sc.final_score > NEG_INF / 2
    )


def run_ablation(sample_path: Path) -> None:
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    candidates = {c["candidate_id"]: c for c in data}
    current_year = infer_current_year(sample_path)

    configs = {
        "full": {"enable_traps": True, "enable_trust": True, "enable_expansion": True},
        "-trap-gate": {"enable_traps": False, "enable_trust": True, "enable_expansion": True},
        "-skill-trust": {"enable_traps": True, "enable_trust": False, "enable_expansion": True},
        "-query-expansion": {"enable_traps": True, "enable_trust": True, "enable_expansion": False},
    }

    print(f"{'config':<20} {'nDCG@50':>10} {'leak@100':>10}")
    print("-" * 42)

    for name, cfg in configs.items():
        query = (
            build_query_text(ROOT / "job_description.txt")
            if cfg["enable_expansion"]
            else (ROOT / "job_description.txt").read_text(encoding="utf-8")
        )
        recall, _ = recall_candidates(str(sample_path), query, k=len(data))
        lexical = {r.candidate_id: r.lexical_score for r in recall}

        scored = score_shortlist(
            candidates,
            lexical,
            current_year,
            enable_traps=cfg["enable_traps"],
            enable_trust=cfg["enable_trust"],
        )
        top = rank_top_n(scored, n=min(50, len(data)))
        ranked_ids = [sc.candidate_id for sc in top]

        from eval.gold import GOLD_LABELS, _auto_label

        labels = dict(GOLD_LABELS)
        for cid, cand in candidates.items():
            if cid not in labels:
                labels[cid] = _auto_label(cand)
        ndcg = ndcg_at_k(ranked_ids, labels, 50)

        # Honeypot leak on synthetic set merged into pool
        trap_pool = dict(candidates)
        honeypot_ids: set[str] = set()
        for _, hp in build_honeypots():
            trap_pool[hp["candidate_id"]] = hp
            honeypot_ids.add(hp["candidate_id"])
        leak = honeypot_leak_at_100(
            trap_pool,
            lexical,
            current_year,
            honeypot_ids,
            enable_traps=cfg["enable_traps"],
            enable_trust=cfg["enable_trust"],
        )

        print(f"{name:<20} {ndcg:>10.4f} {leak:>10}")


def main():
    sample = ROOT / "sample_candidates.json"
    if not sample.exists():
        print(f"Missing {sample}")
        sys.exit(1)
    run_ablation(sample)


if __name__ == "__main__":
    main()