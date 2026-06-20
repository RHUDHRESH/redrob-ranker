#!/usr/bin/env python3
"""
Redrob Ranker — retrieve-then-rerank pipeline.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from ranker.fuse import rank_top_n, score_shortlist
from ranker.io_stream import build_query_text, infer_current_year
from ranker.lexical import load_shortlist, recall_candidates

RANDOM_SEED = 42


def write_submission(rows, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for sc, rank in rows:
            reasoning = sc.reasoning.replace("\n", " ").replace("\r", " ")
            writer.writerow([sc.candidate_id, rank, sc.final_score, reasoning])


def run_pipeline(
    candidates_path: str,
    out_path: str,
    jd_path: str | None = None,
    recall_k: int = 1500,
    top_n: int = 100,
    *,
    enable_traps: bool = True,
    enable_trust: bool = True,
    enable_expansion: bool = True,
    verbose: bool = False,
) -> list:
    t0 = time.time()
    query = build_query_text(jd_path) if enable_expansion else Path(jd_path or "job_description.txt").read_text(encoding="utf-8")
    current_year = infer_current_year(candidates_path)

    if verbose:
        print(f"Query expanded: {enable_expansion}", flush=True)
        print(f"CURRENT_YEAR={current_year}", flush=True)

    recall_results, _ = recall_candidates(candidates_path, query, k=recall_k)
    lexical_scores = {r.candidate_id: r.lexical_score for r in recall_results}
    shortlist_ids = set(lexical_scores)

    if verbose:
        print(f"Recall: {len(recall_results)} candidates in {time.time()-t0:.1f}s", flush=True)

    candidates = load_shortlist(candidates_path, shortlist_ids)
    scored = score_shortlist(
        candidates,
        lexical_scores,
        current_year,
        enable_traps=enable_traps,
        enable_trust=enable_trust,
    )
    top = rank_top_n(scored, n=top_n)

    rows = [(sc, i + 1) for i, sc in enumerate(top)]
    write_submission(rows, Path(out_path))

    elapsed = time.time() - t0
    print(f"Wrote {len(rows)} rows -> {out_path}")
    if len(rows) < top_n:
        print(f"[warn] only {len(rows)} trap-clean candidates available (requested {top_n})")
    print(f"Trap-flagged candidates excluded from output")
    print(f"Total time: {elapsed:.1f}s")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob retrieve-then-rerank ranker")
    parser.add_argument("--candidates", required=True, help="candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="output CSV path")
    parser.add_argument("--jd", default=None, help="job description text file")
    parser.add_argument("--recall-k", type=int, default=1500)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not Path(args.candidates).exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        args.candidates,
        args.out,
        jd_path=args.jd,
        recall_k=args.recall_k,
        top_n=args.top,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()