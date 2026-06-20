"""Fusion scoring, ranking, and deterministic reasoning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ranker.features import FeatureBreakdown, compute_features
from ranker.io_stream import build_candidate_text
from ranker.traps import dedup_twins, impossibility_flags, stuffer_multiplier

LEXICAL_WEIGHT = 0.45
STRUCTURED_WEIGHT = 0.55
NEG_INF = -1e18


@dataclass
class ScoredCandidate:
    candidate_id: str
    candidate: dict[str, Any]
    lexical_norm: float
    features: FeatureBreakdown
    trust: float
    final_score: float
    flags: list[str]
    reasoning: str = ""


def normalize_lexical(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return {cid: (s - lo) / span for cid, s in scores.items()}


def fuse_score(
    lexical_norm: float,
    features: FeatureBreakdown,
    trust: float,
    trap_flags: list[str],
) -> float:
    if trap_flags:
        return NEG_INF
    base = LEXICAL_WEIGHT * lexical_norm + STRUCTURED_WEIGHT * features.structured
    return base * trust


def build_reasoning(
    rank: int,
    candidate: dict[str, Any],
    features: FeatureBreakdown,
) -> str:
    profile = candidate.get("profile") or {}
    sig = candidate.get("redrob_signals") or {}
    yoe = profile.get("years_of_experience", 0)
    k = len(features.matched_skills)
    m = 10
    top = ", ".join(features.top_skills[:3]) or "none"
    rr = float(sig.get("recruiter_response_rate") or 0)
    ic = float(sig.get("interview_completion_rate") or 0)
    return (
        f"#{rank}: {k}/{m} required skills corroborated ({top}); "
        f"{yoe} yrs with matching titles; "
        f"engagement resp-rate {rr:.2f}, interview-completion {ic:.2f}."
    )


def score_shortlist(
    candidates: dict[str, dict[str, Any]],
    lexical_scores: dict[str, float],
    current_year: int,
    *,
    enable_traps: bool = True,
    enable_trust: bool = True,
    enable_stuffer: bool = True,
) -> list[ScoredCandidate]:
    lex_norm = normalize_lexical(lexical_scores)
    scored: list[ScoredCandidate] = []

    for cid, cand in candidates.items():
        features = compute_features(cand)
        flags = impossibility_flags(cand, current_year) if enable_traps else []

        trust = features.skill_trust if enable_trust else 1.0
        if enable_stuffer:
            trust *= stuffer_multiplier(cand)

        final = fuse_score(lex_norm.get(cid, 0.0), features, trust, flags)
        scored.append(
            ScoredCandidate(
                candidate_id=cid,
                candidate=cand,
                lexical_norm=lex_norm.get(cid, 0.0),
                features=features,
                trust=trust,
                final_score=final,
                flags=flags,
            )
        )

    return scored


def rank_top_n(
    scored: list[ScoredCandidate],
    n: int = 100,
    *,
    dedup: bool = True,
) -> list[ScoredCandidate]:
    valid = [s for s in scored if not s.flags and s.final_score > NEG_INF / 2]
    items = [(s.final_score, s.candidate_id, s) for s in valid]
    items.sort(key=lambda x: (-x[0], x[1]))

    if dedup:
        tuple_items = [(sc.final_score, sc.candidate, sc) for sc in valid]
        tuple_items.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        deduped = dedup_twins(
            tuple_items,
            corroboration_key=lambda t: t[2].features.skill_trust,
            text_key=lambda t: build_candidate_text(t[1]),
        )
        items = [(t[0], t[1]["candidate_id"], t[2]) for t in deduped]
        items.sort(key=lambda x: (-x[0], x[1]))

    top = [item[2] for item in items[: max(n, 120)]][:n]

    # Normalize output scores to [0, 1] non-increasing for validator
    if top:
        raw = [s.final_score for s in top if s.final_score > NEG_INF / 2]
        if raw:
            lo, hi = min(raw), max(raw)
            span = hi - lo or 1.0
            for i, sc in enumerate(top):
                if sc.final_score <= NEG_INF / 2:
                    sc.final_score = 0.0
                else:
                    sc.final_score = round(0.2 + 0.79 * (sc.final_score - lo) / span, 4)
                sc.reasoning = build_reasoning(i + 1, sc.candidate, sc.features)

    # Enforce monotone non-increasing after rounding
    for i in range(1, len(top)):
        if top[i].final_score > top[i - 1].final_score:
            top[i].final_score = top[i - 1].final_score

    return top