"""Structured features for shortlist reranking (normalized to [0, 1])."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from ranker.thesaurus import (
    EXP_IDEAL_HI,
    EXP_IDEAL_LO,
    EXP_SOFT_HI,
    EXP_SOFT_LO,
    REQUIRED_SKILLS,
    TITLE_TERMS,
    TIER_SCORES,
    skill_matches_required,
)
from ranker.traps import REFERENCE_DATE, parse_date, sigmoid

STRUCT_WEIGHTS = {
    "skill_match": 0.30,
    "title_traj": 0.25,
    "exp_fit": 0.15,
    "edu_tier": 0.05,
    "signal_quality": 0.25,
}


@dataclass
class FeatureBreakdown:
    skill_match: float
    title_traj: float
    exp_fit: float
    edu_tier: float
    signal_quality: float
    skill_trust: float
    matched_skills: list[str]
    top_skills: list[str]
    structured: float

    def as_dict(self) -> dict[str, float]:
        return {
            "skill_match": self.skill_match,
            "title_traj": self.title_traj,
            "exp_fit": self.exp_fit,
            "edu_tier": self.edu_tier,
            "signal_quality": self.signal_quality,
            "skill_trust": self.skill_trust,
            "structured": self.structured,
        }


def _trust_for_skill(skill: dict[str, Any], assessments: dict[str, float]) -> float:
    endorsements = float(skill.get("endorsements") or 0)
    duration = float(skill.get("duration_months") or 0)
    name = skill.get("name") or ""
    assess = 0.0
    for key, val in assessments.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            assess = float(val) / 100.0
            break
    raw = 0.35 * math.log1p(endorsements) + 0.35 * math.log1p(duration) + 0.30 * assess
    return sigmoid(raw - 0.5)


def skill_match_with_trust(candidate: dict[str, Any]) -> tuple[float, float, list[str], list[str]]:
    sig = candidate.get("redrob_signals") or {}
    assessments = sig.get("skill_assessment_scores") or {}
    matched: list[tuple[str, float]] = []

    for skill in candidate.get("skills") or []:
        concept = skill_matches_required(skill.get("name") or "")
        if concept:
            trust = _trust_for_skill(skill, assessments)
            matched.append((concept, trust))

    if not REQUIRED_SKILLS:
        return 0.0, 0.0, [], []

    by_concept: dict[str, float] = {}
    for concept, trust in matched:
        by_concept[concept] = max(by_concept.get(concept, 0.0), trust)

    coverage = len(by_concept) / len(REQUIRED_SKILLS)
    if not by_concept:
        return 0.0, 0.0, [], []

    trust_vals = list(by_concept.values())
    skill_match = sum(trust_vals) / len(REQUIRED_SKILLS)
    skill_trust = sum(trust_vals) / len(trust_vals)
    concepts = sorted(by_concept, key=lambda c: -by_concept[c])
    top = [s.get("name", "") for s in candidate.get("skills") or [] if skill_matches_required(s.get("name", ""))][:3]
    return min(1.0, skill_match), skill_trust, concepts, top


def title_trajectory_match(candidate: dict[str, Any]) -> float:
    career = candidate.get("career_history") or []
    profile = candidate.get("profile") or {}
    score = 0.0
    total_months = 0

    for job in career:
        title = (job.get("title") or "").lower()
        months = int(job.get("duration_months") or 0)
        title_hit = 0.0
        for term in TITLE_TERMS:
            if term in title:
                title_hit = max(title_hit, 1.0 if "senior" in term or "ml" in term or "ai" in term else 0.85)
        if title_hit:
            score += title_hit * months
            total_months += months

    headline = (profile.get("headline") or "").lower()
    for term in TITLE_TERMS:
        if term in headline:
            score += 0.25
            break

    if total_months == 0:
        return min(1.0, score)
    return min(1.0, score / (total_months + 12))


def experience_fit(candidate: dict[str, Any]) -> float:
    yoe = float((candidate.get("profile") or {}).get("years_of_experience") or 0)
    if EXP_IDEAL_LO <= yoe <= EXP_IDEAL_HI:
        return 1.0
    if EXP_SOFT_LO <= yoe < EXP_IDEAL_LO:
        return 0.6 + 0.4 * (yoe - EXP_SOFT_LO) / (EXP_IDEAL_LO - EXP_SOFT_LO)
    if EXP_IDEAL_HI < yoe <= EXP_SOFT_HI:
        return 0.6 + 0.4 * (EXP_SOFT_HI - yoe) / (EXP_SOFT_HI - EXP_IDEAL_HI)
    if yoe < EXP_SOFT_LO:
        return max(0.1, yoe / EXP_SOFT_LO * 0.5)
    return max(0.2, 1.0 - (yoe - EXP_SOFT_HI) * 0.15)


def education_tier_score(candidate: dict[str, Any]) -> float:
    tiers = [
        TIER_SCORES.get((e.get("tier") or "").lower(), 0.4)
        for e in candidate.get("education") or []
    ]
    return max(tiers) if tiers else 0.4


def signal_quality(candidate: dict[str, Any]) -> float:
    sig = candidate.get("redrob_signals") or {}
    parts: list[float] = []

    rr = float(sig.get("recruiter_response_rate") or 0)
    parts.append(min(1.0, rr / 0.5))

    icr = float(sig.get("interview_completion_rate") or 0)
    parts.append(icr)

    if sig.get("open_to_work_flag"):
        parts.append(1.0)
    else:
        parts.append(0.5)

    last = parse_date(sig.get("last_active_date"))
    if last:
        days = (REFERENCE_DATE - last).days
        recency = math.exp(-days / 120.0)
        parts.append(recency)

    parts.append(min(1.0, math.log1p(float(sig.get("connection_count") or 0)) / 6))
    parts.append(min(1.0, math.log1p(float(sig.get("endorsements_received") or 0)) / 4))
    parts.append(min(1.0, math.log1p(float(sig.get("profile_views_received_30d") or 0)) / 3))

    gh = sig.get("github_activity_score")
    if isinstance(gh, (int, float)) and gh >= 0:
        parts.append(min(1.0, float(gh) / 10.0))

    return sum(parts) / len(parts) if parts else 0.0


def compute_features(candidate: dict[str, Any]) -> FeatureBreakdown:
    skill_m, skill_trust, matched, top = skill_match_with_trust(candidate)
    title = title_trajectory_match(candidate)
    exp = experience_fit(candidate)
    edu = education_tier_score(candidate)
    signal = signal_quality(candidate)

    structured = (
        STRUCT_WEIGHTS["skill_match"] * skill_m
        + STRUCT_WEIGHTS["title_traj"] * title
        + STRUCT_WEIGHTS["exp_fit"] * exp
        + STRUCT_WEIGHTS["edu_tier"] * edu
        + STRUCT_WEIGHTS["signal_quality"] * signal
    )

    return FeatureBreakdown(
        skill_match=skill_m,
        title_traj=title,
        exp_fit=exp,
        edu_tier=edu,
        signal_quality=signal,
        skill_trust=skill_trust,
        matched_skills=matched,
        top_skills=top,
        structured=structured,
    )