"""Trap gate: honeypot validators, keyword-stuffer penalty, twin dedup."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime
from typing import Any

import re

from ranker.io_stream import build_candidate_text
from ranker.thesaurus import tokenize

REFERENCE_DATE = date(2026, 6, 1)
STUFFER_TAU = 4.0

STRONG_OSS_RE = re.compile(
    r"\b(open[- ]source contributor|github\.com/|active github|oss contributor|"
    r"maintainer of|open[- ]source project|github profile)\b",
    re.IGNORECASE,
)


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def months_between(start: date | None, end: date | None) -> int | None:
    if start is None:
        return None
    end = end or REFERENCE_DATE
    return (end.year - start.year) * 12 + (end.month - start.month)


def min_career_start(career: list[dict]) -> date | None:
    dates = [parse_date(j.get("start_date")) for j in career]
    dates = [d for d in dates if d]
    return min(dates) if dates else None


def mentions_oss(text: str) -> bool:
    return bool(STRONG_OSS_RE.search(text or ""))


def impossibility_flags(candidate: dict[str, Any], current_year: int) -> list[str]:
    """F1–F9 deterministic honeypot validators."""
    flags: list[str] = []
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    sig = candidate.get("redrob_signals") or {}
    yoe = float(profile.get("years_of_experience") or 0)

    # F1: YoE vs education timeline (3-year slack for gaps / overlapping degrees)
    grad_years = [e.get("end_year") for e in candidate.get("education") or [] if e.get("end_year")]
    if grad_years and yoe > 2:
        grad = max(grad_years)
        years_since = (current_year - grad) + 1
        if years_since + 3 < yoe:
            flags.append("yoe_exceeds_time_since_graduation")

    # F2: tenure sum vs claimed YoE
    total_months = sum(int(j.get("duration_months") or 0) for j in career)
    if yoe > 3 and total_months > int((yoe + 2) * 12):
        flags.append("tenure_sum_exceeds_yoe")

    # F3: duration_months vs date span (6-month slack; ignore short stints)
    for job in career:
        span = months_between(parse_date(job.get("start_date")), parse_date(job.get("end_date")))
        dur = int(job.get("duration_months") or 0)
        if span is not None and dur > 12 and abs(span - dur) > 6:
            flags.append("duration_date_mismatch")
            break

    # F4: is_current contradicts end_date
    for job in career:
        if job.get("is_current") and job.get("end_date") is not None:
            flags.append("current_job_has_end_date")

    # F5: skill duration exceeds observable career span
    career_months = months_between(min_career_start(career), None) or int(yoe * 12)
    for skill in skills:
        dur = int(skill.get("duration_months") or 0)
        if dur > career_months + 12:
            flags.append("skill_duration_exceeds_career")
            break

    # F6: severely inverted salary band (mild inversion is generator noise in ~13% of profiles)
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    sal_min = sal.get("min")
    sal_max = sal.get("max")
    if (
        sal_min is not None
        and sal_max is not None
        and sal_min > sal_max
        and (sal_min - sal_max) > 10
    ):
        flags.append("salary_min_gt_max")

    # F7: OSS claim without GitHub
    gh = sig.get("github_activity_score")
    text = build_candidate_text(candidate)
    if gh == -1 and mentions_oss(text):
        flags.append("oss_claim_no_github")

    # F8: activity incoherent
    icr = float(sig.get("interview_completion_rate") or 0)
    apps = int(sig.get("applications_submitted_30d") or 0)
    views = int(sig.get("profile_views_received_30d") or 0)
    if icr > 0 and apps == 0 and views == 0:
        flags.append("activity_incoherent")

    # F9: rate bounds
    rr = sig.get("recruiter_response_rate")
    if rr is not None and not (0 <= float(rr) <= 1):
        flags.append("bad_rate")
    if icr < 0 or icr > 1:
        flags.append("bad_rate")

    # Additional hard checks from reference implementations
    signup = parse_date(sig.get("signup_date"))
    last_active = parse_date(sig.get("last_active_date"))
    if signup and signup > REFERENCE_DATE:
        flags.append("future_signup")
    for job in career:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start and end and end < start:
            flags.append("role_end_before_start")

    return flags


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def stuffer_multiplier(candidate: dict[str, Any]) -> float:
    skills = candidate.get("skills") or []
    sig = candidate.get("redrob_signals") or {}
    n_skills = len(skills)
    endorsements = int(sig.get("endorsements_received") or 0)
    assessments = len(sig.get("skill_assessment_scores") or {})
    ratio = n_skills / (1 + endorsements + assessments)
    mult = 1 - 0.5 * sigmoid(ratio - STUFFER_TAU)
    return clip(mult, 0.3, 1.0)


def signal_vector(candidate: dict[str, Any]) -> list[float]:
    sig = candidate.get("redrob_signals") or {}
    return [
        float(sig.get("recruiter_response_rate") or 0),
        float(sig.get("interview_completion_rate") or 0),
        float(sig.get("profile_completeness_score") or 0) / 100,
        math.log1p(float(sig.get("connection_count") or 0)),
        math.log1p(float(sig.get("endorsements_received") or 0)),
        math.log1p(float(sig.get("profile_views_received_30d") or 0)),
        float(sig.get("open_to_work_flag") or 0),
        float(sig.get("github_activity_score") if sig.get("github_activity_score", -1) >= 0 else 0),
    ]


def text_cosine(a: str, b: str) -> float:
    ta, tb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ta or not tb:
        return 0.0
    dot = sum(ta[t] * tb.get(t, 0) for t in ta)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


def vector_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dedup_twins(
    ranked: list[tuple],
    corroboration_key,
    text_key,
    threshold: float = 0.97,
) -> list[tuple]:
    """Drop near-duplicate behavioral twins; keep stronger corroboration."""
    kept: list[tuple] = []
    for item in ranked:
        drop = False
        text_a = text_key(item)
        sig_a = signal_vector(item[1]) if isinstance(item[1], dict) else []
        corr_a = corroboration_key(item)
        for other in kept:
            text_b = text_key(other)
            sig_b = signal_vector(other[1]) if isinstance(other[1], dict) else []
            if (
                text_cosine(text_a, text_b) > threshold
                and vector_cosine(sig_a, sig_b) > threshold
                and corr_a <= corroboration_key(other)
            ):
                drop = True
                break
        if not drop:
            kept.append(item)
    return kept