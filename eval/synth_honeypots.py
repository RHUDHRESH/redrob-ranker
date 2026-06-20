#!/usr/bin/env python3
"""Synthetic honeypots — assert 100% trap-gate catch rate."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranker.traps import impossibility_flags

REFERENCE_YEAR = 2026


def _base() -> dict:
    sample = ROOT / "sample_candidates.json"
    data = json.loads(sample.read_text(encoding="utf-8"))
    return copy.deepcopy(data[0])


def build_honeypots() -> list[tuple[str, dict]]:
    traps: list[tuple[str, dict]] = []

    c = _base()
    c["candidate_id"] = "HONEYPOT_001"
    c["profile"]["years_of_experience"] = 25
    c["education"][0]["end_year"] = 2020
    traps.append(("yoe_exceeds_time_since_graduation", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_002"
    c["profile"]["years_of_experience"] = 5
    for job in c["career_history"]:
        job["duration_months"] = 120
    traps.append(("tenure_sum_exceeds_yoe", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_003"
    c["career_history"][0]["duration_months"] = 999
    traps.append(("duration_date_mismatch", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_004"
    c["career_history"][0]["is_current"] = True
    c["career_history"][0]["end_date"] = "2025-01-01"
    traps.append(("current_job_has_end_date", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_005"
    c["skills"].append({"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 500})
    traps.append(("skill_duration_exceeds_career", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_006"
    c["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 50, "max": 20}
    traps.append(("salary_min_gt_max", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_007"
    c["redrob_signals"]["github_activity_score"] = -1
    c["profile"]["summary"] += " Active open-source contributor at github.com/example."
    traps.append(("oss_claim_no_github", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_008"
    c["redrob_signals"]["interview_completion_rate"] = 0.8
    c["redrob_signals"]["applications_submitted_30d"] = 0
    c["redrob_signals"]["profile_views_received_30d"] = 0
    traps.append(("activity_incoherent", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_009"
    c["redrob_signals"]["recruiter_response_rate"] = 1.5
    traps.append(("bad_rate", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_010"
    c["redrob_signals"]["signup_date"] = "2027-01-01"
    traps.append(("future_signup", c))

    c = _base()
    c["candidate_id"] = "HONEYPOT_012"
    c["career_history"][0]["start_date"] = "2024-01-01"
    c["career_history"][0]["end_date"] = "2023-01-01"
    traps.append(("role_end_before_start", c))

    return traps


def main() -> int:
    traps = build_honeypots()
    missed = []
    for expected, cand in traps:
        flags = impossibility_flags(cand, REFERENCE_YEAR)
        if not flags:
            missed.append((cand["candidate_id"], expected))
        else:
            print(f"OK  {cand['candidate_id']}: caught via {flags[0]}")

    if missed:
        print(f"\nFAILED: {len(missed)} honeypots not caught:")
        for cid, reason in missed:
            print(f"  - {cid} ({reason})")
        return 1

    print(f"\nAll {len(traps)} synthetic honeypots caught (100%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())