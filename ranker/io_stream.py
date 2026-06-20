"""Stream JSONL/JSONL.gz and build candidate/query text."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from ranker.thesaurus import build_expanded_query


def _open_path(path: str | Path):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate records from .jsonl, .jsonl.gz, or JSON array."""
    path = Path(path)
    with _open_path(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for row in data:
                yield row
            return
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_candidate_text(candidate: dict[str, Any]) -> str:
    """Concatenate searchable fields per spec §3."""
    parts: list[str] = []
    profile = candidate.get("profile") or {}
    parts.append(profile.get("headline") or "")
    parts.append(profile.get("summary") or "")

    for job in candidate.get("career_history") or []:
        parts.append(job.get("title") or "")
        parts.append(job.get("description") or "")

    for skill in candidate.get("skills") or []:
        parts.append(skill.get("name") or "")

    return " ".join(p for p in parts if p)


def load_job_description(path: str | Path | None = None) -> str:
    if path is None:
        for candidate in (
            Path("job_description.txt"),
            Path("data/job_description.txt"),
        ):
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError("job_description.txt not found")
    return Path(path).read_text(encoding="utf-8")


def build_query_text(jd_path: str | Path | None = None) -> str:
    jd = load_job_description(jd_path)
    return build_expanded_query(jd)


def infer_current_year(candidates_path: str | Path) -> int:
    """Max education end_year / last_active year observed in data."""
    max_year = 2026
    for cand in iter_candidates(candidates_path):
        for edu in cand.get("education") or []:
            end = edu.get("end_year")
            if isinstance(end, int):
                max_year = max(max_year, end)
        sig = cand.get("redrob_signals") or {}
        last = sig.get("last_active_date")
        if last and len(str(last)) >= 4:
            try:
                max_year = max(max_year, int(str(last)[:4]))
            except ValueError:
                pass
    return max_year