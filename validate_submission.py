#!/usr/bin/env python3
"""Validate submission CSV per challenge rules."""

import csv
import re
import sys
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_DATA_ROWS = 100


def validate_submission(csv_path):
    errors = []
    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        errors.append("Filename must use a .csv extension.")

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("Row 1 must be the header row; file is empty.")
                return errors

            if header != REQUIRED_HEADER:
                errors.append(
                    f"Row 1 (header) must be exactly: {','.join(REQUIRED_HEADER)}"
                )

            data_rows = [row for row in reader if any(cell.strip() for cell in row)]
    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded.")
        return errors
    except OSError as e:
        errors.append(f"Cannot read file: {e}")
        return errors

    if len(data_rows) != EXPECTED_DATA_ROWS:
        errors.append(f"Expected {EXPECTED_DATA_ROWS} data rows; found {len(data_rows)}.")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    by_rank: list[tuple[int, float, str]] = []

    for i, cells in enumerate(data_rows):
        row_num = 2 + i
        if len(cells) != len(REQUIRED_HEADER):
            errors.append(f"Row {row_num}: expected {len(REQUIRED_HEADER)} columns.")
            continue

        row = dict(zip(REQUIRED_HEADER, cells))
        cid = row["candidate_id"].strip()

        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id {cid!r}.")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id {cid!r}.")
        else:
            seen_ids.add(cid)

        try:
            rank = int(row["rank"].strip())
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank must be 1-100.")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}.")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer.")
            rank = None

        try:
            score = float(row["score"].strip())
        except ValueError:
            errors.append(f"Row {row_num}: score must be float.")
            score = None

        if rank is not None and score is not None:
            by_rank.append((rank, score, cid))

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"score non-increasing violated: rank {r1} ({s1}) < rank {r2} ({s2})")
        if s1 == s2 and c1 > c2:
            errors.append(f"tie-break violated at ranks {r1}/{r2}: {c1!r} > {c2!r}")

    return errors


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <submission.csv>")
        sys.exit(1)
    errors = validate_submission(sys.argv[1])
    if errors:
        print(f"Validation failed ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Submission is valid.")


if __name__ == "__main__":
    main()