# Redrob Ranker

Ranks candidates against a Senior AI Engineer job description: BM25 recall over the candidate pool, then structured scoring, trap checks and a ranked CSV.

**Built for:** Redrob India Runs · Senior AI Engineer challenge  
**Stack:** Python 3 standard library only (no third-party packages in `requirements.txt`)

Retrieve-then-rerank IR system for the India Runs **Senior AI Engineer** challenge.
The job description is the query; 100k candidates are the document collection;
`submission.csv` is the ranked result set.

## Quick start

Run on the bundled sample, then validate the output:

```bash
python rank.py --candidates sample_candidates.json --out submission.csv
python validate_submission.py submission.csv
```

The full candidate file (`candidates.jsonl` or `.jsonl.gz`) is not in this repository, so point `--candidates` at your copy of it to rank the full set:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Accepts `.jsonl`, `.jsonl.gz`, or a JSON array (`sample_candidates.json`).

## Pipeline

1. **Recall (BM25)** — expanded JD query over all candidates → top ~1,500
2. **Structured features** — skill-trust, title trajectory, experience, education, signals
3. **Trap gate** — honeypot hard-demote, keyword-stuffer penalty, twin dedup
4. **Fusion** — `(0.45·lexical + 0.55·structured) × trust`
5. **Reasoning** — templated from the same features (no LLM at runtime)

## Evaluation

```bash
python eval/gold.py
python eval/synth_honeypots.py
python eval/ablation.py
```

## Layout

```
rank.py
ranker/          io_stream, lexical, thesaurus, features, traps, fuse
eval/            gold, synth_honeypots, ablation
job_description.txt
sample_candidates.json
```

CPU only · ≤5 min · ≤16 GB RAM · deterministic · no network at ranking time.

---
Built by [Rhudhresh R](https://github.com/RHUDHRESH)
