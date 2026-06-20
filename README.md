# Redrob Ranker

Retrieve-then-rerank IR system for the India Runs **Senior AI Engineer** challenge.
The job description is the query; 100k candidates are the document collection;
`submission.csv` is the ranked result set.

## Run

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Accepts `.jsonl`, `.jsonl.gz`, or a JSON array (`sample_candidates.json`).

Validate:

```bash
python validate_submission.py ./submission.csv
```

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