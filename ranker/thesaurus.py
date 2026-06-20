"""Controlled-vocabulary query expansion for the Senior AI Engineer JD."""

from __future__ import annotations

import re
from functools import lru_cache

# JD concept -> synonyms and plain-language paraphrases
THESAURUS: dict[str, list[str]] = {
    "machine learning": [
        "ml", "machine learning", "deep learning", "neural network", "neural networks",
        "built models", "trained models", "model training", "inference", "ml pipeline",
        "mlops", "production ml", "deployed models",
    ],
    "embeddings": [
        "embedding", "embeddings", "vector", "vectors", "semantic search",
        "sentence-transformers", "sentence transformers", "bge", "e5",
        "dense retrieval", "bi-encoder", "representation learning",
        "encoded text", "vector representation",
    ],
    "retrieval": [
        "retrieval", "information retrieval", "search", "semantic search",
        "hybrid search", "bm25", "sparse retrieval", "candidate generation",
        "query understanding", "matching engine", "relevance",
        "found candidates", "matched profiles", "search ranking",
    ],
    "ranking": [
        "ranking", "reranking", "re-ranking", "learning to rank", "ltr",
        "ndcg", "mrr", "map", "precision at k", "recall at k",
        "scored results", "ordered results", "relevance scoring",
    ],
    "rag": [
        "rag", "retrieval augmented generation", "retrieval-augmented",
        "knowledge retrieval", "context retrieval", "augmented generation",
    ],
    "llm": [
        "llm", "large language model", "language model", "transformer",
        "transformers", "gpt", "llama", "mistral", "fine-tuning", "fine tuning",
        "finetuning", "instruction tuning", "prompt engineering", "lora", "qlora", "peft",
    ],
    "nlp": [
        "nlp", "natural language processing", "text processing", "text classification",
        "named entity", "summarization", "language understanding",
    ],
    "vector database": [
        "vector database", "vector db", "vectordb", "pinecone", "weaviate", "qdrant",
        "milvus", "faiss", "opensearch", "elasticsearch", "chroma", "chromadb", "pgvector",
    ],
    "recommendation": [
        "recommendation", "recommender", "recsys", "recommendation systems",
        "personalization", "collaborative filtering", "two-tower",
    ],
    "python": [
        "python", "pytorch", "tensorflow", "pandas", "numpy", "scikit-learn", "sklearn",
        "fastapi", "flask",
    ],
    "evaluation": [
        "evaluation", "a/b testing", "ab testing", "offline evaluation", "online evaluation",
        "experiment tracking", "model monitoring", "benchmark", "metrics",
    ],
    "data pipeline": [
        "data pipeline", "etl", "airflow", "spark", "pyspark", "kafka", "streaming",
        "batch processing", "feature pipeline", "data engineering",
    ],
    "ml engineer": [
        "ml engineer", "ai engineer", "machine learning engineer", "applied scientist",
        "research engineer", "nlp engineer", "senior ai", "senior ml",
    ],
    "production": [
        "production", "deployed", "shipped", "launched", "real users", "at scale",
        "serving", "inference", "latency", "monitoring", "rollout",
    ],
}

# Canonical required skills derived from JD
REQUIRED_SKILLS: list[str] = [
    "embeddings", "vector database", "retrieval", "ranking", "python",
    "evaluation", "nlp", "llm", "rag", "machine learning",
]

TITLE_TERMS: list[str] = [
    "ai engineer", "ml engineer", "machine learning engineer", "applied scientist",
    "nlp engineer", "research engineer", "senior ai", "senior ml", "data scientist",
    "search engineer", "relevance engineer", "applied ml",
]

OSS_TERMS: tuple[str, ...] = (
    "open source", "open-source", "oss", "github contributor", "contributor to",
    "maintainer", "open source project",
)

TIER_SCORES: dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.85,
    "tier_3": 0.65,
    "tier_4": 0.45,
    "tier_5": 0.35,
}

EXP_IDEAL_LO = 5.0
EXP_IDEAL_HI = 9.0
EXP_SOFT_LO = 4.0
EXP_SOFT_HI = 11.0

WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*[a-z0-9]|[a-z0-9]", re.IGNORECASE)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for concept, aliases in THESAURUS.items():
        for alias in aliases:
            index[alias.lower()] = concept
    return index


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def expand_text(text: str) -> str:
    """Append controlled synonyms for concepts already present in text."""
    lower = (text or "").lower()
    extras: list[str] = []
    for concept, aliases in THESAURUS.items():
        if any(alias in lower for alias in aliases):
            extras.extend(aliases[:6])
    if not extras:
        return text
    return f"{text} {' '.join(sorted(set(extras)))}"


def canonicalize_skill(name: str) -> str | None:
    key = (name or "").strip().lower()
    if not key:
        return None
    idx = _alias_index()
    if key in idx:
        return idx[key]
    for alias, concept in idx.items():
        if alias in key or key in alias:
            return concept
    return None


def skill_matches_required(skill_name: str) -> str | None:
    concept = canonicalize_skill(skill_name)
    if concept and concept in REQUIRED_SKILLS:
        return concept
    lower = (skill_name or "").lower()
    for req in REQUIRED_SKILLS:
        for alias in THESAURUS.get(req, [req]):
            if alias in lower or lower in alias:
                return req
    return None


def mentions_oss(text: str) -> bool:
    lower = (text or "").lower()
    return any(term in lower for term in OSS_TERMS)


def build_expanded_query(jd_text: str) -> str:
    return expand_text(jd_text)