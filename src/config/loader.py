"""Configuration loader: YAML parsing, env var interpolation, validation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# --- Pydantic models for validation ---

class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 80
    separators: list[str] = Field(default_factory=lambda: ["\n## ", "\n### ", "\n", "。", ".", " "])
    enable_deduplication: bool = True


class EmbeddingConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "BAAI/bge-m3"
    model_revision: str = "main"
    dimensions: int = 1024
    normalize: bool = True
    batch_size: int = 32
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    cache_dir: str = "data/embedding_cache"
    cache_max_entries: int = 10000


class VectorStoreConfig(BaseModel):
    backend: str = "chromadb"
    persist_dir: str = "data/chroma_db"
    distance_metric: str = "cosine"


class RetrievalConfig(BaseModel):
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    hybrid_top_k: int = 50
    rerank_top_n: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    semantic_cache: dict = Field(default_factory=lambda: {
        "enabled": True,
        "similarity_threshold": 0.95,
        "max_size": 1000,
    })


class LLMRetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: float = 1.0


class LLMProviderConfig(BaseModel):
    name: str
    base_url: str
    model: str
    api_key: str


class LLMConfig(BaseModel):
    temperature: float = 0.3
    max_tokens: int = 1024
    stream: bool = True
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)


class HallucinationConfig(BaseModel):
    enabled: bool = True
    entity_overlap_threshold: float = 0.7


class PathsConfig(BaseModel):
    kb_registry: str = "data/kb_registry.json"
    bm25_index_dir: str = "data/bm25"
    chroma_db: str = "data/chroma_db"


class DefaultsConfig(BaseModel):
    kb: str = "default"
    provider: str = "qwen"


class AppConfig(BaseModel):
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


# --- Loader ---

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} patterns in strings with environment values."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML file with env var interpolation.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    config_path = Path(config_path)

    # Load .env first so env vars are available for interpolation
    env_file = config_path.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _interpolate_env_vars(raw)

    # Flatten provider configs into LLMConfig
    llm_raw = raw.get("llm", {})
    providers_raw = llm_raw.pop("providers", {})
    llm_raw["providers"] = {
        key: LLMProviderConfig(**val) for key, val in providers_raw.items()
    }

    return AppConfig(
        defaults=DefaultsConfig(**raw.get("defaults", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**llm_raw),
        hallucination=HallucinationConfig(**raw.get("hallucination", {})),
        paths=PathsConfig(**raw.get("paths", {})),
    )


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    return Path(__file__).parent.parent.parent