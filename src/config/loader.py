"""Configuration loader: YAML parsing, env var interpolation, validation."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.generation.providers import LLMProviderConfig

logger = logging.getLogger(__name__)


# --- Pydantic models for validation ---

class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_method: str = "recursive"  # "recursive" or "semantic"
    min_chunk_size: int = 100  # For semantic chunking: merge chunks below this
    separators: list[str] = Field(default_factory=lambda: [
        "\n# ", "\n## ", "\n### ", "\n#### ", "\n", "。", ".", "！", "？", "；", " ",
    ])
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
    reranker_model_revision: str = "main"
    semantic_cache: dict = Field(default_factory=lambda: {
        "enabled": True,
        "similarity_threshold": 0.95,
        "max_size": 1000,
    })


class LLMRetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: float = 1.0


class LLMConfig(BaseModel):
    temperature: float = 0.3
    max_tokens: int = 1024
    stream: bool = True
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)


class HallucinationConfig(BaseModel):
    enabled: bool = True
    entity_overlap_threshold: float = 0.7
    llm_verification: bool = False  # Optional LLM-based second-pass verification


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


def _interpolate_env_vars(value: Any) -> tuple[Any, list[str]]:
    """Recursively replace ${ENV_VAR} patterns in strings with environment values.

    Returns:
        Tuple of (interpolated_value, list_of_unresolved_var_names).
    """
    unresolved: list[str] = []

    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                unresolved.append(var_name)
                return m.group(0)
            return env_val
        return _ENV_VAR_RE.sub(_replace, value), unresolved
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            new_v, uv = _interpolate_env_vars(v)
            result[k] = new_v
            unresolved.extend(uv)
        return result, unresolved
    if isinstance(value, list):
        result = []
        for v in value:
            new_v, uv = _interpolate_env_vars(v)
            result.append(new_v)
            unresolved.extend(uv)
        return result, unresolved
    return value, unresolved


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file with env var interpolation.

    Args:
        config_path: Path to config.yaml file. Defaults to
            <project_root>/config.yaml.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    config_path = Path(config_path)

    # Load .env first so env vars are available for interpolation
    env_file = config_path.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw, unresolved = _interpolate_env_vars(raw)

    # Warn about unresolved env vars
    if unresolved:
        logger.warning(
            "Unresolved environment variables: %s. "
            "LLM calls using these providers will fail with authentication errors.",
            ", ".join(sorted(set(unresolved))),
        )

    # Flatten provider configs into LLMConfig
    llm_raw = raw.get("llm") or {}
    providers_raw = llm_raw.pop("providers", {})
    llm_raw["providers"] = {
        key: LLMProviderConfig(**val) for key, val in providers_raw.items()
    }

    config = AppConfig(
        defaults=DefaultsConfig(**raw.get("defaults", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**llm_raw),
        hallucination=HallucinationConfig(**raw.get("hallucination", {})),
        paths=PathsConfig(**raw.get("paths", {})),
    )

    # Warn if embedding model_revision is not pinned
    if config.embedding.model_revision == "main":
        logger.warning(
            "Embedding model_revision is 'main' (floating). "
            "This can cause embedding drift when the model is updated upstream. "
            "Set a specific revision hash in config.yaml to pin the version."
        )

    return config


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    return Path(__file__).parent.parent.parent