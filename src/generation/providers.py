"""LLM provider configuration for OpenAI-compatible API providers.

All providers support the OpenAI-compatible API format.
"""

from pydantic import BaseModel


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    name: str           # Display name (e.g., "阿里通义千问")
    base_url: str       # OpenAI-compatible API base URL
    model: str          # Model name
    api_key: str        # API key or OAuth client_id
    auth_type: str = "bearer"  # "bearer" (default) or "oauth" for Baidu ERNIE
    client_secret: str = ""    # OAuth client secret; defaults to api_key if empty