"""LLM provider configurations for 5 Chinese LLM providers.

All providers support the OpenAI-compatible API format.
"""

from pydantic import BaseModel


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    name: str           # Display name (e.g., "阿里通义千问")
    base_url: str       # OpenAI-compatible API base URL
    model: str          # Model name
    api_key: str        # API key


# Provider defaults (api_key is loaded from env vars via config.yaml)
PROVIDER_DEFAULTS = {
    "qwen": {
        "name": "阿里通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
    },
    "glm": {
        "name": "智谱GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "hunyuan": {
        "name": "腾讯混元",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "model": "hunyuan-lite",
    },
    "ernie": {
        "name": "百度文心一言",
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ernie-speed-128k",
    },
}