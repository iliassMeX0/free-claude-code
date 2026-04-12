"""Z.ai provider implementation (GLM-5.1) using OpenAI-compatible API."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


class ZAIProvider(OpenAICompatibleProvider):
    """Z.ai provider using OpenAI-compatible chat completions API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZAI",
            base_url=config.base_url or ZAI_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        return build_request_body(request)
