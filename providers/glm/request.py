"""Request builder for Z.ai provider (GLM-5.1)."""

from typing import Any

from providers.common.message_converter import build_base_request_body

ZAI_DEFAULT_MAX_TOKENS = 32768


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for Z.ai."""
    body = build_base_request_body(
        request_data,
        default_max_tokens=ZAI_DEFAULT_MAX_TOKENS,
    )
    body["thinking"] = {"type": "enabled"}
    return body
