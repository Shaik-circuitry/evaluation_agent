"""Portkey-backed OpenAI-compatible chat completions."""

from __future__ import annotations

from typing import Any

from config import (
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    PORTKEY_API_KEY,
    PORTKEY_BASE_URL,
    PORTKEY_CONFIG,
    LLM_MODEL,
)


def _portkey_headers() -> dict[str, str]:
    from portkey_ai import createHeaders

    api_key = PORTKEY_API_KEY or "portkey-placeholder"
    try:
        return createHeaders(
            api_key=api_key,
            config=PORTKEY_CONFIG,
        )
    except TypeError:
        return createHeaders(
            {
                "api_key": api_key,
                "config": PORTKEY_CONFIG,
            }
        )


def chat_completion(
    *,
    system: str | None,
    user: str,
    max_tokens: int,
    temperature: float | None = None,
    response_format_json: bool = False,
) -> str:
    """Single chat completion; returns assistant message content (may be empty on failure)."""
    from openai import OpenAI

    temp = LLM_TEMPERATURE if temperature is None else temperature
    hdrs = _portkey_headers()
    client = OpenAI(
        api_key="litellm-dummy",
        base_url=PORTKEY_BASE_URL.rstrip("/"),
        default_headers=hdrs,
        timeout=LLM_TIMEOUT_S,
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "temperature": temp,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()
