"""DeepSeek LLM client via Anthropic-compatible API.

Uses the DeepSeek API endpoint with Anthropic Messages API format.
Endpoint: api.deepseek.com/anthropic
Model: deepseek-v4-flash
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_MODEL = "deepseek-v4-flash"

MAX_TOKENS = 4096
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 10


class DeepSeekError(Exception):
    """Raised when the DeepSeek API returns an error."""


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "x-api-key": DEEPSEEK_API_KEY,
        "anthropic-version": "2023-06-01",
    }


def complete(
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.3,
) -> str:
    """Send a completion request to DeepSeek via Anthropic Messages API.

    Args:
        system: System prompt.
        messages: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.

    Returns:
        The text content of the assistant's response.

    Raises:
        DeepSeekError: If the API returns an error or all retries are exhausted.
    """
    if not DEEPSEEK_API_KEY:
        raise DeepSeekError("DEEPSEEK_API_KEY environment variable is not set")

    payload = {
        "model": DEEPSEEK_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
    }

    last_error: str | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/v1/messages",
                headers=_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RETRY_DELAY))
                logger.warning("DeepSeek rate limited (429), waiting %ds...", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code != 200:
                error_body = resp.text[:500]
                logger.error("DeepSeek API returned %d: %s", resp.status_code, error_body)
                raise DeepSeekError(f"HTTP {resp.status_code}: {error_body}")

            data = resp.json()

            # Anthropic Messages API response format
            if "content" in data:
                content_blocks = data["content"]
                # Content is a list of blocks; extract text
                text_parts = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                result = "".join(text_parts)
                if result:
                    logger.info("DeepSeek response: %d chars", len(result))
                    return result
                # Content blocks exist but are empty — check stop_reason
                stop_reason = data.get("stop_reason", "")
                if stop_reason:
                    logger.warning(
                        "DeepSeek returned empty content with stop_reason=%r. "
                        "Full response: %s",
                        stop_reason,
                        json.dumps(data, ensure_ascii=False)[:2000],
                    )
                    if stop_reason == "max_tokens":
                        raise DeepSeekError(
                            f"Response truncated: stop_reason=max_tokens. "
                            f"Increase max_tokens (currently {max_tokens}) or reduce input size."
                        )
                    raise DeepSeekError(
                        f"Empty content with stop_reason={stop_reason!r}"
                    )
                # Empty content with no stop_reason — log and fall through to next checks
                logger.warning(
                    "DeepSeek returned empty content blocks with no stop_reason. "
                    "Full response: %s",
                    json.dumps(data, ensure_ascii=False)[:2000],
                )

            # Fallback: check for "choices" (OpenAI-compatible response)
            if "choices" in data:
                result = data["choices"][0].get("message", {}).get("content", "")
                if result:
                    logger.info("DeepSeek (OpenAI-format) response: %d chars", len(result))
                    return result

            # Fallback: check for standalone "message" field
            if "message" in data:
                msg = data["message"]
                if isinstance(msg, dict):
                    result = msg.get("content", "")
                elif isinstance(msg, str):
                    result = msg
                else:
                    result = str(msg)
                if result:
                    logger.info("DeepSeek (message field) response: %d chars", len(result))
                    return result

            # Fallback: check for standalone "text" field
            if "text" in data and isinstance(data["text"], str) and data["text"].strip():
                logger.info("DeepSeek (text field) response: %d chars", len(data["text"]))
                return data["text"]

            # Nothing matched — log full response for debugging
            logger.error(
                "Unexpected DeepSeek response shape. "
                "Full response (first 2000 chars): %s",
                json.dumps(data, ensure_ascii=False)[:2000],
            )
            raise DeepSeekError(
                f"Unexpected response format; keys present: {list(data.keys())}"
            )

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
            logger.warning("DeepSeek API timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES + 1)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            logger.warning("DeepSeek API request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except DeepSeekError:
            raise
        except Exception as exc:
            last_error = str(exc)
            logger.error("Unexpected error in DeepSeek call: %s", exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))

    raise DeepSeekError(f"All {MAX_RETRIES + 1} attempts failed: {last_error}")
