"""Knowledge-MCP HTTP client for DS-004 pipeline.

Sends MCP JSON-RPC 2.0 requests to ``knowledge-mcp:8000/mcp``
to search, read, write, and update notes in the vault.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

KMCP_BASE_URL = "http://knowledge-mcp:8000/mcp"
REQUEST_TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAY = 5
_INITIALIZED = False


def _rpc_payload(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }


def _initialize() -> bool:
    """Send MCP initialize request. Must succeed before any tools/call."""
    global _INITIALIZED
    if _INITIALIZED:
        return True

    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "ds004-pipeline",
                "version": "1.0.0",
            },
        },
    }

    try:
        resp = requests.post(
            KMCP_BASE_URL,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            logger.error("k-mcp initialize failed: %s", result["error"])
            return False
        _INITIALIZED = True
        logger.info("k-mcp initialize succeeded")
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("k-mcp initialize request failed: %s", exc)
        return False


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """Call an MCP tool via tools/call. Returns the result dict or None."""
    if not _initialize():
        logger.error("k-mcp not initialized, cannot call tool: %s", name)
        return None

    payload = _rpc_payload("tools/call", {
        "name": name,
        "arguments": arguments,
    })

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                KMCP_BASE_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                logger.error("k-mcp tool '%s' error: %s", name, result["error"])
                return None
            # Extract result from MCP response structure
            return result.get("result", result)
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                logger.warning("k-mcp timeout for '%s', retrying (%d/%d)...", name, attempt + 1, MAX_RETRIES)
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error("k-mcp timeout for '%s' after %d retries", name, MAX_RETRIES)
                return None
        except requests.exceptions.RequestException as exc:
            logger.error("k-mcp request failed for '%s': %s", name, exc)
            return None

    return None


def search(query: str, limit: int = 10, exclude_path: str | None = None) -> list[dict[str, Any]]:
    """Search the vault using hybrid (BM25 + vector) search.

    Returns list of {path, section_title, snippet, bm25_score, vec_score, combined_score}.
    Only results with vec_score > 0.75 are returned.
    Filters out log.md, index.md, SCHEMA.md, and excluded path.
    """
    result = _call_tool("search", {"query": query, "limit": limit})
    if result is None:
        return []

    # Parse MCP content format: [{"type":"text","text":"[...]"}] → actual results
    content = result.get("content", [])
    results = []

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                try:
                    parsed = json.loads(block["text"])
                    if isinstance(parsed, list):
                        results = parsed
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(block, dict) and "vec_score" in block:
                results.append(block)  # Direct result object

    if not results:
        # Fallback: results might be in a different shape
        if isinstance(content, list) and all(isinstance(r, dict) and "vec_score" in r for r in content):
            results = content
        elif isinstance(result, list) and all(isinstance(r, dict) and "vec_score" in r for r in result):
            results = result

    # Filter by vec_score threshold
    threshold = 0.75
    filtered = [r for r in results if r.get("vec_score", 0) > threshold]

    # Filter out non-episodic files (log.md, index.md, SCHEMA.md)
    excluded_files = {"log.md", "index.md", "SCHEMA.md"}
    filtered = [r for r in filtered if r.get("path") not in excluded_files]

    # Exclude specific path (e.g., the note being consolidated)
    if exclude_path:
        filtered = [r for r in filtered if r.get("path") != exclude_path]

    logger.info("Search: %d raw results, %d after filters", len(results), len(filtered))
    return filtered


def get_note(path: str) -> dict[str, Any] | None:
    """Read a full markdown note with frontmatter and backlinks."""
    return _call_tool("get_note", {"path": path})


def list_notes() -> list[str]:
    """Return all .md filenames in the vault root directory."""
    result = _call_tool("list_notes", {})
    if result is None:
        return []
    # Handle different response shapes
    content = result.get("content", result) if isinstance(result, dict) else result
    if isinstance(content, list):
        return content
    return []


def write_note(path: str, content: str, frontmatter: dict[str, Any] | None = None, force: bool = True) -> bool:
    """Create or overwrite a markdown note in the vault."""
    args: dict[str, Any] = {"path": path, "content": content, "force": force}
    if frontmatter:
        args["frontmatter"] = frontmatter
    result = _call_tool("write_note", args)
    if result is None:
        return False
    logger.info("Written note: %s", path)
    return True


def update_note(path: str, old_string: str, new_string: str) -> bool:
    """Apply a string replacement to an existing note and re-index."""
    result = _call_tool("update_note", {
        "path": path,
        "old_string": old_string,
        "new_string": new_string,
    })
    if result is None:
        return False
    logger.info("Updated note: %s", path)
    return True


def index_status() -> dict[str, Any] | None:
    """Return current index statistics."""
    return _call_tool("index_status", {})


def read_vault_file(path: str) -> str | None:
    """Read a file directly from the vault filesystem (bypasses MCP).

    Falls back to filesystem read for concept pages and episodic notes.
    """
    import os
    vault_path = os.environ.get("DS004_VAULT_PATH", "/vault")
    full_path = os.path.join(vault_path, path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, FileNotFoundError):
        logger.warning("Could not read vault file: %s", full_path)
        return None


def write_vault_file(path: str, content: str) -> bool:
    """Write a file directly to the vault filesystem."""
    import os
    vault_path = os.environ.get("DS004_VAULT_PATH", "/vault")
    full_path = os.path.join(vault_path, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Written vault file: %s", path)
        return True
    except OSError as exc:
        logger.error("Failed to write vault file '%s': %s", path, exc)
        return False
