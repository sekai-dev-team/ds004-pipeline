"""DS-004 Semantic Consolidation Engine.

Core logic for:
1. Detecting new episodic notes in the vault
2. Semantic search for related knowledge
3. Trigger decision (RecMem lazy consolidation)
4. LLM consolidation into concept pages
5. Maintaining index.md and log.md
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pipeline.kmcp_client import (
    search,
    get_note,
    list_notes,
    write_note,
    update_note,
    read_vault_file,
    write_vault_file,
)
from pipeline.deepseek_client import complete, DeepSeekError

logger = logging.getLogger(__name__)

# Trigger threshold: at least θ matching old notes required
MATCH_THRESHOLD = 2
# Vector score threshold for matching
VEC_SCORE_THRESHOLD = 0.75
# Maximum query characters for search
MAX_QUERY_CHARS = 2000
# Maximum old note chars to include as context
MAX_CONTEXT_CHARS = 1000

VAULT_PATH = os.environ.get("DS004_VAULT_PATH", "/vault")
STATE_FILE = os.path.join(VAULT_PATH, ".ds004_state.json")

# Directories to skip when scanning for new notes
SKIP_DIRS = {"concepts", "insights", "digests", "daily-digest", "weekly-digest"}


def _load_state() -> tuple[set[str], dict[str, dict]]:
    """Returns (processed_notes, failed_notes)."""
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            processed = set(data.get("processed_notes", []))
            failed = data.get("failed_notes", {})
            return processed, failed
    except (FileNotFoundError, json.JSONDecodeError):
        return set(), {}


def _save_state(processed: set[str], failed: dict[str, dict] | None = None) -> None:
    """Save state. Preserves existing failed_notes if not provided."""
    if failed is None:
        _, failed = _load_state()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "processed_notes": sorted(processed),
            "failed_notes": failed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, f)


def _mark_failed(note_path: str, error_msg: str) -> None:
    """Mark a note as failed (stops retry, but preserves option to retry later)."""
    processed, failed = _load_state()
    processed.add(note_path)
    failed[note_path] = {
        "error": error_msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(processed, failed)


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_text = content[3:end].strip()
    try:
        fm = yaml.safe_load(fm_text)
        return fm if isinstance(fm, dict) else {}
    except yaml.YAMLError:
        return {}


def _is_episodic_note(content: str) -> bool:
    """Check if a note has memory_type: episodic in its frontmatter."""
    fm = _parse_frontmatter(content)
    return fm.get("memory_type") == "episodic"


def _find_new_episodic_notes() -> list[str]:
    """Scan vault root for new episodic notes not yet processed.

    Returns list of relative file paths.
    """
    processed, _ = _load_state()
    new_notes: list[str] = []

    try:
        vault_root = Path(VAULT_PATH)
        for entry in sorted(vault_root.iterdir()):
            if not entry.is_file():
                continue
            if not entry.suffix == ".md":
                continue

            rel_path = entry.name

            # Skip known non-episodic files
            if rel_path in ("index.md", "log.md", "SCHEMA.md"):
                continue

            # Skip files in subdirectories
            if any(skip in rel_path.lower() for skip in SKIP_DIRS):
                continue

            # Skip already processed
            if rel_path in processed:
                continue

            # Check if it's an episodic note
            try:
                content = entry.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            if _is_episodic_note(content):
                new_notes.append(rel_path)
                logger.info("Found new episodic note: %s", rel_path)

    except (OSError, FileNotFoundError) as exc:
        logger.error("Failed to scan vault: %s", exc)

    return new_notes


def _build_search_query(content: str) -> str:
    """Extract a search query from note content (first MAX_QUERY_CHARS chars)."""
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]

    # Clean up markdown formatting
    content = content.strip()
    # Take first N characters
    if len(content) > MAX_QUERY_CHARS:
        content = content[:MAX_QUERY_CHARS]

    return content


def _truncate_context(content: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate content for LLM context, stripping frontmatter."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]
    content = content.strip()
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[...truncated...]"
    return content


def _read_concept_pages() -> dict[str, str]:
    """Read all existing concept pages from /vault/concepts/.

    Returns dict of {concept_name: full_content}.
    """
    concepts: dict[str, str] = {}
    concepts_dir = Path(VAULT_PATH) / "concepts"
    if not concepts_dir.is_dir():
        return concepts

    for entry in sorted(concepts_dir.iterdir()):
        if entry.suffix == ".md":
            try:
                concepts[entry.name] = entry.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass

    return concepts


def _read_schema() -> str:
    """Read SCHEMA.md if it exists."""
    schema_path = Path(VAULT_PATH) / "SCHEMA.md"
    try:
        return schema_path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return "(No SCHEMA.md found)"


def _build_consolidation_prompt(
    new_note_path: str,
    new_note_content: str,
    matched_notes: list[dict[str, Any]],
    concept_pages: dict[str, str],
    schema: str,
) -> tuple[str, str]:
    """Build the system prompt and user message for LLM consolidation.

    Returns (system_prompt, user_message).
    """
    system_prompt = """You are the DS-004 Semantic Consolidation Engine. Your job is to maintain a knowledge vault using the Karpathy LLM Wiki pattern — markdown concept pages that are incrementally updated as new episodic notes arrive.

## Vault Structure
- `/vault/*.md` — episodic notes (immutable, output from DS-001 pipeline)
- `/vault/concepts/{name}.md` — semantic concept pages (you maintain these)
- `/vault/index.md` — vault directory (you maintain)
- `/vault/log.md` — operation log (append-only)

## Your Task
Given a new episodic note, determine which concept(s) it relates to, then produce the exact text content for updating/creating concept pages.

## Rules
1. Map the topic to a concept name in kebab-case (e.g., "agent-memory", "multi-agent-architecture").
2. If a concept page exists, UPDATE it incrementally:
   - Add the new note link to "## 相关文章" section
   - If new content adds a new angle, append a paragraph under an appropriate sub-heading
   - Update "## 趋势判断" if temporal patterns are visible across the matched notes
   - Update `last_updated` in frontmatter to today's date
   - If new note CONTRADICTS existing content: add "⚠️ 矛盾标注" section, do NOT delete old content
3. If concept page doesn't exist, CREATE it following the template.
4. Never rewrite entire concept pages — always incremental.
5. Be concise but substantive.

## Output Format
You MUST output a JSON object with these fields:
```json
{
  "concepts_to_update": [
    {
      "name": "kebab-case-concept-name",
      "title": "Human-readable title",
      "action": "update" or "create",
      "full_page_content": "Complete markdown content for the concept page (frontmatter + body)"
    }
  ],
  "index_update": "Content to append or update in index.md. Use ### for new entries.",
  "log_entry": "Single line starting with ## [timestamp] format for log.md"
}
```

IMPORTANT: Output ONLY the JSON object, no other text. The `full_page_content` must be valid markdown with YAML frontmatter."""

    # Build user message with all context
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    user_lines = [
        f"## New Episodic Note",
        f"**File:** {new_note_path}",
        f"**Date:** {date_str}",
        f"**Timestamp:** {timestamp}",
        f"",
        f"```markdown",
        new_note_content[:8000],  # Limit to avoid token overflow
        f"```",
        f"",
    ]

    if matched_notes:
        user_lines.append(f"## Matched Related Notes ({len(matched_notes)} found)")
        user_lines.append("")
        for i, note in enumerate(matched_notes[:5], 1):
            path = note.get("path", "unknown")
            score = note.get("vec_score", 0)
            snippet = note.get("snippet", "")
            user_lines.append(f"### {i}. {path} (score: {score:.3f})")
            user_lines.append(f"```")
            user_lines.append(snippet[:MAX_CONTEXT_CHARS])
            user_lines.append(f"```")
            user_lines.append("")

    if concept_pages:
        user_lines.append(f"## Existing Concept Pages ({len(concept_pages)} total)")
        user_lines.append("")
        for name, content in sorted(concept_pages.items()):
            user_lines.append(f"### {name}")
            user_lines.append(f"```markdown")
            user_lines.append(_truncate_context(content, 2000))
            user_lines.append(f"```")
            user_lines.append("")

    user_lines.append("## Vault Schema (SCHEMA.md)")
    user_lines.append("```markdown")
    user_lines.append(schema[:2000] if schema else "(empty)")
    user_lines.append("```")

    return system_prompt, "\n".join(user_lines)


def _concept_page_template(name: str, title: str, date_str: str) -> str:
    """Generate a new concept page from template."""
    return f"""---
memory_type: semantic
concept: {name}
tags: [type/semantic, topic/{name.replace('-', '_')}]
created: {date_str}
last_updated: {date_str}
related_concepts: []
---

# {title}

## 概述
[Newly created concept page — overview will be filled as related episodic notes are ingested.]

## 核心子主题

## 趋势判断
No trend data yet — this concept was just created.

## 相关文章

## ⚠️ 矛盾标注

## 待探索问题
"""


def _parse_llm_json(response: str) -> dict[str, Any] | None:
    """Parse the LLM JSON response, handling markdown code blocks."""
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1).strip()
    
    # Try to find raw JSON
    json_match = re.search(r'\{[\s\S]*"concepts_to_update"[\s\S]*\}', response)
    if json_match:
        response = json_match.group(0)

    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON response: %s", exc)
        logger.debug("Raw response: %s", response[:1000])
        return None


def _update_index_md(new_entries: str) -> bool:
    """Update index.md with new concept entries."""
    index_path = Path(VAULT_PATH) / "index.md"
    
    if index_path.exists():
        try:
            current = index_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            current = "# Knowledge Vault Index\n\n"
    else:
        current = "# Knowledge Vault Index\n\nGenerated by DS-004 pipeline.\n\n"

    # Append new entries
    updated = current.rstrip() + "\n\n" + new_entries + "\n"
    
    return write_vault_file("index.md", updated)


def _append_log_md(log_entry: str) -> bool:
    """Append an entry to log.md."""
    log_path = Path(VAULT_PATH) / "log.md"
    
    if log_path.exists():
        try:
            current = log_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            current = "# DS-004 Operation Log\n\n"
    else:
        current = "# DS-004 Operation Log\n\n"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"## [{timestamp}] {log_entry}\n"
    updated = current.rstrip() + "\n\n" + entry + "\n"
    
    return write_vault_file("log.md", updated)


def consolidate(note_path: str) -> bool:
    """Run full consolidation pipeline for a single new episodic note.

    Args:
        note_path: Relative path to the episodic note in the vault.

    Returns:
        True if consolidation was triggered and completed, False if skipped.
    """
    logger.info("=== Consolidating: %s ===", note_path)

    # Step 1: Read the new note
    content = read_vault_file(note_path)
    if not content:
        logger.error("Could not read note: %s", note_path)
        return False

    # Step 2: Semantic search for related notes
    query = _build_search_query(content)
    logger.info("Search query: %s", query[:200])
    
    results = search(query, limit=10)
    
    # Filter to episodic notes only, exclude self
    related = [
        r for r in results
        if r.get("path") != note_path
        and not any(skip in r.get("path", "") for skip in SKIP_DIRS)
    ]

    match_count = len(related)
    logger.info("Found %d related notes (vec_score > %.2f)", match_count, VEC_SCORE_THRESHOLD)

    for r in related[:5]:
        logger.info("  - %s (score: %.3f) %s", r.get("path", "?"), r.get("vec_score", 0), r.get("snippet", "")[:80])

    # Step 3: Trigger decision (RecMem pattern)
    if match_count < MATCH_THRESHOLD:
        log_msg = f"[skip] {note_path} — only {match_count} related notes found"
        logger.info(log_msg)
        _append_log_md(log_msg)
        # Mark as processed even if skipped
        processed, _ = _load_state()
        processed.add(note_path)
        _save_state(processed)
        return False

    logger.info("Trigger threshold met: %d >= %d — proceeding to LLM consolidation", match_count, MATCH_THRESHOLD)

    # Step 4: Gather context for LLM
    concept_pages = _read_concept_pages()
    schema = _read_schema()

    system_prompt, user_message = _build_consolidation_prompt(
        note_path, content, related, concept_pages, schema
    )

    # Step 5: Call DeepSeek LLM
    logger.info("Calling DeepSeek LLM for consolidation...")
    try:
        llm_response = complete(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=4096,
            temperature=0.3,
        )
    except DeepSeekError as exc:
        logger.error("DeepSeek LLM call failed: %s", exc)
        _append_log_md(f"[error] {note_path} — LLM call failed: {exc}")
        _mark_failed(note_path, str(exc))
        return False

    # Step 6: Parse LLM output
    parsed = _parse_llm_json(llm_response)
    if not parsed:
        logger.error("Failed to parse LLM output, raw response: %s", llm_response[:500])
        _append_log_md(f"[error] {note_path} — failed to parse LLM output")
        return False

    # Step 7: Write updated concept pages
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    concepts_to_update = parsed.get("concepts_to_update", [])

    for concept in concepts_to_update:
        name = concept.get("name", "unknown")
        action = concept.get("action", "create")
        full_content = concept.get("full_page_content", "")

        if not full_content:
            logger.warning("Empty content for concept '%s', skipping", name)
            continue

        concept_path = f"concepts/{name}.md"

        if action == "create":
            logger.info("Creating new concept page: %s", concept_path)
            write_note(concept_path, full_content, force=True)
        else:
            logger.info("Updating concept page: %s", concept_path)
            # Write the full updated content
            write_vault_file(concept_path, full_content)

    # Step 8: Update index.md
    index_update = parsed.get("index_update", "")
    if index_update:
        _update_index_md(index_update)

    # Step 9: Update log.md
    log_entry = parsed.get("log_entry", f"ingest | {note_path} → consolidated")
    concept_names = [c.get("name", "unknown") for c in concepts_to_update]
    log_line = f"ingest | {note_path} → updated {', '.join(concept_names)} ({match_count} related)"
    _append_log_md(log_line)

    # Mark as processed
    processed, _ = _load_state()
    processed.add(note_path)
    _save_state(processed)

    logger.info("=== Consolidation complete: %s ===", note_path)
    return True


def consolidate_all() -> dict[str, Any]:
    """Scan for all new episodic notes and consolidate them.

    Returns summary dict with counts.
    """
    new_notes = _find_new_episodic_notes()
    
    if not new_notes:
        logger.info("No new episodic notes found")
        return {"new_notes": 0, "consolidated": 0, "skipped": 0}

    logger.info("Found %d new episodic notes", len(new_notes))
    
    consolidated = 0
    skipped = 0
    errors = 0

    for note_path in new_notes:
        try:
            result = consolidate(note_path)
            if result:
                consolidated += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Error consolidating '%s': %s", note_path, exc, exc_info=True)
            errors += 1
            # Still mark as processed to avoid infinite retries
            processed, _ = _load_state()
            processed.add(note_path)
            _save_state(processed)

    summary = {
        "new_notes": len(new_notes),
        "consolidated": consolidated,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info("Consolidation round complete: %s", summary)
    return summary
