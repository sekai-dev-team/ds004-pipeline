"""DS-004 Semantic Consolidation Engine — v2.1 (Prompt + Memory Optimized).

Core logic for:
1. Detecting new episodic notes in the vault
2. Semantic search for related knowledge (filtered, summary-based query)
3. Trigger decision (RecMem lazy consolidation)
4. LLM consolidation into concept pages (compact prompt, on-demand concept loading)
5. Maintaining index.md and log.md
6. Monitoring report generation
"""

from __future__ import annotations

import gc
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
VEC_SCORE_THRESHOLD = 0.90
# Maximum query characters for search
MAX_QUERY_CHARS = 2000
# Maximum old note chars to include as context
MAX_CONTEXT_CHARS = 1000
# Maximum notes to process per consolidate_all() run
MAX_NOTES_PER_RUN = 50

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


def _extract_summary(content: str, max_chars: int = 500) -> str:
    """Extract the ## 摘要 (summary) section from note content.

    Falls back to the first max_chars of content if no summary section found.
    """
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]

    # Look for ## 摘要 section
    summary_match = re.search(
        r'^##\s+摘要\s*\n(.*?)(?=\n##\s|\Z)',
        content,
        re.MULTILINE | re.DOTALL,
    )
    if summary_match:
        result = summary_match.group(1).strip()
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n[...truncated...]"
        return result

    # Fallback: first max_chars after frontmatter
    content = content.strip()
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[...truncated...]"
    return content


def _build_search_query(content: str) -> str:
    """Extract a search query from note content, preferring the ## 摘要 section.

    The 摘要 section is more focused than full text, yielding better search results.
    """
    summary = _extract_summary(content, max_chars=MAX_QUERY_CHARS)
    return summary


def _find_all_related(note_path: str, content: str, limit: int = 15) -> list[dict[str, Any]]:
    """Search k-mcp for all related articles, filtering out self and non-episodic paths.

    Args:
        note_path: Path of the note being consolidated (excluded from results).
        content: Note content used to build search query.
        limit: Maximum number of results to return (default 15).

    Returns:
        List of dicts with path, vec_score, snippet, etc., sorted by vec_score desc.
    """
    query = _build_search_query(content)
    # Search with a higher limit than needed — search() already filters by vec_score > 0.75
    results = search(query, limit=max(limit * 2, 20), exclude_path=note_path)

    # Filter out subdirectory paths (concepts, insights, digests, etc.)
    related = [
        r for r in results
        if not any(skip in r.get("path", "") for skip in SKIP_DIRS)
    ]

    # Sort by vec_score descending
    related.sort(key=lambda r: r.get("vec_score", 0), reverse=True)

    return related[:limit]


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


def _read_concept_pages(concept_names: list[str] | None = None) -> dict[str, str]:
    """Read concept pages from /vault/concepts/, optionally only specific ones.

    Args:
        concept_names: Optional list of concept names (without .md extension).
                       If None (default), loads ALL concept pages (legacy behavior).
                       For v2.1 optimization, pass only relevant concept names.

    Returns dict of {filename: full_content}.
    """
    concepts: dict[str, str] = {}
    concepts_dir = Path(VAULT_PATH) / "concepts"
    if not concepts_dir.is_dir():
        return concepts

    if concept_names is not None:
        # On-demand: load only the specified concept pages
        for name in concept_names:
            filename = f"{name}.md"
            filepath = concepts_dir / filename
            try:
                concepts[filename] = filepath.read_text(encoding="utf-8")
            except (OSError, FileNotFoundError):
                pass
    else:
        # Legacy: load all concept pages
        for entry in sorted(concepts_dir.iterdir()):
            if entry.suffix == ".md":
                try:
                    concepts[entry.name] = entry.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    pass

    return concepts


def _find_relevant_concepts(
    matched_notes: list[dict[str, Any]],
    max_concepts: int = 3,
) -> list[str]:
    """Determine relevant concept pages from matched notes' topic tags.

    Reads matched notes' frontmatter to extract topic/ tags (e.g., topic/agent_memory),
    converts them to concept page names (e.g., agent-memory).
    Returns up to max_concepts concept names (without .md extension).
    """
    concept_names: list[str] = []

    for note in matched_notes:
        note_path = note.get("path", "")
        if not note_path:
            continue

        content = read_vault_file(note_path)
        if not content:
            continue

        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        if not isinstance(tags, list):
            continue

        for tag in tags:
            if isinstance(tag, str) and tag.startswith("topic/"):
                # topic/agent_memory → agent-memory
                topic_name = tag[len("topic/"):]
                concept_name = topic_name.replace("_", "-")
                if concept_name not in concept_names:
                    concept_names.append(concept_name)

        if len(concept_names) >= max_concepts:
            break

    return concept_names[:max_concepts]


def _build_consolidation_prompt(
    new_note_path: str,
    new_note_content: str,
    matched_notes: list[dict[str, Any]],
    concept_pages: dict[str, str],
) -> tuple[str, str]:
    """Build a compact system prompt and user message for LLM consolidation.

    v2.1 optimizations:
    - System prompt reduced from ~1966 to ~800 chars
    - New note: uses ## 摘要 (summary) section, max 500 chars (was 8000)
    - Matched notes: uses 摘要 from each note, max 300 chars (was 1000 snippet)
    - Concept pages: top 3 only, max 1000 chars each (was all 74 × 2000)
    - Schema section removed (no informational value)

    Returns (system_prompt, user_message).
    """
    system_prompt = (
        "You are DS-004 Consolidation Engine. Given a new episodic note "
        "and related context, produce concept page body content.\n\n"
        "Rules:\n"
        "1. Map topic to kebab-case concept name (e.g., \"agent-memory\").\n"
        "2. EXISTING page → UPDATE: write body sections (概述, 核心观点, 趋势判断) "
        "incorporating new information incrementally. "
        "If contradiction detected, add \"⚠️ 矛盾标注\" section.\n"
        "3. NEW page → CREATE: write fresh body content with 概述, 核心观点, 趋势判断 sections.\n"
        "4. Never rewrite entire pages — always incremental.\n"
        "5. Output ONLY body markdown. Do NOT include YAML frontmatter (---). "
        "Do NOT include a '## 相关文章' section. "
        "Frontmatter and related articles are injected by the pipeline.\n\n"
        "Output ONLY valid JSON:\n"
        '{"concepts_to_update": [{"name": "...", "title": "...", '
        '"action": "update|create", "body_content": "markdown body only"}], '
        '"index_update": "...", "log_entry": "..."}'
    )

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # v2.1: new note uses summary (max 500 chars instead of 8000)
    new_note_summary = _extract_summary(new_note_content, max_chars=500)

    user_lines = [
        f"## New Episodic Note",
        f"**File:** {new_note_path}",
        f"**Date:** {date_str}",
        f"**Timestamp:** {timestamp}",
        "",
        "```markdown",
        new_note_summary,
        "```",
        "",
    ]

    if matched_notes:
        user_lines.append(
            f"## Matched Related Notes ({len(matched_notes)} found)"
        )
        user_lines.append("")
        for i, note in enumerate(matched_notes[:5], 1):
            path = note.get("path", "unknown")
            score = note.get("vec_score", 0)
            # v2.1: use 摘要 section from each matched note (300 chars)
            note_content = read_vault_file(path)
            summary = (
                _extract_summary(note_content, max_chars=300)
                if note_content
                else ""
            )
            user_lines.append(f"### {i}. {path} (score: {score:.3f})")
            user_lines.append("```")
            user_lines.append(
                summary if summary else note.get("snippet", "")[:300]
            )
            user_lines.append("```")
            user_lines.append("")

    if concept_pages:
        user_lines.append(
            f"## Existing Concept Pages ({len(concept_pages)} total)"
        )
        user_lines.append("")
        for name, content in sorted(concept_pages.items()):
            user_lines.append(f"### {name}")
            user_lines.append("```markdown")
            # v2.1: each concept page truncated to 1000 chars (was 2000)
            user_lines.append(_truncate_context(content, 1000))
            user_lines.append("```")
            user_lines.append("")

    # v2.1: Schema section removed — no informational value

    return system_prompt, "\n".join(user_lines)


def _concept_page_template(name: str, title: str, date_str: str, related_paths: list[str] | None = None) -> str:
    """Generate a new concept page from template (used as fallback when LLM fails).

    Args:
        name: Concept name (kebab-case).
        title: Page title.
        date_str: ISO date string for created/last_updated.
        related_paths: Optional list of related note paths for wikilinks.
    """
    related_lines = ""
    if related_paths:
        related_lines = "\n".join(f"- [[{p}]]" for p in related_paths)

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

## 趋势判断
No trend data yet — this concept was just created.

## 相关文章

{related_lines}

## ⚠️ 矛盾标注
"""


def _validate_frontmatter(content: str) -> bool:
    """Check that markdown content has valid frontmatter with required keys.

    Required keys: memory_type, concept, created, last_updated.
    Returns True if frontmatter is valid, False and logs a warning otherwise.
    """
    if not content.startswith("---"):
        logger.warning("Frontmatter validation failed: content does not start with '---'")
        return False
    fm = _parse_frontmatter(content)
    if not fm:
        logger.warning("Frontmatter validation failed: could not parse YAML frontmatter")
        return False
    required = ["memory_type", "concept", "created", "last_updated"]
    for key in required:
        if key not in fm:
            logger.warning("Frontmatter missing required key: %s", key)
            return False
    return True


def _validate_related_section(content: str) -> bool:
    """Check that content has a '## 相关文章' section with at least one wikilink.

    Returns True if the section exists with ≥1 [[wikilink]], False otherwise.
    """
    if "## 相关文章" not in content:
        logger.warning("Related section validation failed: missing '## 相关文章' heading")
        return False
    if not re.search(r'\[\[.*?\]\]', content):
        logger.warning("Related section validation failed: no wikilinks found in '## 相关文章'")
        return False
    return True


def _inject_frontmatter(
    name: str,
    title: str,
    body_content: str,
    related_paths: list[str],
    date_str: str,
    existing_created: str | None = None,
) -> str:
    """Generate complete markdown page with frontmatter, body, and related articles.

    On updates, preserves the original 'created' date from the existing page.

    Args:
        name: Concept name (kebab-case).
        title: Page title.
        body_content: LLM-generated body markdown (no frontmatter, no related section).
        related_paths: List of related note paths for wikilinks (sorted by relevance).
        date_str: ISO date string for last_updated (and created if new).
        existing_created: Original created date from existing page (preserved on update).

    Returns:
        Complete markdown content ready to write to vault.
    """
    created = existing_created if existing_created else date_str

    related_lines = ""
    if related_paths:
        related_lines = "\n".join(f"- [[{p}]]" for p in related_paths)

    return f"""---
memory_type: semantic
concept: {name}
title: "{title}"
tags: [type/semantic]
created: {created}
last_updated: {date_str}
related_count: {len(related_paths)}
---

# {title}

{body_content}

## 相关文章

{related_lines}
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


def consolidate(note_path: str) -> dict[str, Any]:
    """Run full consolidation pipeline for a single new episodic note.

    v2.1 changes:
    - Returns rich dict instead of bool (includes concepts, token estimates)
    - On-demand concept page loading (only relevant concepts)
    - Explicit `del` + `gc.collect()` after consolidation for memory cleanup
    - exclude_path passed to search() to filter self + log.md at source

    Args:
        note_path: Relative path to the episodic note in the vault.

    Returns:
        Dict with keys: success (bool), action (str: consolidated|skipped|error),
        concepts (list of {name, action}), input_chars (int), output_chars (int),
        frontmatter_validated (int), frontmatter_fallback (int), related_injected (int).
    """
    logger.info("=== Consolidating: %s ===", note_path)

    result: dict[str, Any] = {
        "success": False,
        "action": "error",
        "concepts": [],
        "input_chars": 0,
        "output_chars": 0,
    }

    # Step 1: Read the new note
    content = read_vault_file(note_path)
    if not content:
        logger.error("Could not read note: %s", note_path)
        return {**result, "action": "error"}

    # Step 2: Semantic search for related notes (v2.1: code-injected related articles)
    all_related = _find_all_related(note_path, content, limit=20)
    match_count = len(all_related)
    logger.info(
        "Found %d related notes (vec_score > %.2f)",
        match_count,
        VEC_SCORE_THRESHOLD,
    )

    for r in all_related[:5]:
        logger.info("  - %s (score: %.3f)", r.get("path", "?"), r.get("vec_score", 0))

    # Step 3: Trigger decision (RecMem pattern)
    # v2.1: trigger requires MATCH_THRESHOLD matches, but once triggered,
    # ALL matches (up to 15) are injected as related articles.
    if match_count < MATCH_THRESHOLD:
        log_msg = f"[skip] {note_path} — only {match_count} related notes found"
        logger.info(log_msg)
        _append_log_md(log_msg)
        processed, _ = _load_state()
        processed.add(note_path)
        _save_state(processed)
        return {**result, "success": False, "action": "skipped"}

    logger.info(
        "Trigger threshold met: %d >= %d — proceeding to LLM consolidation",
        match_count,
        MATCH_THRESHOLD,
    )

    # Step 4: Gather context for LLM
    # v2.1: on-demand concept loading — only load relevant concepts from matched notes
    relevant_concepts = _find_relevant_concepts(all_related, max_concepts=3)
    concept_pages = _read_concept_pages(
        relevant_concepts if relevant_concepts else None
    )
    logger.info(
        "Loaded %d relevant concept pages: %s",
        len(concept_pages),
        list(concept_pages.keys()),
    )

    # v2.1: _build_consolidation_prompt no longer takes schema param
    system_prompt, user_message = _build_consolidation_prompt(
        note_path, content, all_related, concept_pages
    )

    prompt_chars = len(system_prompt) + len(user_message)
    logger.info(
        "Prompt size: ~%d chars (system: %d, user: %d) — target < 8000",
        prompt_chars,
        len(system_prompt),
        len(user_message),
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
        return {**result, "action": "error"}

    # Step 6: Parse LLM output
    parsed = _parse_llm_json(llm_response)
    if not parsed:
        logger.error(
            "Failed to parse LLM output, raw response: %s", llm_response[:500]
        )
        _append_log_md(f"[error] {note_path} — failed to parse LLM output")
        return {**result, "action": "error"}

    # Step 7: Write updated concept pages (v2.1: code-injected frontmatter + related)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    concepts_to_update = parsed.get("concepts_to_update", [])

    # v2.1: Prepare related article paths for injection (up to 15, sorted by vec_score)
    all_related_paths = [r.get("path", "") for r in all_related[:15]]
    related_injected = 0
    frontmatter_validated = 0
    frontmatter_fallback = 0

    for concept in concepts_to_update:
        name = concept.get("name", "unknown")
        title = concept.get("title", name)
        action = concept.get("action", "create")
        body_content = concept.get("body_content", "")

        if not body_content or not body_content.strip():
            logger.warning(
                "Empty body_content for concept '%s', using fallback", name
            )
            body_content = (
                "## 概述\n\nLLM consolidation failed. "
                "Related articles linked below.\n"
            )
            frontmatter_fallback += 1

        concept_path = f"concepts/{name}.md"

        # On UPDATE, read existing frontmatter to preserve 'created' date
        existing_created = None
        if action == "update":
            existing_content = read_vault_file(concept_path)
            if existing_content:
                existing_fm = _parse_frontmatter(existing_content)
                existing_created = existing_fm.get("created")

        # v2.1: code generates frontmatter + injects related articles
        full_page = _inject_frontmatter(
            name, title, body_content, all_related_paths,
            date_str, existing_created,
        )

        # Validate
        if _validate_frontmatter(full_page):
            frontmatter_validated += 1
        if _validate_related_section(full_page):
            related_injected += 1

        # Write to vault
        if action == "create":
            logger.info("Creating new concept page: %s", concept_path)
            write_note(concept_path, full_page, force=True)
        else:
            logger.info("Updating concept page: %s", concept_path)
            write_vault_file(concept_path, full_page)

        result["concepts"].append({"name": name, "action": action})

    # Step 8: Update index.md
    index_update = parsed.get("index_update", "")
    if index_update:
        _update_index_md(index_update)

    # Step 9: Update log.md
    concept_names = [c.get("name", "unknown") for c in concepts_to_update]
    log_line = (
        f"ingest | {note_path} → updated {', '.join(concept_names)} "
        f"({match_count} related, {related_injected} injected)"
    )
    _append_log_md(log_line)

    # Mark as processed
    processed, _ = _load_state()
    processed.add(note_path)
    _save_state(processed)

    logger.info("=== Consolidation complete: %s ===", note_path)

    # v2.1: track token estimates and injection counts for monitoring report
    result["success"] = True
    result["action"] = "consolidated"
    result["input_chars"] = prompt_chars
    result["output_chars"] = len(llm_response)
    result["frontmatter_validated"] = frontmatter_validated
    result["frontmatter_fallback"] = frontmatter_fallback
    result["related_injected"] = related_injected

    # v2.1: explicit memory cleanup — del large strings + gc.collect()
    del content, concept_pages, system_prompt, user_message
    del llm_response, parsed, concepts_to_update, all_related, all_related_paths
    gc.collect()

    return result


def _get_peak_rss_mb() -> int:
    """Get peak RSS memory in MB from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError):
        pass
    return 0


def _build_report(
    summary: dict[str, int],
    per_concept: dict[str, dict],
    total_input_chars: int,
    total_output_chars: int,
) -> dict[str, Any]:
    """Build the monitoring report dict.

    Args:
        summary: {new_notes, consolidated, skipped, errors,
                  frontmatter_validated, frontmatter_fallback, related_injected}
        per_concept: {concept_name: {action: str, related_notes: int}}
        total_input_chars: Accumulated input prompt chars across all calls.
        total_output_chars: Accumulated output chars across all calls.

    Returns:
        Full report dict matching the DS-004 v2.1 spec format.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Rough estimate: 4 chars ≈ 1 token for mixed Chinese/English
    total_input_tokens = total_input_chars // 4
    total_output_tokens = total_output_chars // 4

    # DeepSeek v4 flash pricing (approx): $0.15/M input, $0.60/M output
    estimated_cost = (
        total_input_tokens * 0.00000015 + total_output_tokens * 0.0000006
    )

    return {
        "pipeline": "ds004-consolidate",
        "timestamp": timestamp,
        "summary": {
            "new_notes": summary["new_notes"],
            "consolidated": summary["consolidated"],
            "skipped": summary["skipped"],
            "errors": summary["errors"],
            "frontmatter_validated": summary.get("frontmatter_validated", 0),
            "frontmatter_fallback": summary.get("frontmatter_fallback", 0),
            "related_injected": summary.get("related_injected", 0),
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        },
        "per_concept": per_concept,
        "memory": {
            "peak_rss_mb": _get_peak_rss_mb(),
        },
    }


def _write_report(report: dict[str, Any]) -> None:
    """Write monitoring report to vault and stdout.

    Writes machine-readable JSON to /vault/reports/consolidate-{timestamp}.json
    and prints the same JSON to stdout for shell script parsing.
    """
    # Write to vault
    timestamp_safe = report["timestamp"].replace(":", "-")
    report_path = f"reports/consolidate-{timestamp_safe}.json"
    report_content = json.dumps(report, ensure_ascii=False, indent=2)
    write_vault_file(report_path, report_content)
    logger.info("Report written to vault: %s", report_path)

    # Print machine-readable JSON to stdout (for ds004-consolidate.sh to parse)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def consolidate_all() -> dict[str, Any]:
    """Scan for new episodic notes and consolidate them, with monitoring report.

    v2.1 changes:
    - Limited to MAX_NOTES_PER_RUN (50) notes per call
    - Returns full report dict with token/memory telemetry
    - Writes monitoring report to /vault/reports/consolidate-{timestamp}.json
    - Prints JSON report to stdout for external cron script consumption

    Returns:
        Report dict with summary, token_usage, per_concept, memory sections.
        Flat keys (new_notes, consolidated, etc.) kept for backward compat.
    """
    new_notes = _find_new_episodic_notes()

    if not new_notes:
        logger.info("No new episodic notes found")
        report = _build_report(
            {"new_notes": 0, "consolidated": 0, "skipped": 0, "errors": 0,
             "frontmatter_validated": 0, "frontmatter_fallback": 0, "related_injected": 0},
            {},
            0,
            0,
        )
        _write_report(report)
        return {"new_notes": 0, "consolidated": 0, "skipped": 0, "errors": 0,
                "frontmatter_validated": 0, "frontmatter_fallback": 0, "related_injected": 0}

    # v2.1: limit to MAX_NOTES_PER_RUN
    if len(new_notes) > MAX_NOTES_PER_RUN:
        logger.warning(
            "Truncating %d new notes to %d (max per run)",
            len(new_notes),
            MAX_NOTES_PER_RUN,
        )
        new_notes = new_notes[:MAX_NOTES_PER_RUN]

    logger.info("Found %d new episodic notes", len(new_notes))

    consolidated = 0
    skipped = 0
    errors = 0
    per_concept: dict[str, dict] = {}
    total_input_chars = 0
    total_output_chars = 0
    total_frontmatter_validated = 0
    total_frontmatter_fallback = 0
    total_related_injected = 0

    for note_path in new_notes:
        try:
            note_result = consolidate(note_path)
            action = note_result.get("action", "error")
            if action == "consolidated":
                consolidated += 1
            elif action == "skipped":
                skipped += 1
            else:
                errors += 1

            # Accumulate token estimates
            total_input_chars += note_result.get("input_chars", 0)
            total_output_chars += note_result.get("output_chars", 0)

            # v2.1: accumulate injection/validation counts
            total_frontmatter_validated += note_result.get("frontmatter_validated", 0)
            total_frontmatter_fallback += note_result.get("frontmatter_fallback", 0)
            total_related_injected += note_result.get("related_injected", 0)

            # Track per-concept actions for report
            for c in note_result.get("concepts", []):
                name = c["name"]
                if name not in per_concept:
                    per_concept[name] = {
                        "action": c["action"],
                        "related_notes": 0,
                    }
                per_concept[name]["related_notes"] += 1

        except Exception as exc:
            logger.error(
                "Error consolidating '%s': %s", note_path, exc, exc_info=True
            )
            errors += 1
            processed, _ = _load_state()
            processed.add(note_path)
            _save_state(processed)

    summary = {
        "new_notes": len(new_notes),
        "consolidated": consolidated,
        "skipped": skipped,
        "errors": errors,
        "frontmatter_validated": total_frontmatter_validated,
        "frontmatter_fallback": total_frontmatter_fallback,
        "related_injected": total_related_injected,
    }

    # Build and write monitoring report
    report = _build_report(
        summary, per_concept, total_input_chars, total_output_chars
    )
    _write_report(report)

    logger.info("Consolidation round complete: %s", report)
    return {**summary, "_report": report}
