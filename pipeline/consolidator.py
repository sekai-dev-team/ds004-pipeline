"""DS-004 Semantic Consolidation Engine — v3.0 (SYNAPSE Architecture).

Core logic for:
1. Detecting new episodic notes in the vault
2. Tag-based grouping via topic/ frontmatter tags (no k-mcp search)
3. Trigger decision (θ=2 for concept page creation)
4. LLM consolidation into concept pages (per-tag, not per-note)
5. Wikilink injection into source episodic notes
6. Cross-concept association via embedding similarity
7. Maintaining index.md, log.md, and tag library
8. Monitoring report generation
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
    read_vault_file,
    write_vault_file,
    embed,
    cosine_similarity,
)
from pipeline.deepseek_client import complete, DeepSeekError

logger = logging.getLogger(__name__)

# ---- Thresholds & constants ----

CONCEPT_CREATE_THRESHOLD = 2  # θ=2: minimum article_count to create a concept page
CROSS_CONCEPT_SIMILARITY_THRESHOLD = 0.70  # embedding similarity for cross-links
MAX_QUERY_CHARS = 2000
MAX_CONTEXT_CHARS = 1000
MAX_NOTES_PER_RUN = 50

VAULT_PATH = os.environ.get("DS004_VAULT_PATH", "/vault")
STATE_FILE = os.path.join(VAULT_PATH, ".ds004_state.json")
TAG_LIBRARY_DIR = os.path.join(VAULT_PATH, "tags")

# Directories to skip when scanning for new notes
SKIP_DIRS = {"concepts", "insights", "digests", "daily-digest", "weekly-digest", "tags", "reports"}


# ============================================================================
#  State management (unchanged from v2.1)
# ============================================================================


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


# ============================================================================
#  Note detection (unchanged from v2.1)
# ============================================================================


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

    v3.0 Phase 2: Notes in processed_notes that lack a '## 相关概念' section
    (meaning wikilinks were never injected) are treated as "new" for
    consolidation purposes, avoiding the need for --reset-state.

    Returns list of relative file paths.
    """
    processed, _ = _load_state()
    new_notes: list[str] = []
    truly_new_count = 0
    reprocess_count = 0

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

            # Read content for checks
            try:
                content = entry.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # Skip already processed — but re-process if no wikilinks section
            if rel_path in processed:
                if "## 相关概念" not in content:
                    logger.info("Re-processing (no wikilinks): %s", rel_path)
                    reprocess_count += 1
                else:
                    continue

            if _is_episodic_note(content):
                new_notes.append(rel_path)
                if rel_path not in processed:
                    truly_new_count += 1
                    logger.info("Found new episodic note: %s", rel_path)

    except (OSError, FileNotFoundError) as exc:
        logger.error("Failed to scan vault: %s", exc)

    if new_notes:
        logger.info(
            "Found %d new + %d re-processable notes",
            truly_new_count,
            reprocess_count,
        )

    return new_notes


# ============================================================================
#  Content extraction helpers (unchanged from v2.1)
# ============================================================================


def _extract_summary(content: str, max_chars: int = 500) -> str:
    """Extract the ## 摘要 (summary) section from note content."""
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


def _extract_title(content: str) -> str:
    """Extract the article title from the first # heading in markdown."""
    title_match = re.search(r'^#\s+(.+?)(?:\n|$)', content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()
    return "Untitled"


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


def _strip_metadata_sections(content: str) -> str:
    """Remove ## 相关概念 and ## 相关文章 sections from concept page content.

    These sections are pipeline-injected metadata (cross-concept links and
    related article links).  They are useless (and harmful) as LLM context
    because they consume prompt budget and can cause the LLM to reproduce
    the sections, leading to duplicate headers (Bug 3 fix).

    Args:
        content: Raw concept page content, possibly including metadata sections.

    Returns:
        Content with everything after the first occurrence of either
        ``## 相关概念`` or ``## 相关文章`` stripped.
    """
    for marker in ("## 相关概念", "## 相关文章"):
        idx = content.find(marker)
        if idx != -1:
            content = content[:idx].rstrip()
    return content


# ============================================================================
#  Tag library management (NEW in v3.0)
# ============================================================================


def read_tag_library() -> dict[str, dict[str, Any]]:
    """Read all .md files in /vault/tags/, parse frontmatter.

    Returns dict of {tag_name: tag_info} where tag_info contains
    name, embedding, label_text, article_count, concept_page,
    first_seen, last_seen.
    """
    tag_library: dict[str, dict[str, Any]] = {}
    tags_dir = Path(TAG_LIBRARY_DIR)

    if not tags_dir.is_dir():
        logger.info("Tag library directory does not exist: %s", TAG_LIBRARY_DIR)
        return tag_library

    for entry in sorted(tags_dir.iterdir()):
        if not entry.suffix == ".md":
            continue

        try:
            content = entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        fm = _parse_frontmatter(content)
        if not fm:
            continue

        tag_name = fm.get("name") or entry.stem
        embedding_raw = fm.get("embedding")
        if isinstance(embedding_raw, str):
            try:
                embedding_raw = json.loads(embedding_raw)
            except (json.JSONDecodeError, TypeError):
                embedding_raw = None

        tag_library[tag_name] = {
            "name": tag_name,
            "embedding": embedding_raw if isinstance(embedding_raw, list) else None,
            "label_text": fm.get("label_text", ""),
            "article_count": fm.get("article_count", 0),
            "concept_page": fm.get("concept_page"),
            "first_seen": fm.get("first_seen"),
            "last_seen": fm.get("last_seen"),
        }

    logger.info("Loaded %d tags from tag library", len(tag_library))
    return tag_library


def write_tag_note(name: str, label_text: str, embedding: list[float]) -> bool:
    """Create /vault/tags/{name}.md with frontmatter."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # Serialize embedding compactly
    emb_str = json.dumps(embedding)

    content = f"""---
name: {name}
embedding: {emb_str}
label_text: "{label_text}"
article_count: 1
concept_page:
first_seen: {date_str}
last_seen: {date_str}
---

# {name}

{label_text}
"""

    tag_path = f"tags/{name}.md"
    return write_vault_file(tag_path, content)


def update_tag_note(name: str, updates: dict[str, Any]) -> bool:
    """Patch frontmatter fields in /vault/tags/{name}.md.

    Updates are applied to the frontmatter only. If a key is set to None,
    it is removed from the frontmatter. The body content is preserved.

    Args:
        name: Tag name (kebab-case).
        updates: Dict of frontmatter key-value pairs to update.
    """
    tag_path = f"tags/{name}.md"
    full_path = os.path.join(VAULT_PATH, tag_path)

    if not os.path.isfile(full_path):
        logger.warning("Tag note not found: %s", tag_path)
        return False

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return False

    # Parse existing frontmatter
    fm = _parse_frontmatter(content)
    if not fm:
        return False

    # Apply updates
    for key, value in updates.items():
        if value is None:
            fm.pop(key, None)
        else:
            fm[key] = value

    # Rebuild the file
    # Get the body (everything after frontmatter)
    fm_end = content.find("---", 3)
    body = content[fm_end + 3:].strip() if fm_end != -1 else ""

    # Build new frontmatter
    fm_lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            # Embedding arrays: compact JSON
            fm_lines.append(f"{k}: {json.dumps(v)}")
        elif isinstance(v, str) and (" " in v or ":" in v or v.startswith('"')):
            fm_lines.append(f'{k}: "{v}"')
        elif v is None or v == "":
            fm_lines.append(f"{k}:")
        else:
            fm_lines.append(f"{k}: {v}")
    fm_lines.append("---")

    new_content = "\n".join(fm_lines) + "\n\n" + body + "\n"

    return write_vault_file(tag_path, new_content)


# ============================================================================
#  Tag-based grouping (NEW in v3.0)
# ============================================================================


def _group_by_tags(note_paths: list[str]) -> dict[str, list[str]]:
    """Group episodic notes by their topic/ frontmatter tags.

    Reads each note's frontmatter, extracts topic/ tags (e.g., topic/agent-memory
    → agent-memory), and groups note paths under each tag name.

    Returns dict of {tag_name: [note_path, ...]}.
    """
    tag_groups: dict[str, list[str]] = {}

    for note_path in note_paths:
        content = read_vault_file(note_path)
        if not content:
            continue

        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        if not isinstance(tags, list):
            continue

        # Extract topic/ tags
        topics = [str(t).replace("topic/", "") for t in tags if isinstance(t, str) and t.startswith("topic/")]

        for topic in topics:
            tag_groups.setdefault(topic, []).append(note_path)

    logger.info("Grouped %d notes into %d tag groups", len(note_paths), len(tag_groups))
    for tag_name, paths in tag_groups.items():
        logger.info("  %s: %d notes", tag_name, len(paths))

    return tag_groups


# ============================================================================
#  LLM prompt building (NEW in v3.0 — replaces per-note prompts)
# ============================================================================


def _build_tag_prompt(
    action: str,
    tag_name: str,
    note_paths: list[str],
    concept_content: str | None = None,
    max_summary_chars: int = 500,
    max_concept_chars: int = 2000,
) -> tuple[str, str]:
    """Build system prompt and user message for per-tag LLM consolidation.

    Args:
        action: "create" or "update".
        tag_name: Concept tag name (kebab-case).
        note_paths: List of episodic note paths in this tag group.
        concept_content: Existing concept page content (only for "update").

    Returns:
        (system_prompt, user_message) tuple.
    """
    system_prompt = (
        "You are DS-004 Consolidation Engine. Given a concept tag and "
        "related article summaries, produce or update a concept page.\n\n"
        "Rules:\n"
        "1. For CREATE: write fresh body content with 概述, 核心观点, 趋势判断 sections.\n"
        "2. For UPDATE: incorporate new information incrementally into existing sections.\n"
        "   If contradiction detected, add '⚠️ 矛盾标注' section.\n"
        "3. Never rewrite entire pages — always incremental on updates.\n"
        "4. Output ONLY body markdown. Do NOT include YAML frontmatter (---).\n"
        "5. Do NOT include a '## 相关文章' section (injected by pipeline).\n\n"
        "Output ONLY valid JSON:\n"
        '{"title": "Concept Page Title", '
        '"body_content": "markdown body (概述, 核心观点, 趋势判断 sections)", '
        '"index_update": "...", "log_entry": "..."}'
    )

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    user_lines = [
        "## Concept Tag",
        f"**Name:** {tag_name}",
        f"**Action:** {action}",
        "",
    ]

    # Related articles
    user_lines.append(f"## Related Articles ({len(note_paths)} articles)")
    user_lines.append("")

    for i, note_path in enumerate(note_paths, 1):
        note_content = read_vault_file(note_path)
        if note_content:
            title = _extract_title(note_content)
            summary = _extract_summary(note_content, max_chars=max_summary_chars)
        else:
            title = "Untitled"
            summary = "(could not read note)"

        user_lines.append(f"### {i}. {title}")
        user_lines.append(f"**File:** {note_path}")
        user_lines.append("```markdown")
        user_lines.append(summary)
        user_lines.append("```")
        user_lines.append("")

    # For UPDATE: include existing concept page content
    if action == "update" and concept_content:
        user_lines.append("## Current Concept Page")
        user_lines.append("```markdown")
        user_lines.append(_truncate_context(concept_content, max_chars=max_concept_chars))
        user_lines.append("```")
        user_lines.append("")

    return system_prompt, "\n".join(user_lines)


# ============================================================================
#  LLM response parsing (unchanged from v2.1)
# ============================================================================


def _parse_llm_json(response: str) -> dict[str, Any] | None:
    """Parse the LLM JSON response, handling markdown code blocks."""
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1).strip()

    # Try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*"title"[\s\S]*\}', response)
    if json_match:
        response = json_match.group(0)

    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON response: %s", exc)
        logger.debug("Raw response: %s", response[:1000])
        return None


# ============================================================================
#  Frontmatter injection (adapted for v3.0)
# ============================================================================


def _inject_concept_frontmatter(
    name: str,
    title: str,
    body_content: str,
    date_str: str,
    existing_created: str | None = None,
) -> str:
    """Generate complete markdown page with frontmatter and body.

    On updates, preserves the original 'created' date from the existing page.
    Does NOT include '## 相关文章' (v3.0: wikilinks go into episodic notes).
    The '## 相关概念' section is added later by _build_cross_concept_links.

    Bug 1 fix: strips metadata sections from body_content (in case the LLM
    included ## 相关概念 or ## 相关文章), and avoids duplicating the
    ``# {title}`` heading when body_content already starts with one.
    """
    created = existing_created if existing_created else date_str

    # ---- Bug 1 / Bug 3: strip pipeline-injected sections from LLM output ----
    body_content = _strip_metadata_sections(body_content)

    # ---- Bug 1: avoid duplicating the # title heading ----
    title_heading = f"# {title}"
    if body_content.lstrip().startswith(title_heading):
        # LLM already emitted the title heading — do not prepend another
        body_block = body_content
    else:
        body_block = f"# {title}\n\n{body_content}"

    return f"""---
memory_type: semantic
concept: {name}
title: "{title}"
tags: [type/semantic, topic/{name}]
created: {created}
last_updated: {date_str}
---

{body_block}

## 相关概念

"""
    # Cross-concept links will be appended here by _build_cross_concept_links


# ============================================================================
#  Wikilink injection (NEW in v3.0)
# ============================================================================


def _inject_wikilinks_to_notes(
    note_paths: list[str],
    tag_library: dict[str, dict[str, Any]],
) -> int:
    """Add [[concepts/xxx.md]] wikilinks to source episodic notes.

    For each note, reads frontmatter to get topic/ tags, looks up
    whether each tag has a concept_page, and injects wikilinks.

    Returns number of wikilinks injected.
    """
    injected_count = 0

    for note_path in note_paths:
        content = read_vault_file(note_path)
        if not content:
            continue

        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        if not isinstance(tags, list):
            continue

        # Extract topic/ tags
        topics = [str(t).replace("topic/", "") for t in tags if isinstance(t, str) and t.startswith("topic/")]

        # Build wikilinks for concepts that exist
        wikilinks_to_add: list[str] = []
        for topic in topics:
            tag_info = tag_library.get(topic, {})
            concept_page = tag_info.get("concept_page")
            if concept_page:
                link = f"- [[{concept_page}]]"
                # M3: Check only within the "## 相关概念" section, not full content
                section_match = re.search(
                    r'^##\s+相关概念\s*\n(.*?)(?=\n##\s|\Z)',
                    content,
                    re.MULTILINE | re.DOTALL,
                )
                if section_match:
                    section_content = section_match.group(1)
                    if link not in section_content:
                        wikilinks_to_add.append(link)
                else:
                    # No "## 相关概念" section exists yet — always add
                    wikilinks_to_add.append(link)

        if not wikilinks_to_add:
            continue

        # Inject wikilinks at the end of the note, before existing related links if any
        wikilink_block = "\n## 相关概念\n\n" + "\n".join(wikilinks_to_add) + "\n"

        # If there's already a "## 相关概念" section, append to it
        if "## 相关概念" in content:
            # M3: Check only within the section, not full content
            section_match = re.search(
                r'^##\s+相关概念\s*\n(.*?)(?=\n##\s|\Z)',
                content,
                re.MULTILINE | re.DOTALL,
            )
            section_content = section_match.group(1) if section_match else ""
            # Add new links after the heading
            for link in wikilinks_to_add:
                if link not in section_content:
                    # Simple append at end of file
                    updated = content.rstrip() + "\n" + link + "\n"
                    write_vault_file(note_path, updated)
                    injected_count += 1
                    content = updated  # Track for subsequent checks
        else:
            # Append new section at end
            updated = content.rstrip() + "\n\n" + wikilink_block
            write_vault_file(note_path, updated)
            injected_count += len(wikilinks_to_add)
            logger.info("Injected %d wikilink(s) into %s", len(wikilinks_to_add), note_path)

    logger.info("Total wikilinks injected into notes: %d", injected_count)
    return injected_count


def _build_cross_concept_links(
    updated_tag_names: list[str],
    tag_library: dict[str, dict[str, Any]],
) -> int:
    """Find similar concept pages via embedding similarity and add [[wikilinks]].

    For each concept that was updated/created in this run, compute cosine
    similarity against all other concepts with concept pages. If similarity
    > CROSS_CONCEPT_SIMILARITY_THRESHOLD (0.70), add a [[wikilink]].

    Returns number of cross-concept links added.
    """
    links_added = 0

    # Get all concepts that have pages
    concepts_with_pages = [
        (name, info) for name, info in tag_library.items()
        if info.get("concept_page") and info.get("embedding")
    ]

    if len(concepts_with_pages) < 2:
        return 0

    for tag_name in updated_tag_names:
        tag_info = tag_library.get(tag_name, {})
        source_embedding = tag_info.get("embedding")
        source_concept_page = tag_info.get("concept_page")

        if not source_embedding or not source_concept_page:
            continue

        # Read the source concept page
        source_content = read_vault_file(source_concept_page)
        if not source_content:
            continue

        cross_links: list[str] = []

        for other_name, other_info in concepts_with_pages:
            if other_name == tag_name:
                continue

            other_embedding = other_info.get("embedding")
            other_concept_page = other_info.get("concept_page")
            if not other_embedding or not other_concept_page:
                continue

            similarity = cosine_similarity(source_embedding, other_embedding)
            if similarity > CROSS_CONCEPT_SIMILARITY_THRESHOLD:
                link = f"- [[{other_concept_page}]]"
                # Check only within "## 相关概念" section for dedup
                section_match = re.search(
                    r'^##\s+相关概念\s*\n(.*?)(?=\n##\s|\Z)',
                    source_content,
                    re.MULTILINE | re.DOTALL,
                )
                section_content = section_match.group(1) if section_match else ""
                if link not in section_content:
                    cross_links.append(link)
                    logger.info(
                        "Cross-concept link: %s ↔ %s (similarity: %.3f)",
                        tag_name, other_name, similarity,
                    )

        if cross_links:
            links_str = "\n".join(cross_links) + "\n"

            # Find the "## 相关概念" section and insert links within it
            # (Bug 1 fix: insert BEFORE any subsequent ## heading such as
            #  ``## 相关文章``, NOT at the end of the file)
            heading_match = re.search(
                r'^##\s+相关概念\s*$',
                source_content,
                re.MULTILINE,
            )
            if heading_match:
                # Insert links right after the heading line (skip blank lines)
                insert_pos = heading_match.end()
                rest = source_content[insert_pos:]
                blank_match = re.match(r'[ \t]*\n', rest)
                if blank_match:
                    insert_pos += blank_match.end()
                updated = (
                    source_content[:insert_pos]
                    + links_str
                    + source_content[insert_pos:]
                )
                write_vault_file(source_concept_page, updated)
                links_added += len(cross_links)
            else:
                # No "## 相关概念" section exists — add at end of file
                section = "\n\n## 相关概念\n\n" + links_str
                updated = source_content.rstrip() + section
                write_vault_file(source_concept_page, updated)
                links_added += len(cross_links)

    logger.info("Total cross-concept links added: %d", links_added)
    return links_added


def _inject_related_articles(
    concept_path: str,
    note_paths: list[str],
) -> None:
    """Add ``## 相关文章`` section to a concept page (Bug 2 fix).

    Inserts wikilinks pointing back to the episodic source notes that
    contributed to this concept.  If the section already exists from a
    previous run, new links are appended without duplicating existing ones.

    Args:
        concept_path: Relative vault path of the concept page
                      (e.g. ``concepts/reinforcement-learning.md``).
        note_paths:   Episodic note paths in this tag group.
    """
    content = read_vault_file(concept_path)
    if not content:
        logger.warning(
            "Cannot inject related articles — page not found: %s", concept_path
        )
        return

    article_links = [f"- [[{path}]]" for path in sorted(set(note_paths))]

    # Check whether a ``## 相关文章`` section already exists
    section_match = re.search(
        r'^##\s+相关文章\s*\n(.*?)(?=\n##\s|\Z)',
        content,
        re.MULTILINE | re.DOTALL,
    )

    if section_match:
        # Append links that aren't already present
        existing = section_match.group(1)
        links_to_add = [l for l in article_links if l not in existing]
        if not links_to_add:
            logger.debug("No new related articles to add to %s", concept_path)
            return

        insert_before = section_match.end(1)
        updated = (
            content[:insert_before]
            + "\n".join(links_to_add)
            + "\n"
            + content[insert_before:]
        )
        logger.info(
            "Appended %d new article link(s) to %s",
            len(links_to_add),
            concept_path,
        )
    else:
        # Create a brand-new section
        section_body = "\n".join(article_links)
        updated = content.rstrip() + f"\n\n## 相关文章\n{section_body}\n"
        logger.info(
            "Added ## 相关文章 section (%d links) to %s",
            len(article_links),
            concept_path,
        )

    write_vault_file(concept_path, updated)


# ============================================================================
#  Index and log maintenance (unchanged from v2.1)
# ============================================================================


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


# ============================================================================
#  Monitoring report (unchanged from v2.1)
# ============================================================================


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
    """Build the monitoring report dict."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    total_input_tokens = total_input_chars // 4
    total_output_tokens = total_output_chars // 4
    estimated_cost = (
        total_input_tokens * 0.00000015 + total_output_tokens * 0.0000006
    )

    return {
        "pipeline": "ds004-consolidate",
        "version": "v3.0-synapse",
        "timestamp": timestamp,
        "summary": {
            "new_notes": summary["new_notes"],
            "consolidated": summary["consolidated"],
            "skipped": summary["skipped"],
            "errors": summary["errors"],
            "cross_concept_links": summary.get("cross_concept_links", 0),
            "wikilinks_injected": summary.get("wikilinks_injected", 0),
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
    """Write monitoring report to vault and stdout."""
    timestamp_safe = report["timestamp"].replace(":", "-")
    report_path = f"reports/consolidate-{timestamp_safe}.json"
    report_content = json.dumps(report, ensure_ascii=False, indent=2)
    write_vault_file(report_path, report_content)
    logger.info("Report written to vault: %s", report_path)

    # Print machine-readable JSON to stdout (for ds004-consolidate.sh)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _cleanup_duplicate_headers() -> None:
    """One-time cleanup: deduplicate ``# title`` and ``## 相关概念`` headers.

    Existing concept pages may have accumulated duplicate headers from
    pre-fix consolidation runs.  This function scans every page under
    ``concepts/`` and cleans them up.  It is idempotent and safe to
    call on every run (it short-circuits when no duplicates are found).
    """
    concepts_dir = Path(VAULT_PATH) / "concepts"
    if not concepts_dir.is_dir():
        return

    cleaned = 0
    for entry in sorted(concepts_dir.iterdir()):
        if not entry.suffix == ".md":
            continue
        try:
            content = entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        original = content

        # --- Deduplicate # title (keep only the first occurrence) ---
        lines = content.split("\n")
        seen_title = False
        new_lines: list[str] = []
        for line in lines:
            if line.startswith("# ") and not line.startswith("## "):
                if not seen_title:
                    seen_title = True
                    new_lines.append(line)
                # else: skip duplicate title
            else:
                new_lines.append(line)
        content = "\n".join(new_lines)

        # --- Deduplicate ## 相关概念 headings (keep only the first) ---
        heading_pat = re.compile(r"^##\s+相关概念\s*$", re.MULTILINE)
        headings = list(heading_pat.finditer(content))
        if len(headings) > 1:
            for h in reversed(headings[1:]):
                content = content[: h.start()] + content[h.end() :]

        if content != original:
            entry.write_text(content, encoding="utf-8")
            cleaned += 1
            logger.info("Cleaned duplicate headers in concepts/%s", entry.name)

    if cleaned:
        logger.info("Deduplicated headers in %d concept page(s)", cleaned)
    else:
        logger.debug("No duplicate headers found in concept pages")


# ============================================================================
#  Main consolidation loop (REWRITTEN for v3.0 SYNAPSE)
# ============================================================================


def consolidate_all() -> dict[str, Any]:
    """Scan for new episodic notes and consolidate them by tag groups.

    v3.0 SYNAPSE flow:
    1. Find new episodic notes
    2. Group by topic/ frontmatter tags
    3. For each tag: decide create/update/skip based on article_count
    4. Serial LLM calls (one per tag)
    5. Inject wikilinks into source episodic notes
    6. Build cross-concept association edges via embedding similarity
    7. Update index.md and log.md

    Returns:
        Report dict with summary, token_usage, per_concept, memory sections.
        Flat keys (new_notes, consolidated, etc.) kept for backward compat.
    """
    # Bug 1 one-time cleanup: deduplicate headers in existing concept pages
    _cleanup_duplicate_headers()

    # Step 1: Find new episodic notes
    new_notes = _find_new_episodic_notes()

    if not new_notes:
        logger.info("No new episodic notes found")
        report = _build_report(
            {"new_notes": 0, "consolidated": 0, "skipped": 0, "errors": 0,
             "cross_concept_links": 0, "wikilinks_injected": 0},
            {},
            0,
            0,
        )
        _write_report(report)
        return {"new_notes": 0, "consolidated": 0, "skipped": 0, "errors": 0}

    # v3.0: limit to MAX_NOTES_PER_RUN
    if len(new_notes) > MAX_NOTES_PER_RUN:
        logger.warning(
            "Truncating %d new notes to %d (max per run)",
            len(new_notes),
            MAX_NOTES_PER_RUN,
        )
        new_notes = new_notes[:MAX_NOTES_PER_RUN]

    logger.info("Found %d new episodic notes", len(new_notes))

    # Step 2: Group by topic/ tags
    tag_groups = _group_by_tags(new_notes)

    if not tag_groups:
        logger.info("No topic/ tags found in new notes")
        # Mark all as processed even though no tags
        processed, _ = _load_state()
        for note_path in new_notes:
            processed.add(note_path)
        _save_state(processed)

        report = _build_report(
            {"new_notes": len(new_notes), "consolidated": 0, "skipped": len(new_notes), "errors": 0,
             "cross_concept_links": 0, "wikilinks_injected": 0},
            {},
            0,
            0,
        )
        _write_report(report)
        return {"new_notes": len(new_notes), "consolidated": 0, "skipped": len(new_notes), "errors": 0}

    # Step 3: Read tag library and decide actions
    tag_library = read_tag_library()

    # v3.0 Phase 2: Auto-regenerate empty tag embeddings before cross-concept linking
    if tag_library:
        regenerated = 0
        skipped_no_page = 0
        for tag_name, tag_info in tag_library.items():
            embedding = tag_info.get("embedding")
            concept_page = tag_info.get("concept_page")

            # Skip if embedding is already a non-empty list
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                continue

            if not concept_page:
                logger.info("Skipping tag '%s': no concept_page", tag_name)
                skipped_no_page += 1
                continue

            try:
                new_embedding = embed(tag_name)
                if new_embedding is not None:
                    # Persist to tag .md file
                    update_tag_note(tag_name, {"embedding": new_embedding})
                    # Update in-memory cache
                    tag_info["embedding"] = new_embedding
                    regenerated += 1
                    logger.info("Regenerated embedding for tag '%s'", tag_name)
                else:
                    logger.warning(
                        "Failed to generate embedding for tag '%s'", tag_name
                    )
            except Exception as exc:
                logger.warning(
                    "Error generating embedding for tag '%s': %s", tag_name, exc
                )

        logger.info(
            "Regenerated %d/%d tag embeddings (skipped %d: no concept_page)",
            regenerated,
            len(tag_library),
            skipped_no_page,
        )

    consolidation_tasks: list[tuple[str, str, list[str], str | None]] = []
    updated_tag_names: list[str] = []
    skipped_tags: list[str] = []

    for tag_name, note_paths in tag_groups.items():
        # Cap notes per tag at 15, keeping most recent (sorted by path)
        if len(note_paths) > 15:
            logger.info(
                "Tag '%s': capping %d notes → 15 (most recent)",
                tag_name, len(note_paths),
            )
            note_paths = sorted(note_paths)[:15]

        tag_info = tag_library.get(tag_name, {})
        concept_page = tag_info.get("concept_page")
        article_count = tag_info.get("article_count", len(note_paths))

        if concept_page:
            # UPDATE: existing concept page always gets updated
            concept_content = read_vault_file(concept_page)
            # Bug 3: strip metadata sections before sending to LLM
            concept_content = _strip_metadata_sections(concept_content)
            consolidation_tasks.append(("update", tag_name, note_paths, concept_content))
            updated_tag_names.append(tag_name)
        elif article_count >= CONCEPT_CREATE_THRESHOLD:
            # CREATE: enough articles for a new concept page
            consolidation_tasks.append(("create", tag_name, note_paths, None))
            updated_tag_names.append(tag_name)
        else:
            # SKIP: not enough articles yet
            skipped_tags.append(tag_name)
            logger.info(
                "Skipping tag '%s': article_count=%d < θ=%d",
                tag_name, article_count, CONCEPT_CREATE_THRESHOLD,
            )

    logger.info(
        "Consolidation plan: %d tasks (%d create, %d update, %d skip)",
        len(consolidation_tasks),
        sum(1 for t in consolidation_tasks if t[0] == "create"),
        sum(1 for t in consolidation_tasks if t[0] == "update"),
        len(skipped_tags),
    )

    # Step 4: Serial LLM calls (one per tag)
    consolidated = 0
    errors = 0
    per_concept: dict[str, dict] = {}
    total_input_chars = 0
    total_output_chars = 0

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for action, tag_name, note_paths, concept_content in consolidation_tasks:
        logger.info("=== Consolidating tag: %s (action: %s, notes: %d) ===", tag_name, action, len(note_paths))

        # Build prompt
        system_prompt, user_message = _build_tag_prompt(action, tag_name, note_paths, concept_content)
        prompt_chars = len(system_prompt) + len(user_message)
        total_input_chars += prompt_chars
        logger.info("Prompt size: ~%d chars (system: %d, user: %d)", prompt_chars, len(system_prompt), len(user_message))

        # M1: Prompt size hard cap at 8000 chars
        MAX_PROMPT_CHARS = 8000
        if prompt_chars > MAX_PROMPT_CHARS:
            logger.warning(
                "Prompt too large (%d > %d), trimming notes for tag '%s'",
                prompt_chars, MAX_PROMPT_CHARS, tag_name,
            )
            # Level 1: reduce to 10 notes + 300 char summaries
            trimmed_paths = note_paths[:10]
            system_prompt, user_message = _build_tag_prompt(
                action, tag_name, trimmed_paths, concept_content,
                max_summary_chars=300,
            )
            prompt_chars = len(system_prompt) + len(user_message)
            total_input_chars += prompt_chars
            logger.info(
                "After trim (10 notes, 300 char summaries): ~%d chars",
                prompt_chars,
            )
            # Update note_paths for downstream use
            note_paths = trimmed_paths

            # Level 2: still too large? Reduce concept_page to 1000 chars
            if prompt_chars > MAX_PROMPT_CHARS:
                logger.warning(
                    "Still too large (%d > %d), reducing concept page context",
                    prompt_chars, MAX_PROMPT_CHARS,
                )
                system_prompt, user_message = _build_tag_prompt(
                    action, tag_name, note_paths, concept_content,
                    max_summary_chars=300, max_concept_chars=1000,
                )
                prompt_chars = len(system_prompt) + len(user_message)
                total_input_chars += prompt_chars
                logger.info(
                    "After concept truncation (1000 chars): ~%d chars",
                    prompt_chars,
                )

            # Level 3: still too large? Skip this tag
            if prompt_chars > MAX_PROMPT_CHARS:
                logger.error(
                    "Prompt still too large (%d > %d) after all trimming — "
                    "skipping tag '%s'",
                    prompt_chars, MAX_PROMPT_CHARS, tag_name,
                )
                _append_log_md(
                    f"[skip] tag/{tag_name} — prompt too large "
                    f"({prompt_chars} > {MAX_PROMPT_CHARS}) after trimming"
                )
                errors += 1
                continue

        # Call DeepSeek LLM
        try:
            llm_response = complete(
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=4096,
                temperature=0.3,
            )
            total_output_chars += len(llm_response)
        except DeepSeekError as exc:
            logger.error("DeepSeek LLM call failed for tag '%s': %s", tag_name, exc)
            _append_log_md(f"[error] tag/{tag_name} — LLM call failed: {exc}")
            errors += 1
            continue

        # Parse LLM output
        parsed = _parse_llm_json(llm_response)
        if not parsed:
            logger.error("Failed to parse LLM output for tag '%s'", tag_name)
            _append_log_md(f"[error] tag/{tag_name} — failed to parse LLM output")
            errors += 1
            continue

        # Extract fields
        title = parsed.get("title", tag_name.replace("-", " ").title())
        body_content = parsed.get("body_content", "")
        index_update = parsed.get("index_update", "")
        log_entry = parsed.get("log_entry", "")

        if not body_content or not body_content.strip():
            logger.warning("Empty body_content for tag '%s', using fallback", tag_name)
            body_content = (
                f"## 概述\n\nLLM consolidation incomplete. "
                f"Related articles: {len(note_paths)}.\n"
            )

        concept_path = f"concepts/{tag_name}.md"

        # On UPDATE, preserve original 'created' date
        existing_created = None
        if action == "update" and concept_content:
            existing_fm = _parse_frontmatter(concept_content)
            existing_created = existing_fm.get("created")

        # Build full concept page with frontmatter
        full_page = _inject_concept_frontmatter(
            tag_name, title, body_content, date_str, existing_created,
        )

        # Write to vault
        if action == "create":
            logger.info("Creating new concept page: %s", concept_path)
            write_vault_file(concept_path, full_page)
            # Update tag library: set concept_page
            update_tag_note(tag_name, {"concept_page": concept_path})
        else:
            logger.info("Updating concept page: %s", concept_path)
            write_vault_file(concept_path, full_page)

        # Update tag library: increment article_count, update last_seen
        if tag_name in tag_library:
            new_count = tag_library[tag_name].get("article_count", 0) + len(note_paths)
            update_tag_note(tag_name, {
                "article_count": new_count,
                "last_seen": date_str,
            })
            tag_library[tag_name]["article_count"] = new_count
            tag_library[tag_name]["concept_page"] = concept_path
        else:
            # This tag wasn't in the library yet — create it
            label_text = title
            tag_embedding = embed(label_text)
            if tag_embedding:
                write_tag_note(tag_name, label_text, tag_embedding)
                tag_library[tag_name] = {
                    "name": tag_name,
                    "embedding": tag_embedding,
                    "label_text": label_text,
                    "article_count": len(note_paths),
                    "concept_page": concept_path,
                    "first_seen": date_str,
                    "last_seen": date_str,
                }

        # Update index.md
        if index_update:
            _update_index_md(index_update)

        # Log
        if log_entry:
            _append_log_md(log_entry)
        else:
            _append_log_md(
                f"[{action}] tag/{tag_name} — {len(note_paths)} notes consolidated"
            )

        consolidated += 1

        # Track per-concept actions
        per_concept[tag_name] = {
            "action": action,
            "related_notes": len(note_paths),
            "note_paths": note_paths,
        }

        # Memory cleanup
        del system_prompt, user_message, llm_response, parsed
        gc.collect()

    # Step 5: Inject wikilinks into source episodic notes
    wikilinks_injected = _inject_wikilinks_to_notes(new_notes, tag_library)

    # Step 6: Cross-concept association edges
    # Phase 2.1: Process ALL concept-bearing tags, not just updated_tag_names
    all_concept_tags = [
        name for name, info in tag_library.items()
        if info.get("concept_page") and info.get("embedding")
    ]
    cross_concept_links = _build_cross_concept_links(all_concept_tags, tag_library)

    # Step 7: Inject ## 相关文章 into all consolidated concept pages
    # (moved after cross-concept links so ## 相关概念 appears first)
    for tag_name, data in per_concept.items():
        concept_path = f"concepts/{tag_name}.md"
        _inject_related_articles(concept_path, data["note_paths"])

    # Step 8: Mark all new notes as processed
    processed, _ = _load_state()
    for note_path in new_notes:
        processed.add(note_path)
    _save_state(processed)

    # Build summary
    skipped_count = len(new_notes) - consolidated - errors
    summary = {
        "new_notes": len(new_notes),
        "consolidated": consolidated,
        "skipped": skipped_count if skipped_count > 0 else 0,
        "errors": errors,
        "cross_concept_links": cross_concept_links,
        "wikilinks_injected": wikilinks_injected,
    }

    # Build and write monitoring report
    report = _build_report(summary, per_concept, total_input_chars, total_output_chars)
    _write_report(report)

    logger.info("=== Consolidation round complete: %s ===", summary)
    return {**summary, "_report": report}
