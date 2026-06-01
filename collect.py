#!/usr/bin/env python3
"""DS-004 Pipeline — Semantic Consolidation Engine (v3.0 SYNAPSE).

Watches for new episodic notes in the knowledge vault, groups them
by topic/ frontmatter tags, and calls DeepSeek LLM to consolidate
concepts into semantic concept pages.

Architecture (v3.0):
    DS-001 pipeline writes new episodic note to /vault/
      → ds004-pipeline detects new file
      → Group by topic/ frontmatter tags (no k-mcp search)
      → If article_count >= θ (2), trigger consolidation
      → DeepSeek LLM updates/creates concept pages per tag
      → Inject [[wikilinks]] into source episodic notes
      → Build cross-concept association edges
      → Updates index.md and log.md

Usage:
    python collect.py --mode consolidate
    python collect.py --mode watch     # Continuous watch mode (poll every 60s)

Modes:
    consolidate    Single scan + consolidation pass for new notes
    watch          Continuous watch mode, polls every 60 seconds
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

from pipeline.consolidator import consolidate_all, _load_state, _save_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("ds004")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-004 Semantic Consolidation Pipeline"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["consolidate", "watch"],
        help="Pipeline mode: consolidate (single pass) or watch (continuous)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds for watch mode (default: 60)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Clear failed_notes queue and re-process failed notes",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Wipe all processing state (start fresh)",
    )
    args = parser.parse_args()

    pipeline_name = f"ds004-{args.mode}"
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("Starting pipeline: %s at %s", pipeline_name, timestamp)

    # Handle --reset-state: wipe all processing state
    if args.reset_state:
        logger.warning("--reset-state: wiping all processing state")
        _save_state(set(), {})
        logger.info("State file cleared — will re-process all notes")

    # Handle --retry-failed: clear the failed queue so failed notes are retried
    if args.retry_failed:
        processed, _ = _load_state()
        _save_state(processed, {})
        logger.info("Cleared failed_notes queue — will retry all previously failed notes")

    if args.mode == "consolidate":
        _run_consolidate()
    elif args.mode == "watch":
        _run_watch(args.interval)


def _run_consolidate() -> None:
    """Run a single consolidation pass."""
    summary = consolidate_all()

    summary_counts = summary  # Flat keys for backward compat logging
    logger.info("=" * 60)
    logger.info("DS-004 Consolidation Summary (v3.0 SYNAPSE):")
    logger.info("  New notes found:        %d", summary_counts.get("new_notes", 0))
    logger.info("  Consolidated (tags):    %d", summary_counts.get("consolidated", 0))
    logger.info("  Skipped (below θ):      %d", summary_counts.get("skipped", 0))
    logger.info("  Errors:                 %d", summary_counts.get("errors", 0))
    logger.info("  Wikilinks injected:     %d", summary_counts.get("wikilinks_injected", 0))
    logger.info("  Cross-concept links:    %d", summary_counts.get("cross_concept_links", 0))
    logger.info("=" * 60)

    # The full report (with pipeline, token_usage, memory sections) is
    # already printed to stdout by consolidate_all() → _write_report()
    # The ds004-consolidate.sh cron script captures this line.


def _run_watch(interval: int) -> None:
    """Run continuous watch mode, polling every `interval` seconds."""
    logger.info("Watch mode started, polling every %ds", interval)
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            logger.info("--- Polling for new notes ---")
            try:
                summary = consolidate_all()
                if summary.get("new_notes", 0) > 0:
                    logger.info(
                        "Round complete: %d new, %d consolidated, %d skipped, %d errors",
                        summary["new_notes"],
                        summary["consolidated"],
                        summary["skipped"],
                        summary["errors"],
                    )
                else:
                    logger.info("No new notes to process")
            except Exception as exc:
                logger.error("Error in consolidation round: %s", exc, exc_info=True)

            logger.info("Sleeping %ds...", interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Watch mode stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
