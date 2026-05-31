#!/usr/bin/env python3
"""DS-004 Pipeline — Semantic Consolidation Engine.

Watches for new episodic notes in the knowledge vault, uses k-mcp
hybrid search to find semantically related knowledge, and calls
DeepSeek LLM to consolidate concepts into semantic concept pages.

Architecture:
    DS-001 pipeline writes new episodic note to /vault/
      → ds004-pipeline detects new file
      → k-mcp search finds semantically similar old notes
      → If match_count >= θ (2), trigger consolidation
      → DeepSeek LLM updates/creates concept pages
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
import logging
import sys
import time
from datetime import datetime, timezone

from pipeline.consolidator import consolidate_all, _load_state

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
    args = parser.parse_args()

    pipeline_name = f"ds004-{args.mode}"
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("Starting pipeline: %s at %s", pipeline_name, timestamp)

    if args.mode == "consolidate":
        _run_consolidate()
    elif args.mode == "watch":
        _run_watch(args.interval)


def _run_consolidate() -> None:
    """Run a single consolidation pass."""
    summary = consolidate_all()
    
    logger.info("=" * 60)
    logger.info("DS-004 Consolidation Summary:")
    logger.info("  New notes found:   %d", summary.get("new_notes", 0))
    logger.info("  Consolidated:      %d", summary.get("consolidated", 0))
    logger.info("  Skipped (below θ): %d", summary.get("skipped", 0))
    logger.info("  Errors:            %d", summary.get("errors", 0))
    logger.info("=" * 60)


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
