"""DS-004 Semantic Consolidation Pipeline — v3.0 SYNAPSE.

Watches for new episodic notes in the knowledge vault, groups them
by topic/ frontmatter tags, and calls DeepSeek LLM to consolidate
concepts into semantic concept pages.

v3.0: Tag-based grouping (SYNAPSE architecture).
No k-mcp search for concept lookup — tags are the concept keys.
"""

from pipeline.consolidator import (
    consolidate_all,
    _load_state,
    _save_state,
    read_tag_library,
    write_tag_note,
    update_tag_note,
)
from pipeline.kmcp_client import embed, cosine_similarity

__all__ = [
    "consolidate_all",
    "_load_state",
    "_save_state",
    "read_tag_library",
    "write_tag_note",
    "update_tag_note",
    "embed",
    "cosine_similarity",
]
