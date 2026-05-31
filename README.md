# DS-004 Semantic Consolidation Pipeline

**Phase 1**: Watches for new episodic notes in the knowledge vault, uses k-mcp hybrid search to find semantically related knowledge, and calls DeepSeek LLM to consolidate concepts into semantic concept pages.

## Architecture

```
DS-001 pipeline writes new episodic note to /vault/
  → ds004-pipeline detects new file
  → k-mcp search finds semantically similar old notes
  → If match_count >= 2 (θ=2), trigger consolidation
  → DeepSeek LLM updates/creates concept pages
  → Updates index.md and log.md
```

## Academic Foundation

- **RecMem** (arxiv:2605.16045): Lazy consolidation — only trigger when semantically similar content recurs
- **GAAMA** (arxiv:2603.27910): Concept-mediated graphs — use topic labels, not entity nodes
- **Karpathy LLM Wiki**: Markdown wiki as persistent knowledge artifact, LLM maintains all pages

## Usage

### Docker

```bash
# Single consolidation pass
docker run --rm \
  --network agent-net \
  -v vault_data:/vault \
  -e DEEPSEEK_API_KEY=sk-... \
  kona01z/ds004-pipeline:latest \
  --mode consolidate

# Continuous watch mode (polls every 60s)
docker run -d --name ds004-pipeline \
  --network agent-net \
  -v vault_data:/vault \
  -e DEEPSEEK_API_KEY=sk-... \
  --restart unless-stopped \
  kona01z/ds004-pipeline:latest \
  --mode watch
```

### Local Development

```bash
pip install -r requirements.txt
DEEPSEEK_API_KEY=sk-... python collect.py --mode consolidate
```

## Key Design Decisions

- **θ = 2**: At least 2 matching old notes to trigger consolidation (RecMem lazy pattern)
- **vec_score threshold = 0.75**: Semantic similarity threshold for k-mcp search results
- **Incremental updates**: Concept pages are updated incrementally, never fully rewritten
- **Contradictions flagged**: Conflicting information is annotated, not auto-resolved
- **LLM-autonomous taxonomy**: Concept pages are created by LLM, no pre-defined taxonomy

## Vault Structure

```
/vault/
├── {date}-{slug}.md              ← episodic notes (DS-001 output, IMMUTABLE)
├── concepts/                     ← semantic concept pages (DS-004 maintains)
├── index.md                      ← vault directory (LLM maintains)
├── log.md                        ← operation log
└── SCHEMA.md                     ← vault conventions
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key for LLM calls |
| `DS004_VAULT_PATH` | No | Vault path (default: `/vault`) |

## Docker Image

- **Image**: `kona01z/ds004-pipeline:latest`
- **Base**: `python:3.13-slim`
- **Network**: Must be on `agent-net` to reach `knowledge-mcp:8000`
