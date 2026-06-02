# DS-001 + DS-004 — Non-Functional Requirements (NFR)

> 2026-05-31 | Must be enforced before production deployment.
> These are not optional. Past incidents: OOM kills, 30¥+ DeepSeek bills,
> runaway process spawn.

---

## 1. Memory Budget

### DS-001

| Limit | Value | Enforcement |
|-------|-------|-------------|
| Python process RSS max | 512 MiB | Check `psutil.Process().memory_info().rss` every 10 articles; if > 512 MiB, log warning and trigger `gc.collect()` |
| Article list in-memory | Max 200 | Truncate articles list before processing; log warning |
| Full-text per article | Max 200 KiB | Truncate during fetch; discard articles with >500 KiB fulltext |
| Pass 1 batch size | 20 per worker | Chunk the article list before submitting to ThreadPoolExecutor |

### DS-004

| Limit | Value | Enforcement |
|-------|-------|-------------|
| Python process RSS max | 256 MiB | Check before each LLM call; if > 256 MiB, skip remaining tags and exit gracefully with report |
| Concept pages in prompt | Max 1 (only the target concept) | Never load multiple concept pages into a single prompt |
| Notes per tag in prompt | Max 15 | If a tag has >15 notes, select top 15 by recency |
| Prompt total chars | Max 8,000 | Truncate each note summary to 400 chars, concept page to 1,500 chars; hard-check before LLM call |

---

## 2. Concurrency Budget

### DS-001

| Resource | Max | Enforcement |
|----------|-----|-------------|
| Pass 1 workers | 5 | `max_workers=5` in ThreadPoolExecutor (existing) |
| Pass 2 workers | 3 | `max_workers=3` (existing) |
| k-mcp write concurrent | 1 (sequential) | write_notes_fs() is already sequential |
| DeepSeek API concurrent calls | Total across both passes ≤ 8 | Sum of Pass 1 + Pass 2 workers must be ≤ 8 |

### DS-004

| Resource | Max | Enforcement |
|----------|-----|-------------|
| LLM calls | 1 at a time (strictly serial) | Consolidate_all() already serial — DO NOT parallelize |
| k-mcp embed calls | 1 at a time | Only during tag processing; not inside parallel loops |
| File I/O | No threading | All vault reads/writes synchronous |
| Docker CPU limit | `--cpus=1` | Enforced in docker-compose |

---

## 3. Token / Cost Budget

### Per-Run Budgets

| Pipeline | Max Input Tokens | Max Output Tokens | Est. Max Cost |
|----------|-----------------|-------------------|---------------|
| DS-001 (hn-arxiv) | 80,000 | 8,000 | ~$0.02 |
| DS-001 (daily) | 200,000 | 20,000 | ~$0.05 |
| DS-004 | 60,000 | 12,000 | ~$0.02 |

### Enforcement

- **Pre-flight estimate**: DS-004 counts tags × avg prompt size. If estimated cost > $0.05, warn and require manual confirmation.
- **Hard stop**: If actual cost exceeds $0.10 in a single run, abort remaining tasks.
- **Per-LLM-call cap**: `max_tokens=4096` for DS-004, `max_tokens=1536` for DS-001 Pass 2.
- **Prompt size hard cap**: DS-004 build_tag_prompt() must check total chars ≤ 8,000 before calling LLM. If exceeded, truncate summaries (200 chars each) and retry. If still exceeded, skip the tag and log error.

### DS-004 Prompt Budget Breakdown

```
System prompt:          ~800 chars
Tag info:               ~200 chars
Per-note summary:       400 chars × max 15 = 6,000 chars
Concept page (update):  1,500 chars
─────────────────────────────────────────
Total max:              ~8,500 chars → must trim to ≤ 8,000
```

Trimming strategy (in order):
1. Reduce per-note summary from 400 → 300 chars
2. Reduce concept page from 1,500 → 1,000 chars
3. Drop oldest notes (keep most recent 10)

---

## 4. Error Recovery & Resilience

| Scenario | Behavior |
|----------|----------|
| k-mcp unreachable | DS-001: skip embedding for new tags, create with empty embedding. DS-004: skip cross-concept links, continue consolidation |
| DeepSeek API 429 (rate limit) | Exponential backoff: 1s, 4s, 16s, 64s. After 4 retries, skip and log. Do NOT infinite-loop. |
| DeepSeek API 5xx | Retry once after 30s. Then skip. |
| LLM returns invalid JSON | Log warning, use fallback body content. Do NOT crash. |
| Tag library directory missing | Create it. Do NOT crash. |
| Embedding vector dimension mismatch | Log warning, treat as cosine=0. Do NOT crash. |
| Tag embedding is empty or stale | DS-004 consolidate_all() checks tag embeddings before building cross-concept links. If embedding is empty, calls k-mcp embed to regenerate. Must NOT silently skip cross-concept links. |
| Single article processing fails | Skip that article, continue with remaining. Do NOT abort entire batch. |
| Docker OOM kill | DS-001/DS-004 must not exceed container memory limits. If killed, cron will retry next tick. |

---

## 5. Cron / Scheduling Constraints

| Rule | Enforcement |
|------|-------------|
| DS-001 and DS-004 must NOT run simultaneously | Docker mutex: DS-004 cron checks if DS-001 container is running before starting |
| DS-001 max frequency | Twice daily (08:00 UTC, 20:00 UTC) |
| DS-004 runs AFTER DS-001 completes | Cron schedule: DS-004 at 08:30 UTC and 20:30 UTC (30min after DS-001) |
| Single instance only | Cron script checks for existing process before spawning new one |
| Max runtime | DS-001: 20min (was 15min). DS-004: 10min. Kill and report if exceeded. |

---

## 6. Monitoring & Alerting

| Metric | Threshold | Action |
|--------|-----------|--------|
| Memory RSS | > 80% of container limit | Log warning, trigger gc |
| LLM cost per run | > $0.05 | Log warning, include in report |
| Consecutive failures | 3 in a row | Cron script sends notification (Discord) |
| Tag library size | > 200 tags | Log warning (may indicate fragmentation) |
| Concept page > 5,000 chars | — | Log info for manual review (may need splitting) |
| Prompt chars > 7,000 | — | Log debug for tuning |

---

## 7. Docker / Container Limits

```
# docker-compose.yml additions:
ds001-pipeline:
  deploy:
    resources:
      limits:
        memory: 768M
        cpus: '1.0'

ds004-pipeline:
  deploy:
    resources:
      limits:
        memory: 512M
        cpus: '0.5'

knowledge-mcp:
  deploy:
    resources:
      limits:
        memory: 3072M     # Must accommodate ONNX model (~1.7 GiB RSS)
        cpus: '2.0'
  environment:
    - MEMORY_GUARD_MIB=2800
```

---

## 8. Anti-Patterns (from past incidents)

| Anti-Pattern | Prevention |
|--------------|------------|
| Loading ALL concept pages into prompt | DS-004 loads only the target concept page, max 1 |
| Spawning N concurrent LLM calls for N notes | DS-004 is strictly serial — one LLM call per tag |
| Embedding without ONNX in container | DS-001/DS-004 call k-mcp embed API; no local ONNX |
| Unbounded article accumulation in memory | Hard limit: 200 articles per run |
| Prompt growing with vault size | Prompt size bounded by notes-per-tag limit (15), independent of vault size |
| No cost tracking | Every run estimates and reports token cost |
