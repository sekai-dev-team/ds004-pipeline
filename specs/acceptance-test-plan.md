# DS-001 v0.6 + DS-004 v3.0 — Acceptance Test Plan

> 验收文档 | 2026-05-31
> 每项测试必须通过才能标记 release。

---

## Test Environment

- vault: clean state (empty `/vault/` directory, no index DB)
- k-mcp: running with `MEMORY_GUARD_MIB=2800`, embed tool available
- DS-001 container: running with memory limit 768M
- DS-004 container: running with memory limit 512M
- Both containers share `/vault` volume

---

## Phase 1: DS-001 Tag Extraction

### T1.1 — Summarization still works
```
Given:  DS-001 runs in hn-arxiv mode
When:   Pass 2 summarization completes
Then:   Each relevant article has ai_summary (Chinese, ≥150 chars, structured)
        Each article has concept_tags (list of {name, label_text})
        No regression on summary quality
```
✅ / ❌

### T1.2 — Known tags appear in LLM prompt
```
Given:  Tag library has high-frequency tags (article_count > 5)
When:   DS-001 runs summarization
Then:   SUMMARIZE_PROMPT includes known tags list
        Log shows known_tags count > 0
```
✅ / ❌

### T1.3 — LLM reuses known tags
```
Given:  Tag library contains "agent-memory" (article_count=15)
When:   An article about AI agent memory is summarized
Then:   concept_tags includes {"name": "agent-memory", ...}
        No new duplicate tag created
```
✅ / ❌

### T1.4 — LLM creates new tags
```
Given:  Tag library does NOT contain "quantum-ml"
When:   An article about quantum machine learning is summarized
Then:   concept_tags includes {"name": "quantum-ml", "label_text": "..."}
        New tag note created at /vault/tags/quantum-ml.md
        Tag note has embedding in frontmatter
```
✅ / ❌

### T1.5 — Embedding dedup prevents near-duplicates
```
Given:  Tag library has "rl-training" with embedding
When:   LLM outputs new tag {"name": "reinforcement-learning", ...}
Then:   cosine_similarity > 0.90 → tag dedup reuses "rl-training"
        No new tag note created
        Log shows dedup decision
```
✅ / ❌

### T1.6 — Frontmatter includes topic/ tags
```
Given:  Article has concept_tags ["agent-memory", "rl-training"]
When:   Note is written to vault
Then:   Frontmatter tags include: ["ai-agent", "type/episodic", "source/arxiv", "topic/agent-memory", "topic/rl-training"]
```
✅ / ❌

### T1.7 — Tag library updated correctly
```
Given:  New article tagged "agent-memory" (existing tag)
When:   process_tags() completes
Then:   /vault/tags/agent-memory.md article_count incremented by 1
        last_seen updated to today's date
```
✅ / ❌

### T1.8 — Handle tag dedup gracefully
```
Given:  Article has concept_tags = []
When:   process_tags() runs
Then:   Article.topic_tags = [] (not None, not error)
        Note written with no topic/ tags (still has type/episodic etc.)
```
✅ / ❌

---

## Phase 2: DS-004 Tag-Based Consolidation

### T2.1 — Group by tags
```
Given:   3 new notes in vault:
           Note A: topic/agent-memory, topic/rl-training
           Note B: topic/agent-memory
           Note C: topic/rl-training
When:    DS-004 consolidate_all() runs
Then:    tag_groups = {
           "agent-memory": [Note A, Note B],
           "rl-training": [Note A, Note C],
         }
```
✅ / ❌

### T2.2 — CREATE concept page (article_count >= 2)
```
Given:   Tag "rl-training" has article_count=2, no concept page
When:    DS-004 processes tag "rl-training"
Then:    concepts/rl-training.md created
         Frontmatter: memory_type=semantic, related_count=2
         Body has: ## 概述, ## 核心观点, ## 趋势判断
         ## 相关文章 section has [[wikilinks]] to both source notes
         Tag note updated: concept_page = "concepts/rl-training.md"
```
✅ / ❌

### T2.3 — UPDATE concept page (existing)
```
Given:   Tag "agent-memory" has concept_page and new note arrives
When:    DS-004 processes tag "agent-memory"
Then:    concepts/agent-memory.md updated
         Original "created" date preserved
         last_updated bumped to today
         New information from the new note incorporated
```
✅ / ❌

### T2.4 — SKIP (article_count < 2)
```
Given:   Tag "quantum-ml" has article_count=1, no concept page
When:    DS-004 processes tag "quantum-ml"
Then:    No concept page created
         Log shows: "Skipping tag 'quantum-ml': article_count=1 < θ=2"
         Note still marked as processed (won't re-process next run)
```
✅ / ❌

### T2.5 — Wikilink injection into source notes
```
Given:   DS-004 created concepts/rl-training.md
When:    _inject_wikilinks_to_notes() runs
Then:    Source notes contain "## 相关概念" section
         Section includes [[concepts/rl-training.md]]
```
✅ / ❌

### T2.6 — Cross-concept association edges
```
Given:   Concepts "agent-memory" and "rl-training" both have embeddings
When:    _build_cross_concept_links() runs
Then:    If cosine > 0.70, concepts/agent-memory.md gets:
           ## 相关概念
           - [[rl-training]]
         And vice versa
```
✅ / ❌

---

## Phase 3: Non-Functional Verification

### T3.1 — Memory stays under limit
```
Given:   DS-001 processes 100 articles
When:    Full run completes
Then:    Peak RSS < 512 MiB (DS-001) / < 256 MiB (DS-004)
         No OOM kill by Docker
```
✅ / ❌

### T3.2 — Prompt size cap
```
Given:   A tag has 20 related notes
When:    DS-004 builds consolidation prompt
Then:    Only 15 notes included (max)
         Total prompt chars ≤ 8,000
         Log shows prompt size
```
✅ / ❌

### T3.3 — Cost within budget
```
Given:   DS-001 hn-arxiv run (~50 articles) + DS-004 consolidation
When:    Both pipelines complete
Then:    Total DeepSeek cost < $0.05
         Report shows estimated_cost_usd
```
✅ / ❌

### T3.4 — Serial LLM execution (DS-004)
```
Given:   5 tags to consolidate
When:    DS-004 runs
Then:    LLM calls occur sequentially (check log timestamps)
         No concurrent DeepSeek API calls
```
✅ / ❌

### T3.5 — Graceful degradation on failure
```
Given:   k-mcp embed endpoint is down
When:    DS-001 processes new tags
Then:    New tags created with empty embedding (no crash)
         Log shows: "k-mcp embed failed, creating tag without embedding"
```
✅ / ❌

---

## Phase 4: End-to-End Integration

### T4.1 — Full pipeline: DS-001 → DS-004
```
Given:   Clean vault
When:    DS-001 runs in hn-arxiv mode, then DS-004 runs
Then:    Episodic notes created with topic/ tags
         DS-004 groups by tags, creates/updates concept pages
         Wikilinks injected both ways (notes → concepts, concepts → concepts)
         index.md and log.md updated
         Monitoring report generated in /vault/reports/
```
✅ / ❌

### T4.2 — Idempotency
```
Given:   DS-004 already ran once
When:    DS-004 runs again immediately
Then:    No new notes found → "No new episodic notes found"
         No duplicate concept pages created
         No errors
```
✅ / ❌

### T4.3 — Incremental growth
```
Given:   Vault has 50 episodic notes, 5 concept pages
When:    New DS-001 run adds 10 more notes with both new and existing tags
Then:    Existing concept pages updated (not recreated)
         New concept pages created for new tags (if article_count >= 2)
         No orphan wikilinks
```
✅ / ❌

---

## Phase 5: Regression Checks

### T5.1 — DS-004 v2.1 features removed
```
Given:   DS-004 v3.0 code
When:    Review consolidator.py
Then:    _find_all_related() — removed ⬜
         _find_relevant_concepts() — removed ⬜
         _build_search_query() — removed ⬜
         No k-mcp search() calls in consolidation path ⬜
```
✅ / ❌

### T5.2 — Old cron compatibility
```
Given:   ds004-consolidate.sh script
When:    Executed
Then:    Calls consolidate_all() successfully
         Returns report JSON to stdout
         No argument or import errors
```
✅ / ❌

---

## Acceptance Criteria Summary

- [ ] All T1.x (DS-001 tag extraction) pass
- [ ] All T2.x (DS-004 consolidation) pass
- [ ] All T3.x (NFR: memory, prompt, cost, serial) pass
- [ ] T4.1 (full end-to-end) passes
- [ ] No regressions (T5.x)
