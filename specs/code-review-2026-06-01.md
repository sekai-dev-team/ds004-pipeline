# Code Review — DS-001 v0.6 + DS-004 v3.0

> 2026-06-01 | Review by Yui | → CC to fix

---

## 🔴 Critical

### R1. DS-001: `filter_articles()` 缺少 `tag_library` 参数

**文件**: `/tmp/ds001-pipeline/collect.py` L102  
**问题**: 调用 `filter_articles(unique_articles, api_key)` 时未传递 `tag_library`，导致后续运行时 LLM 看不到已有高频标签。首次运行无影响（没有已知标签），但第二次起 LLM 不知道有哪些已有概念。

**修复**: 
```python
# collect.py L27 增加导入
from pipeline.knowledge_mcp import write_notes_fs, reindex_vault, write_digest, process_tags, _read_tag_library

# collect.py L102 传入标签库
tag_library = _read_tag_library()
relevant_articles = filter_articles(unique_articles, api_key, tag_library)
```

### R2. DS-001 CI DockerHub 认证失败

**问题**: `DOCKERHUB_TOKEN` 是 org-level secret，按 "Selected repositories" 限制访问。ds001-pipeline 不在白名单中。  
**修复**: 在 GitHub org Settings → Secrets → DOCKERHUB_TOKEN → 添加 `sekai-dev-team/ds001-pipeline` 到 selected repos。  
**Workaround**: 本地已手动构建推送镜像 `kona01z/ds001-pipeline:latest` (sha256:7e3437ed)。

---

## 🟡 Medium

### M1. DS-004: 缺少 Prompt Size 硬上限

**文件**: `pipeline/consolidator.py` `_build_tag_prompt()` (L402-473)  
**问题**: NFR 规定 prompt ≤ 8,000 chars，但代码中无检查。如果一个标签下有 20+ 篇文章，每篇摘要 500 chars，prompt 会超过 10,000 chars。  
**修复**: 在 `consolidate_all()` 的 Step 4 LLM 调用前（L912 附近）加入：
```python
prompt_chars = len(system_prompt) + len(user_message)
MAX_PROMPT_CHARS = 8000
if prompt_chars > MAX_PROMPT_CHARS:
    logger.warning("Prompt too large (%d > %d), trimming notes", prompt_chars, MAX_PROMPT_CHARS)
    # 重新构建 prompt，限制 10 篇笔记 + 每篇 300 chars
```

### M2. DS-004: 缺少每标签笔记数上限

**文件**: `consolidator.py` `_group_by_tags()`  
**问题**: NFR 规定每标签最多 15 篇笔记进 prompt，但 `_group_by_tags()` 无上限。  
**修复**: 在 `consolidate_all()` Step 3 中，对每个 tag group 截断：
```python
if len(note_paths) > 15:
    note_paths = sorted(note_paths)[:15]  # 保留最新 15 篇
```

### M3. DS-004: `_inject_wikilinks_to_notes` 链接去重逻辑脆弱

**文件**: `consolidator.py` L578  
**问题**: `if link not in content` 用字符串包含做去重——如果正文中出现 `[[concepts/agent-memory.md]]` 作为示例文本，会误判为已存在而不注入。  
**修复**: 改为正则匹配 `## 相关概念` section 中的 wikilinks，只检查该 section 内是否已有相同链接。

### M4. DS-004: 概念页 frontmatter 缺少 `topic/` 标签

**文件**: `consolidator.py` L525  
**问题**: `_inject_concept_frontmatter` 生成的 frontmatter 中 tags 为 `[type/semantic]`，缺少 `topic/{name}` 标签。虽然 v3.0 用标签库做查找（不依赖 frontmatter），但一致性上应该加入。  
**修复**: L525 改为 `tags: [type/semantic, topic/{name}]`

---

## 🟢 Low / NFR Compliance

### L1. 所有仓库: Docker 镜像只有 `:latest` 标签

**问题**: CI workflow 只推送 `:latest`，无法回滚。  
**修复**: 在 workflow 中增加 commit SHA 标签：
```yaml
tags: |
  kona01z/ds001-pipeline:latest
  kona01z/ds001-pipeline:sha-${{ github.sha }}
```

### L2. DS-001: `write_note()` (HTTP) 未同步更新

**文件**: `knowledge_mcp.py` L418  
**问题**: `write_note()`（通过 k-mcp API 写入）的 frontmatter 未加入 `topic/` 标签。虽然主流程用 `write_notes_fs()`，但保持一致性。  
**修复**: 同 `write_notes_fs()` 的修改方式。

### L3. DS-004: `_build_tag_prompt` 输出 JSON key 与解析不一致

**文件**: `consolidator.py` L429-432  
**问题**: system prompt 要求输出 `{"title": "...", "body_content": "...", "index_update": "...", "log_entry": "..."}`，但实际解析（L941-944）也读这些 key。验证通过。但 JSON schema 缺少 `action` 字段——LLM 不输出 action，由代码决定。这是设计如此（✅）。

### L4. CI 缺少 Python syntax check

**问题**: 三个仓库的 CI 都没有 `python -m compileall` 步骤，语法错误只会在 Docker build 时暴露。  
**修复**: 在 workflow 的 build 步骤前加入：
```yaml
- name: Syntax check
  run: python -m compileall pipeline/ collect.py
```

---

## ✅ 验证通过

- [x] DS-004 已删除 `_find_all_related()`, `_find_relevant_concepts()`, `_build_search_query()` 
- [x] DS-004 `_group_by_tags()` 正确从 `topic/` 前缀提取标签
- [x] DS-001 `write_notes_fs()` 正确注入 `topic/*` 到 frontmatter tags
- [x] DS-001 `process_tags()` 逻辑完整：复用已有标签 → embedding 去重 → 创建新标签
- [x] DS-001 `_cosine_similarity()` 有 dimension mismatch guard
- [x] DS-001 `_kmcp_embed()` 有 fallback（k-mcp 不可用时创建空 embedding 标签）
- [x] k-mcp `embed` endpoint 通过 `indexer.embed_fn(text)` 调用 ONNX 模型
- [x] DS-004 `cosine_similarity()` 通过 k-mcp 客户端调用
- [x] DS-004 `_build_cross_concept_links()` 使用 embedding 相似度 > 0.70
- [x] CI workflow 已配置（k-mcp + ds004 通过；ds001 需加白名单）
- [x] 所有容器不加载 ONNX 模型（embedding 全走 k-mcp）
- [x] DS-004 LLM 调用严格串行
- [x] DS-001 并行度: Pass1=5, Pass2=3, embed calls 串行

---

## 修复优先级

| 优先级 | ID | 描述 |
|--------|-----|------|
| 🔴 P0 | R2 | ds001-pipeline 加 DockerHub secret 白名单 |
| 🔴 P0 | R1 | filter_articles 传 tag_library |
| 🟡 P1 | M1 | Prompt size ≤ 8,000 硬上限 |
| 🟡 P1 | M2 | 每标签最多 15 篇笔记 |
| 🟡 P2 | M3 | Wikilink 去重用正则 |
| 🟡 P2 | M4 | 概念页加 topic/ 标签 |
| 🟢 P3 | L1-L4 | 镜像 tag、syntax check 等 |
