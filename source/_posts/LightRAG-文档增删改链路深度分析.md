---
title: LightRAG 文档增删改三链路：插入 / 删除 / 更新一次讲透
date: 2026-07-03 12:00:00
---

> 上一篇我们聊了 LightRAG 怎么"查"。这篇聊另一面：文档怎么"存、删、改"。
>
> 表面上看，文档不就是一段文本嘛，写入一个 KV 库不就完了？难就难在 LightRAG 是**图增强的 RAG**——你喂进去的一篇文档，会被拆成 chunk、抽成实体和关系、写进图和三个向量库、散落到七八张表里。删一个文档，绝不能简单 `remove(node)`，因为它贡献的实体可能被另外十个文档共享着。
>
> 阅读前置：知道有"文档 doc""切片 chunk""实体 node""关系 edge"四种东西就够了。本文会用一个贯穿全文的例子讲清增删改全流程，并始终围绕一个核心数据结构：**`source_id`**。

---

## 一、先看清"增删改"难在哪

普通 RAG 的文档就是 KV 里一行 `{doc_id: text}`，增删改三件事都是 O(1)。LightRAG 不行，一篇文档入库后会扩散到这么多地方：

```
一篇文档入库后的"分身"
├── full_docs           （整篇原文）
├── text_chunks         （每个 chunk 的原文 + 元数据）
├── chunks_vdb          （每个 chunk 的向量）
├── doc_status          （文档状态机：pending→processing→processed）
├── 知识图 KG            （实体节点 + 关系边）
├── entities_vdb        （实体向量）
├── relationships_vdb   （关系向量）
├── entity_chunks       （实体 → 它出现在哪些 chunk，全量清单）
├── relation_chunks     （关系 → 它出现在哪些 chunk，全量清单）
├── full_entities       （文档 → 它贡献了哪些实体，崩溃恢复用）
└── full_relations      （文档 → 它贡献了哪些关系，崩溃恢复用）
```

于是三件事各有一个核心难点：

| 操作 | 难点 | 一句话 |
|---|---|---|
| **插入** | 同名实体怎么合并？ | 不能覆盖，要去重 + LLM 重摘要；同一文档重复插要做幂等 |
| **删除** | 共享实体怎么删？ | 实体被多个文档引用时，只能"减去我的贡献"，不能整个删 |
| **更新** | 改一小段怎么办？ | LightRAG **没有 update API**，就是"删旧 + 插新"，靠 LLM cache 把成本摊掉 |

贯穿全文的例子：

> **文档 A**：讲 LightRAG，抽出实体 `LightRAG`、`GraphRAG`，关系 `LightRAG→GraphRAG`
> **文档 B**：也提到 LightRAG，同样抽出实体 `LightRAG`
>
> 于是实体 `LightRAG` 的 `source_id` 是 `"chunkA-001<SEP>chunkB-003"`——两个文档共享。
>
> 接下来：插入 B 时怎么和 A 合并？删掉 A 时 `LightRAG` 还能不能留？更新 A 是什么流程？这就是本文要回答的。

---

## 二、灵魂数据结构：source_id 与它的"真相源"

理解增删改，先盯死一个字段：**每个实体节点和关系边上都挂着一个 `source_id`**。

### 2.1 source_id 是什么

`source_id` 是一个用 `<SEP>` 拼接的 chunk_id 列表，记录"这个实体/关系是从哪些 chunk 抽出来的"：

```python
# 知识图里一个实体节点的完整字段（operate.py:2516-2524）
node = {
    "entity_id":   "LightRAG",
    "entity_type": "框架",
    "description": "基于图的RAG框架，借鉴GraphRAG...",   # 可能是 LLM 合并后的单段
    "source_id":   "chunkA-001<SEP>chunkB-003",          # ← 出处指纹
    "file_path":   "docA.pdf<SEP>docB.pdf",              # 同样 <SEP> 拼接
    "created_at":  "2026-08-06T...",
    "weight":      2,
}
```

关系边同理：

```python
# 一条关系边（operate.py:2044-2056）
edge = {
    "source":    "LightRAG", "target": "GraphRAG",
    "weight":    5.0,                    # 各 chunk 贡献的 weight 之和
    "description": "LightRAG 借鉴了 GraphRAG 的思想",
    "keywords":  "借鉴,框架",
    "source_id": "chunkA-002<SEP>chunkA-007",
    "file_path": "docA.pdf",
}
```

`<SEP>` 是常量 `GRAPH_FIELD_SEP = "<SEP>"`（`constants.py:49`）。读取时 `source_id.split("<SEP>")` 还原成列表。

### 2.2 为什么 source_id 是删改的命门

删除文档 A 时，怎么知道实体 `LightRAG` 还能不能留？

> 把 A 的 chunk_id（`chunkA-001`）从 `LightRAG.source_id` 里扣掉，剩 `chunkB-003`——**还有剩就留着，空了才整个删**。

这就是删除的全部核心逻辑，纯字符串集合差，**不调 LLM**。

### 2.3 但是：图上的 source_id 会被截断！

这里有个坑：图节点上的 `source_id` 字段会按 `max_source_ids_per_entity` 截断（KEEP 保头 / FIFO 保尾，`operate.py:2318`）。如果一个实体被 1000 个 chunk 引用，图上只存前 N 个。

所以 LightRAG 另外维护了一张 **`entity_chunks` KV 表，存不被截断的全量清单**：

```python
# entity_chunks 表（lightrag.py:1393-1397）——不被截断的真相源
{
    "LightRAG": {
        "chunk_ids": ["chunkA-001", "chunkB-003", "chunkC-005", ...],  # 全量
        "count": 37,
        "updated_at": "..."
    }
}
```

删除时**优先从 `entity_chunks` 拿全量清单**做减法，图上的 `source_id` 只当备份（`lightrag.py:4968`）。关系同理，有 `relation_chunks` 表，key 是 `make_relation_chunk_key(src, tgt)`——把两端名字排序后拼起来，保证 `A→B` 和 `B→A` 命中同一行。

记住这条主线，后面所有删改逻辑都是围绕它转的。

---

## 三、插入链路：从一段文本到散落七八张表

### 3.1 全流程七步

```
① enqueue        入队 + 文档级去重，写 full_docs / doc_status(pending)
② 状态机调度     pending → parsing → processing
③ chunking       按 token 滑窗切成 N 个 chunk，生成 chunk_id
④ chunk 入库     并行写 chunks_vdb / text_chunks / doc_status(processing)
⑤ 实体关系抽取   每个 chunk 调一次 LLM，抽实体和关系，每条带 source_id=chunk_id
⑥ 图合并         同名实体去重 + LLM 重摘要，合并 source_id，写图 + 实体/关系向量库
⑦ flush + 落态    先持久化全部 storage，再把 doc_status 置为 processed
```

入口有两个：REST `POST /documents/text`（`document_routes.py:5180`）和 SDK `LightRAG.ainsert`（`lightrag.py:1726`）。两者最终都汇入 `apipeline_enqueue_documents` + `apipeline_process_enqueue_documents`（`pipeline.py:604 / 1584`）。

### 3.2 第①步：入队与文档级去重

文档一进来先做三层去重，防止重复插入污染图：

| 层 | 检查内容 | 命中怎么办 |
|---|---|---|
| 入口 | `file_source` 是否已存在（`document_routes.py:5231`） | 直接 409 拒绝，逼你先删 |
| batch 内 | 同 `file_path` / 同 `content_hash`（`pipeline.py:947`） | 标记 `duplicate_kind`，不重抽 |
| 已存库 | `doc_status.filter_keys`（`pipeline.py:1118`） | 记一条 `dup-` 前缀的 FAILED 行，可被 track_id 追踪，但不覆盖原 doc |

关键设计：**重复插入不静默吞掉，而是显式记一条 FAILED**，这样调用方能知道"我插了但被拒了"。

去重读和 upsert 之间用 `enqueue_serialize_lock` 串行化（`pipeline.py:1097`），防止两个并发请求同时 miss dedup。

### 3.3 第③步：切片与 chunk_id 生成

默认按 token 滑窗切（`chunker/token_size.py:130`）：

```
tokens = tokenizer.encode(content)
step = chunk_token_size - chunk_overlap_token_size   # 步长
for start in range(0, len(tokens), step):
    end = min(start + chunk_token_size, len(tokens))
    chunk = tokenizer.decode(tokens[start:end])
```

切出来的 chunk 结构（`TextChunkSchema`，`base.py:79`）：

```python
{
    "tokens": 512,
    "content": "LightRAG 是一个基于图的RAG框架...",
    "full_doc_id": "doc-abc123",       # 所属文档
    "chunk_order_index": 0,            # 在文档里的序号
    # 实际还会附 file_path / llm_cache_list / heading 等
}
```

**chunk_id 怎么生成**（`utils_pipeline.py:152`）有三种策略，优先级从上到下：

```python
if chunking_result 提供了 raw_chunk_id:
    chunk_id = f"{doc_id}-{raw_chunk_id}"
elif 有 chunk_order_index:
    chunk_id = f"{doc_id}-chunk-{order:03d}"      # ← 最常见：doc-abc123-chunk-001
else:
    chunk_id = compute_mdhash_id(f"{doc_id}:{content}", prefix="chunk-")
```

注意 chunk_id 形如 `doc-abc123-chunk-001`——**前缀就是 doc_id**，这是后面"chunk 反查 doc"的钥匙。

### 3.4 第⑤步：实体关系抽取，source_id 就在这里出生

每个 chunk 调一次 LLM（带 `asyncio.Semaphore` 限流，`operate.py:3975`），用抽取 prompt 拿回实体和关系。**关键：每条抽出来的记录，`source_id` 直接赋值为当前 chunk 的 `chunk_key`**（`operate.py:662` 实体 / `736` 关系）：

```python
# 一个 chunk 抽出的实体（operate.py:658-665）
{
    "entity_name": "LightRAG",
    "entity_type": "框架",
    "description": "基于图的RAG框架",
    "source_id": "docA-chunk-001",     # ← 出生即带出处
    "file_path": "docA.pdf",
    "timestamp": "..."
}
```

这就是 source_id 的"出生时刻"——它一开始就是单个 chunk_id，后面合并时才用 `<SEP>` 越拼越长。

LLM 调用走 `use_llm_func_with_cache`，cache key 含 `chunk_id`（`operate.py:3753`）——**同一个 chunk 重抽时直接命中 cache 跳过 LLM**，这是后面"更新"能省钱的基础。

### 3.5 第⑥步：图合并——插入的核心难点

抽出来的实体可能和已存的重名（文档 B 的 `LightRAG` 和文档 A 的 `LightRAG`）。合并发生在 `_merge_nodes_then_upsert`（`operate.py:2229`），分几步：

**a) 读旧值**：`knowledge_graph.get_node("LightRAG")`，把旧的 `description` / `source_id` / `file_path` 按 `<SEP>` 拆开。

**b) source_id 合并 + 写回真相源**：

```python
full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)
# ["chunkA-001"] + ["chunkB-003"] → ["chunkA-001", "chunkB-003"]
entity_chunks.upsert({"LightRAG": {"chunk_ids": full_source_ids, "count": 2}})  # 写全量
source_ids = apply_source_ids_limit(full_source_ids, max_source, method)         # 截断给图用
```

**c) description 去重 + LLM 重摘要**（`operate.py:2394 / 2413`）：

旧描述 `"基于图的RAG框架"` + 新描述 `"一个RAG框架"`，先用 `_combine_descriptions_dedup` 去重（防止重处理时 fragment 无限累积，issue #3367），再交给 `_handle_entity_relation_summary`——如果 fragment 少且 token 不超，直接 `<SEP>` 拼接；超了就用 LLM 总结成一段。

```python
# 合并后写回图节点
knowledge_graph.upsert_node("LightRAG", {
    "description": "基于图的RAG框架，融合知识图谱与向量检索...",  # LLM 摘要后的单段
    "source_id":   "chunkA-001<SEP>chunkB-003",                   # <SEP> 拼接
    "file_path":   "docA.pdf<SEP>docB.pdf",
    ...
})
```

**d) 写实体向量库**（`operate.py:2530`）：向量化的文本是 `f"{entity_name}\n{description}"`，id 是 `compute_mdhash_id(entity_name, prefix="ent-")`。

**e) 跨进程锁**：合并同一个实体时，用 `get_storage_keyed_lock([entity_name])`（`operate.py:3397`）保证并发安全——两个文档同时插入同名实体，不会互相覆盖。

### 3.6 第⑦步：先 flush 再落态，这个顺序不能反

最后一步有讲究（issue #3400）：**先把所有 storage flush 到磁盘，再把 doc_status 置为 `processed`**。

```
✗ 错误顺序：先 processed → 再 flush
   → 崩在中间：状态显示已处理，但数据没落盘 → 检索查不到 → zombie 文档

✓ 正确顺序：先 flush 全部 storage → 再写 processed
   → 崩在中间：最多是"数据已落盘但状态还是 processing"，下次 resume 重跑即可
```

这就是为什么 `pipeline.py:5304-5346` 严格先 `await self._insert_done()` 再 `_upsert_doc_status_transition(PROCESSED)`。

### 3.7 文档状态机

整个插入过程文档状态在变（`base.py:838`）：

```
PENDING → PARSING → ANALYZING(可选) → PROCESSING → PROCESSED | FAILED
 入队     解析原文    多模态VLM        实体抽取      成功       失败
```

状态记录在 `doc_status` 表，数据结构是 `DocProcessingStatus`（`base.py:853`）：

```python
@dataclass
class DocProcessingStatus:
    content_summary: str        # 前100字预览
    content_length: int
    file_path: str              # 规范化 basename
    status: DocStatus
    created_at: str             # ISO 时间戳
    updated_at: str
    track_id: str | None        # 调用方追踪用
    chunks_count: int | None
    chunks_list: list[str] | None    # ← 该文档所有 chunk_id（删除时靠它）
    error_msg: str | None
    metadata: dict               # process_options / kg_write_state / kg_purge journal 等
    content_hash: str | None     # MD5，重复检测用
```

注意 `chunks_list`——删除时第一步就是读它，拿到这个文档贡献的所有 chunk_id。

---

## 四、删除链路：共享实体的三态决策

这是全文最微妙的部分。删文档 A，但 A 的实体 `LightRAG` 还被 B 用着，怎么办？

### 4.1 三态决策：不动 / 重建 / 整删

入口 `LightRAG.adelete_by_doc_id`（`lightrag.py:5265`），核心在 `_purge_kg_contributions`（`lightrag.py:4582`）。对文档 A 贡献的每个实体/关系，做一个**集合减法**然后分三类：

```python
# lightrag.py:4983-4997 的决策逻辑
existing_sources = entity_chunks["LightRAG"].chunk_ids   # ["chunkA-001", "chunkB-003"]
remaining_sources = subtract_source_ids(existing_sources, A的chunk_ids)
# = ["chunkB-003"]   ← 把 A 的 chunkA-001 扣掉

if not remaining_sources:
    # 没剩了 → 整个删
    entities_to_delete.add("LightRAG")
elif remaining_sources != existing_sources:
    # 还有剩但变了 → 保留，但要重建（修 source_id / description / weight）
    entities_to_rebuild["LightRAG"] = remaining_sources
else:
    # 没变化（A 没贡献这个实体）→ 不动
    pass
```

代入我们的例子：

| 对象 | A 删除前 source_id | 扣掉 A 后 | 决策 |
|---|---|---|---|
| 实体 `LightRAG` | `chunkA-001<SEP>chunkB-003` | `chunkB-003` | **重建**（B 还在用） |
| 实体 `GraphRAG` | `chunkA-002` | 空 | **整删**（只有 A 提过） |
| 关系 `LightRAG→GraphRAG` | `chunkA-002` | 空 | **整删** |
| 实体 `Neo4j` | `chunkC-005` | `chunkC-005` | **不动**（A 没贡献） |

`subtract_source_ids`（`utils.py:5342`）就是个 list comprehension，**纯字符串集合差，不调 LLM**：

```python
def subtract_source_ids(source_ids, ids_to_remove):
    removal_set = set(ids_to_remove)
    return [sid for sid in source_ids if sid and sid not in removal_set]
```

### 4.2 "重建"到底重建什么

对于 `entities_to_rebuild`（如 `LightRAG`），不能只改 source_id 了事——因为 `LightRAG` 的 description 当初是 A、B 两段合并 + LLM 摘要出来的，现在 A 没了，描述得重新基于 B 单独生成。

这里调 `rebuild_knowledge_from_chunks`（`operate.py:995`），但**关键设计：它从 LLM cache 里取 B chunk 当初的抽取结果，不重新调 LLM 抽取**（docstring `operate.py:1010`）：

```
LightRAG 现在的 description = "A和B合并后的摘要"
   ↓ 重建
从 cache 取 chunkB-003 当初抽出的原始 description = "一个RAG框架"
   ↓ 用 _handle_entity_relation_summary 重新摘要（带 cache）
新 description = "一个RAG框架"（单个 fragment，可能直接用，无需 LLM）
```

**weight 衰减**也是自然发生的（`operate.py:2018`）：weight 是各 chunk 贡献的 sum，重建时只遍历 `remaining_sources`，被删的 chunk 不参与 sum，weight 自动减小：

```python
weights = []
for rel_data in all_relationship_data:    # 只遍历剩余 chunk
    if rel_data.get("weight"):
        weights.append(rel_data["weight"])
weight = sum(weights) if weights else current.get("weight", 1.0)
# A 删了，A 贡献的 weight 自然不计入 → 这就是"衰减"
```

所以删除里 LLM 的角色很克制：

| 删除子步骤 | 是否调 LLM |
|---|---|
| 三态决策（集合减法） | ❌ 纯字符串 |
| weight 衰减 | ❌ 纯数值 sum |
| 整删对象 | ❌ 直接 remove |
| 重建共享对象的 description | ⚠️ 带缓存地调 summarize，cache 命中则零成本 |

### 4.3 整删：连带残余边一起清

整删一个实体节点时，不能只 `remove_node`——还得先把它身上挂着的残余边一起删掉（`lightrag.py:5148`）：

```
要整删 GraphRAG：
  1. 先找 GraphRAG 上所有残余边 [LightRAG→GraphRAG, GraphRAG→Neo4j]
  2. 删这些边（图 + relationships_vdb + relation_chunks）
  3. 再删 GraphRAG 节点（图 + entities_vdb + entity_chunks）
```

向量库按 id 删，id 构造规则和插入时一致：

| 向量库 | id 构造 |
|---|---|
| `chunks_vdb` | 直接用 chunk_id |
| `entities_vdb` | `compute_mdhash_id(entity_name, prefix="ent-")` |
| `relationships_vdb` | `compute_mdhash_id(src+tgt, prefix="rel-")`，正反向两个 id 都删（无向图） |

### 4.4 安全破坏序：先修图，后删肉

删除的执行顺序很讲究（issue #3400 重写的核心）。原则：**图对象永远不能指向已删除的 chunk**。

```
① 先清图贡献（_purge_kg_contributions）
   ├─ 修图节点/边（重建或整删）
   ├─ 修向量库 entities_vdb / relationships_vdb
   └─ 修 chunk-tracking entity_chunks / relation_chunks
② 再删 chunk 本体
   ├─ chunks_vdb.delete(chunk_ids)
   └─ text_chunks.delete(chunk_ids)
③ 再删恢复锚点 full_entities / full_relations
④ 最后删 doc_status → full_docs
```

为什么是这个顺序？如果先删 chunk 再修图，中途崩溃会出现"图上的 `source_id` 指向一个已不存在的 chunk"——检索时反查原文会失败。先修图，图就永远是干净的。

还有个小细节：`doc_status` 比 `full_docs` 先删（`lightrag.py:5764`）——如果 `full_docs.delete` 失败，retry 时找不到 doc_status 就把文档当已删，避免 zombie。

### 4.5 删除的状态机：可断点续跑

删除是个长流程（要重建多个共享实体），中途崩溃怎么办？LightRAG 在 `doc_status.metadata.kg_purge` 里写了个 journal，四阶段（`constants.py:317`）：

```
prepared            proof 校验完，还没删任何东西
derived_committed   图/vdb/tracking 已清
anchors_pending     chunk 已删，锚点待删
completed           锚点已删，等收尾
```

每完成一阶段才写下一阶段；retry 时 `phase_at_least` 跳过已完成阶段。这让删除**幂等且可恢复**——崩了重试不会重复删、不会漏删。

### 4.6 并发互斥

删除会抢 pipeline 的 `busy` 槽（`lightrag.py:5345`）：

- 单文档删除：`job_name = "Single document deletion"`
- 批量删除：`job_name` 以 `"deleting"` 开头，可以 join 队列
- 其它写操作（插入）在 busy 时直接 403

这样删除和插入不会同时跑，避免"我正在删 A，你又在插 A"的乱局。

---

## 五、更新链路：删旧 + 插新，靠 cache 伪装增量

### 5.1 LightRAG 没有 update API

这是首先要澄清的：**LightRAG 没有显式的 `aupdate_by_doc_id`**。文档级"更新"在产品层就是两步走：

```
① DELETE /documents/delete_document   删旧
② POST  /documents/text               用新内容重插
```

硬证据在 `document_routes.py:5234`——如果 `file_source` 已存在，直接 409 拒绝重插，detail 里写："Delete the existing record before re-inserting."

那改一小段怎么办？答案是：**全量重切 chunk + 全量重抽实体**，没有 chunk diff。

### 5.2 为什么"删旧+插新"没想象中贵

全量重抽听起来很贵，但有两个机制把成本摊掉：

**机制一：LLM response cache 命中**

重插时 chunk_id 形如 `doc-xxx-chunk-001`——**只要 chunk 内容没变，chunk_id 就一样**（内容 hash 决定），LLM cache key 也一样。于是：

```
原文档 100 个 chunk，你只改了第 5 个
重插时：
  chunk 1-4, 6-100：内容没变 → chunk_id 不变 → cache 命中 → 跳过 LLM
  chunk 5：内容变了 → cache miss → 真正调 LLM 抽取
```

所以"改一小段"的实际 LLM 成本 ≈ 只抽那一个 chunk。

**机制二：删除时默认保留 LLM cache**

`adelete_by_doc_id` 有个参数 `delete_llm_cache`，**默认 false**（`lightrag.py:5548`）。也就是说删文档时 LLM cache 不清，重插时能命中。

> 这就是 LightRAG 把"删旧+插新"伪装成"增量更新"的核心优化点：**用 LLM cache 把全量重抽的实际成本降到接近增量**。

### 5.3 唯一的"细粒度"机制：custom chunks 追加

如果真要增量，有 `ainsert_custom_chunks`（`lightrag.py:1810`，issue #3400 Phase 3）的 patch 模式：对已 PROCESSED 的文档可以**追加** chunks 而非重建。

但注意限制：**只能追加，不能修改、不能删除单个 chunk**。所以它适合"给文档补一段"的场景，不适合"改一段"。

### 5.4 实体描述的合并：插入走合并，删除走重建

更新涉及"删了再插"，所以实体描述会被处理两次，走两条不同的路：

| 场景 | 路径 | LLM 调用 |
|---|---|---|
| 插入时遇到同名实体 | `_combine_descriptions_dedup` 去重 + `_handle_entity_relation_summary` 合并 | fragment 少则不调，多则带 cache 调 |
| 删除时共享实体要重建 | `rebuild_knowledge_from_chunks` 从 cache 取剩余 chunk 原始描述 | 带缓存调 summarize，命中则零成本 |

两条路都靠 `_combine_descriptions_dedup`（`operate.py:2184`）保证 fragment 不重复累积——这是 issue #3367 的修复点，防止反复 reprocess 同一文档让 description 越滚越长。

### 5.5 手工编辑知识图（不是文档编辑）

顺带一提，LightRAG 有 `aedit_entity` / `aedit_relation` / `acreate_entity` / `amerge_entities`（`lightrag.py:6200+`，实现在 `utils_graph.py`）。但这是**手工编辑图节点本身**（比如修正一个实体的 type、合并两个重复实体），跟"文档内容更新触发自动重抽"是两回事。它们绕过 pipeline busy 检查，直接改图。

---

## 六、三链路对比总表

| 维度 | 插入 | 删除 | 更新 |
|---|---|---|---|
| **入口** | `ainsert` / `POST /text` | `adelete_by_doc_id` / `DELETE` | 无显式 API = 删 + 插 |
| **核心难点** | 同名实体合并 | 共享实体不误删 | 成本控制 |
| **关键数据结构** | `source_id` 出生 + 合并 | `source_id` 减法 + `entity_chunks` 真相源 | LLM cache key = chunk_id |
| **LLM 角色** | 抽实体 + 摘要合并 | 仅重建共享对象时带 cache 调 summarize | 重抽未变 chunk 命中 cache 跳过 |
| **状态机** | `pending→processing→processed` | `kg_purge` 四阶段 journal | 复用前两者 |
| **并发控制** | per-entity keyed lock | pipeline busy + per-entity lock | 复用前两者 |
| **幂等保证** | 三层去重 + LLM cache | journal 断点续跑 + recovery anchor | LLM cache 命中 |
| **flush 顺序** | 先 flush 再 processed | 先修图后删肉 | — |
| **失败可恢复** | `kg_write_state` + recovery anchor | `kg_purge` journal 阶段跳过 | 重试即重插 |

---

## 七、灵魂总结：source_id 串起一切

回到开头的问题——增删改到底在干什么？一句话：

> **增删改本质上都是在维护 `source_id` 这个"出处指纹"，让它永远准确反映"每个实体/关系还活在哪些 chunk 上"。**

- **插入**：source_id 出生（=chunk_id），合并时 `<SEP>` 越拼越长，去重防止 description 累积
- **删除**：source_id 做集合减法，剩多少决定"重建/整删/不动"，图上 source_id 永远指向活着的 chunk
- **更新**：删旧（减法）+ 插新（出生），中间靠 LLM cache（key=chunk_id）把全量重抽摊成增量

而 `entity_chunks` KV 表作为不被截断的"真相源"，是删除时判断共享关系的命门——图上的 source_id 可能被截断骗你，但 `entity_chunks` 永远存全量。

三个排序依据也值得记牢：

| 链路 | "顺序"的灵魂 |
|---|---|
| 插入 flush | 先落盘再置 processed——避免"已处理但没数据"的 zombie |
| 删除执行 | 先修图后删肉——图对象永不指向已删 chunk |
| 更新成本 | LLM cache key = chunk_id——内容不变则不调 LLM，全量重抽≈增量 |

理解了 source_id 和它背后的真相源，LightRAG 的增删改就不再是一团乱麻——它们都是同一根线串起来的：**让出处始终可追溯、让共享始终被尊重、让崩溃始终可恢复。**

---

## 附：关键代码位置速查

| 关注点 | 位置 |
|---|---|
| `GRAPH_FIELD_SEP = "<SEP>"` | `constants.py:49` |
| `DocStatus` 状态枚举 | `base.py:838` |
| `DocProcessingStatus` 数据结构 | `base.py:853` |
| `TextChunkSchema` | `base.py:79` |
| 插入入口 `ainsert` | `lightrag.py:1726` |
| chunk_id 生成 | `utils_pipeline.py:152` |
| chunk 切片 `chunking_by_token_size` | `chunker/token_size.py:130` |
| 实体抽取 `extract_entities` | `operate.py:3578` |
| source_id 出生 | `operate.py:662`（实体）/ `736`（关系） |
| 图合并 `_merge_nodes_then_upsert` | `operate.py:2229` |
| description 去重 `_combine_descriptions_dedup` | `operate.py:2184` |
| source_id 截断 `apply_source_ids_limit` | `operate.py:2318` |
| 删除入口 `adelete_by_doc_id` | `lightrag.py:5265` |
| 删除核心 `_purge_kg_contributions` | `lightrag.py:4582` |
| 三态决策 | `lightrag.py:4983` |
| source_id 减法 `subtract_source_ids` | `utils.py:5342` |
| 重建 `rebuild_knowledge_from_chunks` | `operate.py:995` |
| kg_purge 四阶段 journal | `constants.py:317` |
| 重复插入拒绝（409） | `document_routes.py:5234` |
| custom chunks 追加 `ainsert_custom_chunks` | `lightrag.py:1810` |
