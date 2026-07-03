---
title: LightRAG 四条检索链路：local / global / hybrid / mix 一次讲透
date: 2026-06-24 23:14:00
---

> 假设你已经知道 LightRAG 在做"图增强的 RAG"，本文只聚焦检索链路本身：四种模式怎么走、token 怎么砍、多路结果怎么合并排序、rerank 在哪一步起作用。
>
> 阅读前置：知道有"实体""关系""chunk"三种东西就够了——实体是图里的节点（如 `LightRAG`），关系是图里的边（如 `LightRAG → GraphRAG`），chunk 是文档切片原文。

---

## 一、先把四种模式放在一张图里

LightRAG 的查询模式由 `param.mode` 决定，分四种：

| 模式 | 一句话 | 检索信号从哪来 |
|---|---|---|
| **local** | "我想知道某个具体东西" | 低层关键词（具体名词，如 `LightRAG`） |
| **global** | "我想知道宏观上有哪些关系" | 高层关键词（主题概念，如 `RAG框架`） |
| **hybrid** | "local + global 我都要" | 低层 + 高层关键词两路并发 |
| **mix** | "hybrid 之外再加一路纯向量检索" | 上述 + 用户原话直接查 chunk |

四种模式都走同一条主管道 `_build_query_context`，分四个阶段：

```
Stage 1：纯检索     → 把原始结果捞回来，不裁剪
Stage 2：token 截断  → 实体/关系按预算砍
Stage 3：chunk 合并  → 用实体/关系反查原文 chunk 并合并
Stage 4：格式化      → 动态预算 + rerank + 送给 LLM
```

下面先用一个固定例子贯穿全文：

> **用户问题**："LightRAG 如何做图检索增强？"
> **LLM 抽出的关键词**：
> - 低层（ll）：`["LightRAG", "知识图谱"]`
> - 高层（hl）：`["图检索增强", "RAG框架"]`

---

## 二、Stage 1：四条链路各自怎么捞数据

### 2.1 local：从"实体"入手

local 的检索信号是**低层关键词**（具体名词）。流程：

```
低层关键词 embedding
  → 实体向量库 entities_vdb 捞 top_k=40 个候选实体名
  → 图 knowledge_graph 取每个实体的完整属性 + 度数
  → 顺着每个实体的边，捞"一跳邻居关系"
```

**为什么向量库和图都要查？** 两者维度不同：

- 向量库负责"语义召回"——哪些实体的描述跟关键词最像
- 图负责"拓扑补全"——这些实体之间到底什么关系、谁连谁

数据结构产物（举例）：

```python
# 实体（来自向量库 + 图属性）
final_entities = [
    {"entity_name": "LightRAG", "entity_type": "框架",
     "description": "基于图的RAG框架",
     "source_id": "chunk_01<SEP>chunk_05",   # 这个实体在哪些chunk出现过
     "rank": 120},                            # 图里的度数
    {"entity_name": "知识图谱", "rank": 98, ...},
    # ... 约 35 个
]

# 关系（一跳边，按 rank+weight 排序）
final_relations = [
    {"src_id": "LightRAG", "tgt_id": "GraphRAG",
     "weight": 5.0, "rank": 8,
     "description": "LightRAG借鉴了GraphRAG的思想",
     "source_id": "chunk_02<SEP>chunk_09"},
    # ...
]
```

注意 `source_id` 用 `<SEP>` 分隔多个 chunk_id——这是后面 Stage 3 反查原文的钥匙。

### 2.2 global：从"关系"入手，方向完全相反

global 的检索信号是**高层关键词**（主题概念）。流程：

```
高层关键词 embedding
  → 关系向量库 relationships_vdb 捞 top_k=40 个候选边
  → 图里取每条边的完整属性
  → 把边的 src/tgt 端点反查成实体
```

**为什么从关系入手？** 高层关键词描述"主题"，而图里最接近主题级信息的载体是**边**——边描述两个实体的联系，抽象层次比单个实体高。global 召回的主菜是边，实体只是副产品。

```python
# 关系（保持向量相似度顺序，不重排！）
final_relations = [
    {"src_id": "LightRAG", "tgt_id": "GraphRAG", "weight": 5.0, ...},  # 相似度 0.88
    {"src_id": "Neo4j",    "tgt_id": "知识图谱",  "weight": 3.0, ...},  # 相似度 0.85
    # ...
]

# 实体（从边两端抽名字，去重，顺序跟随边）
final_entities = [
    {"entity_name": "LightRAG", ...},
    {"entity_name": "GraphRAG", ...},
    {"entity_name": "Neo4j", ...},
]
```

### 2.3 local 和 global 的关键不对称设计

这是理解后面 hybrid 合并的前提，必须记牢：

| | 实体排序依据 | 关系排序依据 |
|---|---|---|
| **local** | 向量相似度降序（语义） | `(rank度数, weight权重)` 降序（结构） |
| **global** | 跟随边序（不重排） | 向量相似度降序（语义） |

**为什么相反？**

- local 是"围绕一个具体实体展开" → 实体是检索入口必须靠语义找对锚点；关系是上下文，已经锚定后自然看重结构（谁最核心）
- global 是"宏观上哪些主题关系相关" → 关系靠语义贴合度；实体只是关系的注脚，跟随边出现就行

举例对比：

```
local 的实体:  [LightRAG, GraphRAG, 知识图谱]   ← 按"和'LightRAG'这个词的相似度"排
local 的关系:  [LightRAG→GraphRAG, LightRAG→知识图谱]  ← 按度数+权重排，最核心的在前

global 的关系: [LightRAG→GraphRAG, Neo4j→知识图谱]   ← 按"和'RAG框架'这个词的相似度"排
global 的实体: [LightRAG, GraphRAG, Neo4j]            ← 跟着关系边出现的顺序
```

两路召回的"信息优先级"不同，这正是 hybrid 合并能互补的根本原因。

### 2.4 hybrid：两路都跑，然后 round-robin 去重合并

hybrid 简单粗暴：local 和 global 两路都执行，结果按位置**交替取用**：

```python
for i in range(max(len(local_entities), len(global_entities))):
    if i < len(local_entities):
        if name not in seen: append       # 先放 local 的
    if i < len(global_entities):
        if name not in seen: append       # 再放 global 的
```

举例（实体合并）：

```
local 实体序:   [LightRAG, GraphRAG, 知识图谱]
global 实体序:  [LightRAG, GraphRAG, Neo4j]

i=0: local[0]=LightRAG ✓ 加入
     global[0]=LightRAG ✗ 已见过，跳过
i=1: local[1]=GraphRAG ✓ 加入
     global[1]=GraphRAG ✗ 已见过，跳过
i=2: local[2]=知识图谱 ✓ 加入
     global[2]=Neo4j    ✓ 加入

合并结果: [LightRAG, GraphRAG, 知识图谱, Neo4j]
```

**为什么用 round-robin 而不是直接拼接？**

直接拼接会让前几个位置全是 local 的结果，global 的强信号被挤到后面；交替取用让两路"前几名"均匀铺开，LLM 视野里同时能看到"local 最强 + global 最强"。

**关系去重的小细节**：实体去重用 `entity_name`，关系去重用 `tuple(sorted([src, tgt]))`——也就是 `(A→B)` 和 `(B→A)` 算同一条边，排序后比较。

### 2.5 mix：hybrid 之外再加一路纯向量 chunk

mix 在 hybrid 基础上，**额外**用用户原话直接查一次 chunk 向量库：

```python
vector_chunks = chunks_vdb.query(query_embedding, top_k=20)
# vector_chunks 是用户原话最贴的 chunk，不经过图
```

为什么要有这一路？KG 检索再强也有死角——有些知识不在实体里，只在 chunk 原文里。比如用户问"LightRAG 的安装命令是什么"，这个信息可能就藏在某段 chunk 里，根本没被抽成实体。mix 是"图 + 纯向量"双保险。

每个 chunk 在 mix 里都带一个来源标记 `chunk_tracking`：

```python
chunk_tracking = {
    "chunk_01": {"source": "C", "frequency": 1, "order": 1},  # C=纯向量路
    "chunk_05": {"source": "E", "frequency": 2, "order": 2},  # E=实体反查路，被2个实体共享
    "chunk_09": {"source": "R", "frequency": 1, "order": 3},  # R=关系反查路
}
```

这个标记最后会变成日志 `E5/2 R2/1 C1/1`，意思是"实体路贡献5个chunk其中2个被共享、关系路2个、向量路1个"。

### 2.6 Stage 1 产物小结

| 模式 | final_entities | final_relations | vector_chunks |
|---|---|---|---|
| local | ✓ ~35个 | ✓ 一跳边 | ✗ |
| global | ✓ 跟随边 | ✓ ~40个 | ✗ |
| hybrid | ✓ local+global合并 | ✓ 同上 | ✗ |
| mix | ✓ 同hybrid | ✓ 同上 | ✓ ~20个 |

**所有结果都是原始未裁剪的**——Stage 1 只管捞不管砍。

---

## 三、Stage 2：token 截断，从数量上收敛

### 3.1 为什么要截断？

Stage 1 捞回来的实体可能 35 个、关系可能 40 个，全塞给 LLM 会爆。LightRAG 用**双层预算**：

```
第一层（本阶段）：实体 ≤ 6000 token，关系 ≤ 8000 token
第二层（Stage 4）：三者合计 ≤ 30000 token（兜底）
```

### 3.2 截断算法：从前往后加，超了就停

`_apply_token_truncation` 的逻辑很朴素：**按 Stage 1 的检索顺序，从第一个开始往预算里塞，塞到塞不下就停，后面的全扔**。

关键技巧：**截断前先剥掉 `file_path` 和 `created_at`**——这俩字段占 token 但对判断相关性没用。

举例（实体截断，假设预算 6000 token）：

```
Stage 1 的实体顺序: [LightRAG, GraphRAG, 知识图谱, Neo4j, ...] 共 35 个
剥掉 file_path/created_at 后估算 token：
  LightRAG: 180t → 累计 180t  ✓ 留下
  GraphRAG: 150t → 累计 330t  ✓ 留下
  知识图谱: 200t → 累计 530t  ✓ 留下
  ...
  第 19 个: 累计 5950t → 还能塞
  第 20 个: 累计 6100t → 超了！停，第 20 个及之后全扔
最终留下 19 个实体
```

### 3.3 双轨结构：这是全文最核心的设计模式

截断后 LightRAG 重建出**两条并行的数据轨道**：

```python
{
  "entities_context": [       # 轨道1：精简视图，送给 LLM
    {"entity": "LightRAG", "type": "框架", "description": "...", "created_at": "..."},
    # 19 个，字段精简
  ],
  "filtered_entities": [      # 轨道2：原始 dict，供 Stage 3 反查 chunk
    {"entity_name": "LightRAG", "source_id": "chunk_01<SEP>chunk_05", ...},
    # 19 个，保留 source_id
  ],
  "entity_id_to_original": {  # 名字 → 原始 dict 的映射表
    "LightRAG": {完整原始dict},
    ...
  }
}
```

**为什么要两条轨？**

- LLM 推理只需要 `{name, type, description}`，多了 `source_id` 反而干扰
- 但 Stage 3 要从实体反查原文 chunk，必须保留 `source_id`

所以"精简视图送 LLM、原始数据做检索"，两条轨通过实体名关联。关系同理，去重键是 `tuple(sorted([src, tgt]))`。

### 3.4 四条链路的截断完全一样

注意：**local / global / hybrid / mix 在 Stage 2 的截断逻辑完全相同**——都是同一个函数、同样的预算、同样的双轨重建。

差别只在于"输入是什么"：

- local：截断 local 的实体和 local 的关系
- global：截断 global 的实体和 global 的关系
- hybrid / mix：截断 **合并后**的实体和关系（已经 round-robin 过的）

所以 hybrid / mix 截断时，**保留的是两路交织后的前 N 个**——local 和 global 的强信号都被均匀保留。

---

## 四、Stage 3：chunk 合并，找回原文证据

### 4.1 怎么从实体/关系找到原文 chunk？

每个实体和关系都带 `source_id`，长这样：`"chunk_01<SEP>chunk_05<SEP>chunk_09"`。拆开就是候选 chunk_id 集合：

```
实体 LightRAG.source_id = "chunk_01<SEP>chunk_05"
  → 候选 chunk_id = {chunk_01, chunk_05}

实体 知识图谱.source_id = "chunk_05<SEP>chunk_09"
  → 候选 chunk_id = {chunk_05, chunk_09}

合并候选 = {chunk_01, chunk_05, chunk_09}
```

但候选可能很多（十几个实体 × 每个3-5个chunk = 几十个候选），不能全要。LightRAG 提供两种选择策略。

### 4.2 两种 chunk 选择策略

**WEIGHT 法**（按实体重要度梯度分配）：

把实体按 rank 排序，最重的实体分最多 chunk 配额，最轻的保底 1 个：

```
实体按 rank 排序: [LightRAG(120), 知识图谱(98), GraphRAG(80)]  共 3 个
配额上限 related_chunk_number = 5
线性插值: [5, 3, 1]   ← 最重拿5个，最轻拿1个

LightRAG 从它的候选 {chunk_01, chunk_05, chunk_11} 拿前 5 个
知识图谱 从它的候选 {chunk_05, chunk_09} 拿前 3 个
GraphRAG 从它的候选 {chunk_02} 拿前 1 个
```

**VECTOR 法（默认）**：按 chunk 和 query 的余弦相似度选。

```python
num_of_chunks = int(5 * 实体数 / 2)   # 比如实体19个 → 选47个候选
chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
similarity = cosine_similarity(query_embedding, chunk_embedding)
selected = sorted(by_similarity_desc)[:num_of_chunks]
```

**为什么默认 VECTOR？** WEIGHT 法完全不看 query，只看实体重要度；VECTOR 法让 chunk 选择受查询语义影响，更贴合"用户到底问什么"。而且复用 Stage 1 算好的 `query_embedding`，不重复算。

**健壮性回退**：VECTOR 法遇到向量缺失、embedding 不可用、选空，**一律回退 WEIGHT**——检索绝不能因为候选策略故障而崩。

### 4.3 关系 chunk 的"减法去重"

关系版比实体版多一步：**先剔除实体已经选过的 chunk**。

```python
entity_chunk_ids = {chunk_01, chunk_05, chunk_09}   # 实体路已选
for chunk_id in relation_source_ids:
    if chunk_id in entity_chunk_ids: continue        # 已送过，跳过
```

**为什么？** chunk 是"证据"，证据只要一份。同一个 chunk 不需要因为"既被实体引用又被关系引用"就送两次——来源路径记在 `chunk_tracking` 里就够了。

### 4.4 三轮 round-robin 合并（mix 特有）

对 mix 模式，现在有三路 chunk：

```
C 路（vector_chunks）: [chunk_01, chunk_04]            ← 用户原话直接查的
E 路（entity_chunks）: [chunk_04, chunk_09, chunk_02]  ← 实体反查的
R 路（relation_chunks）: [chunk_02]                    ← 关系反查的（实体已选的被剔除）
```

`_merge_all_chunks` 按位置交替合并，优先级 **C > E > R**：

```
i=0: C[0]=chunk_01 ✓ 加入
     E[0]=chunk_04 ✓ 加入
     R[0]=chunk_02 ✓ 加入
i=1: C[1]=chunk_04 ✗ 已见过
     E[1]=chunk_09 ✓ 加入
i=2: E[2]=chunk_02 ✗ 已见过

合并结果: [chunk_01, chunk_04, chunk_02, chunk_09]
```

**为什么 C 优先？** C 路是用户原话直接查的，最贴合 query 语义，最先被 LLM 看到效果最好；E 路是实体反查的，次之；R 路是关系反查的，兜底补全。

hybrid 模式没有 C 路，只有 E 和 R 两路交替（E 优先）。

### 4.5 合并后可选：补文档内标题

如果开了 `enable_content_headings`，会反查 `text_chunks_db` 给每个 chunk 补上**文档内标题路径**：

```python
{"content": "LightRAG 是一个...", "content_headings": "Section 1 → 1.2 架构"}
```

帮 LLM 理解这个 chunk 在原文的位置，受 token 预算约束。

---

## 五、Stage 4：动态预算与 rerank

### 5.1 动态预算公式：先量后分

到这一步，实体和关系的内容已经定死（Stage 2 截断完），只有 chunk 是可压缩变量。预算逻辑是**先量已知开销，再把余量全给 chunk**：

```
已知开销：
  sys_prompt_tokens   = 空context的system prompt开销
  kg_context_tokens   = entities + relations 的 JSON 开销（Stage 2 定死）
  query_tokens        = 用户query
  buffer_tokens       = 200（给引用列表和安全余量）

动态余量：
  available_chunk_tokens = 30000 - 上述全部
```

举例：

```
sys_prompt:    500t
kg_context:   8000t（实体19个+关系20个的JSON）
query:         50t
buffer:       200t
─────────────────────
available_chunk_tokens = 30000 - 8750 = 21250t
```

如果实体多、kg_context 涨到 12000t，chunk 预算就自动缩到 17250t。**总预算永不超 30000t**。

### 5.2 五步 chunk 加工流水线（rerank 在这一步！）

`process_chunks_unified` 是**所有模式共用**的五步流水线：

```
1. rerank:    独立rerank模型按query重排 → 附 rerank_score
2. 过滤:      扔 rerank_score < 0.5 的
3. 截断:      超 chunk_top_k=20 就砍
4. token裁:   按动态 available_chunk_tokens 逐条累计裁剪
5. 编号:      每条附 {"id": "DC1", "DC2", ...}
```

### 5.3 rerank 的依据：在相关性维度上重建顺序

**这是回答"四个阶段在哪做重排"的关键**。

先明确：**Stage 1/2/3 的排序都不叫 rerank**——

- Stage 1 的排序是向量相似度或图结构（rank/weight），发生在**单个存储内部**
- Stage 2 截断只按"Stage 1 的顺序从前往后塞"，不重排
- Stage 3 合并是位置级 round-robin，不重排

**真正的 rerank 只发生在 Stage 4 的 Step 1**，由独立的 rerank 模型（如 Cohere）执行：

```python
# 输入：Stage 3 合并后的 chunk 列表（位置级交织序）
[chunk_01, chunk_04, chunk_02, chunk_09, ...]

# rerank 模型对每个 chunk 打分
[{"index": 3, "relevance_score": 0.92},   # chunk_09 最相关
 {"index": 0, "relevance_score": 0.87},   # chunk_01
 {"index": 2, "relevance_score": 0.81},   # chunk_02
 {"index": 1, "relevance_score": 0.45}]   # chunk_04 相关性低

# 输出：按 relevance_score 降序
[chunk_09, chunk_01, chunk_02, chunk_04, ...]
```

**rerank 模型比 cosine 相似度强在哪？**

cosine 只能比较 embedding 向量夹角，是个"粗筛"；rerank 模型是专门训练的 cross-encoder，能把 query 和 chunk 拼在一起做更精细的相关性判断。所以 rerank 一旦开启，**前面 Stage 1/3 的所有相似度排序、round-robin 交织序全部被覆盖**，最终顺序 = rerank 分数降序。

### 5.4 rerank 在四条链路里的作用范围

| 模式 | rerank 作用对象 |
|---|---|
| local | 对 E 路 chunk 重排 |
| global | 对 R 路 chunk 重排 |
| hybrid | 对 E+R 合并后的 chunk 重排 |
| **mix** | **对 C+E+R 三路合并后的 chunk 统一重排** |

mix 的 rerank 最关键——因为 C/E/R 三路来源不同、内部排序基准不同（C 是向量相似度、E 是 VECTOR 法或 WEIGHT 法、R 同 E），只有 rerank 能在统一的相关性维度上把它们压平成一条序。

**没开 rerank 时怎么办？** 就用 Stage 3 的合并交织序：

- local：E 路内部序（VECTOR 法的余弦序，或 WEIGHT 法的梯度序）
- global：R 路内部序
- hybrid：E/R 交替序（E 优先）
- mix：C>E>R 交替序（C 优先）

### 5.5 引用编号的公平性

按"被引用次数"给文档排序编号：被最多 chunk 引用的文档编号靠前，LLM 优先采信 `[1]`。

```
doc1.pdf 被 3 个 chunk 引用 → reference_id = 1
doc2.pdf 被 1 个 chunk 引用 → reference_id = 2
```

### 5.6 最终送给 LLM 的上下文结构

`kg_query_context` 模板把三块知识装进结构化 JSON：

````text
---Knowledge Graph Data (Entity)---
```json
{"entity": "LightRAG", "type": "框架", "description": "...", "created_at": "..."}
```
---Knowledge Graph Data (Relationship)---
```json
{"entity1": "LightRAG", "entity2": "GraphRAG", "description": "..."}
```
---Document Chunks---
```json
{"reference_id": "1", "content": "LightRAG是...", "content_headings": "Section 1 → ..."}
```
---Reference Document List---
[1] doc1.pdf
[2] doc2.pdf
````

**为什么用 JSON 而非自然语言？** 结构化 JSON 让 LLM 精确解析每个字段，`reference_id` 让它给可追溯出处，`content_headings` 帮它理解 chunk 在原文的位置。检索结果以机器可读结构交付、由 LLM 消化成自然语言。

---

## 六、四条链路全程对比

### 6.1 一个 mix 查询的数据形态演变

```
① query: "LightRAG 如何做图检索增强？"

② 关键词: (ll=["LightRAG","知识图谱"], hl=["图检索增强","RAG框架"])

③ embedding 三元组 (一次批量调用): (query_emb, ll_emb, hl_emb)

④ Stage 1 三路原始检索:
   final_entities ~35个   (local+global 合并去重后)
   final_relations ~40个  (同上)
   vector_chunks  ~20个   (仅 mix)

⑤ Stage 2 token 截断后:
   entities_context 19个 (6000t内)  + filtered_entities 19个原始dict
   relations_context 20个 (8000t内) + filtered_relations 20个原始dict

⑥ Stage 3 chunk 合并去重后:
   merged_chunks ~50个 {content, chunk_id, (content_headings)}
   chunk_tracking 标注来源 E/R/C

⑦ Stage 4 动态预算 + rerank:
   available_chunk_tokens = 30000 - sys_prompt - kg_context - query - 200
   → rerank → 过滤score<0.5 → chunk_top_k=20 → token裁剪 → 15个 {id:"DC1",...}

⑧ 引用编号: 15个chunk带reference_id，references按频次排序

⑨ LLM上下文: kg_query_context模板填充 + raw_data
```

### 6.2 四条链路对比总表

| 阶段 | local | global | hybrid | mix |
|---|---|---|---|---|
| **关键词** | 只用 ll | 只用 hl | ll+hl | ll+hl |
| **Stage 1 实体来源** | entities_vdb | 关系反查 | 两路合并 | 两路合并 |
| **Stage 1 实体排序** | 向量相似度降序 | 跟随边序 | round-robin（local优先）+名字去重 | 同 hybrid |
| **Stage 1 关系来源** | 实体一跳边 | relationships_vdb | 两路合并 | 两路合并 |
| **Stage 1 关系排序** | (rank,weight)降序 | 向量相似度降序 | round-robin + sorted((src,tgt))去重 | 同 hybrid |
| **vector_chunks** | ✗ | ✗ | ✗ | ✓ ~20个 |
| **Stage 2 截断** | 同一函数，预算相同 | 同左 | 截断合并后的结果 | 同 hybrid |
| **Stage 3 chunk 来源** | E 路 | R 路 | E+R 两路 | **C+E+R 三路** |
| **Stage 3 合并序** | E 路内序 | R 路内序 | E/R 交替（E优先） | C>E>R 交替 |
| **Stage 4 rerank 作用对象** | E 路 chunk | R 路 chunk | E+R chunk | **C+E+R 统一** |
| **rerank 开启时最终序** | rerank分数降序 | rerank分数降序 | rerank分数降序 | rerank分数降序 |
| **无 rerank 时最终序** | E 路内序 | R 路内序 | E/R 交织序 | C>E>R 交织序 |

---

## 七、mix 给 LLM 的上下文和前三者一样吗？

一个常见疑问：mix 比 hybrid 多了一路 C（纯向量 chunk），那它最终塞给 LLM 的上下文是不是也比前三者多？

### 7.1 结构一样，但 chunk 来源多样性不同

四种模式最终都套同一个 `kg_query_context` 模板，都是 **Entity / Relationship / Chunks / References** 四块结构。差别只在 chunk 是怎么来的、候选池有多大：

| 模式 | chunk 来源 | Stage 3 合并后候选数 | 最终送 LLM 的 chunk 数 |
|---|---|---|---|
| local | E 路（实体反查） | ~30 | 受 `chunk_top_k=20` 和 token 预算裁剪 |
| global | R 路（关系反查） | ~25 | 同上 |
| hybrid | E + R | ~40 | 同上 |
| **mix** | **C + E + R** | **~50** | **同上** |

关键点：**不管合并了多少候选，最终都过同一个 `process_chunks_unified` 五步流水线**，被 `chunk_top_k=20` 硬上限和动态 token 预算压到十几条。

### 7.2 举例：mix 不是"送更多 chunk"，是"从更多样候选里精选同样数量"

假设四种模式跑同一个 query，Stage 3 合并后的候选数：

```
local:   30 个候选 → rerank → 过滤 → chunk_top_k=20 → token裁 → 最终 12 个
global:  25 个候选 → 同流水线 → 最终 11 个
hybrid:  40 个候选 → 同流水线 → 最终 15 个
mix:     50 个候选 → 同流水线 → 最终 15 个
```

mix 的候选池最大（50），但**出口被同一个 `chunk_top_k=20` 和 token 预算卡住**，所以送进 LLM 的 chunk 数量并不比 hybrid 多多少。**多样性进得来，总量出不去。**

mix 真正多出来的不是 chunk 数量，而是 **C 路这个独立信号源**——用户原话直接查的向量结果，给 rerank 提供了更丰富的候选池，让最终选中的 15 个更可能命中"图检索死角"。

### 7.3 那 mix 到底比 hybrid 强在哪？

不是"信息更多"，而是"**信号更全**"。举个具体场景：

> 用户问："LightRAG 的 pip 安装命令是什么？"

这个信息大概率藏在某段 chunk 原文里，**根本没被抽成实体**（"pip install" 不是实体）。于是：

- hybrid：E+R 两路都查不到 → 这个 chunk 进不来 → LLM 答不出
- mix：C 路用用户原话直接查 chunk 向量库 → 命中那段 chunk → LLM 能答

mix 多出来的不是 chunk 的"数量"，是"**图检索覆盖不到的 chunk 的可达性**"。这才是 mix 的真实价值。

---

## 八、四个排序依据的灵魂总结

回答你最关心的"排序依据"问题，一层一层拆开：

### 8.1 Stage 1 的排序依据：存储内部

| | 实体 | 关系 |
|---|---|---|
| local | 向量相似度（语义） | (度数rank, 权重weight) 双键降序（结构） |
| global | 跟随边序，不排序 | 向量相似度（语义） |

依据是"存储内部天然信号"：向量库给相似度，图给度数和权重。local 实体靠语义找对锚点、关系靠结构展开上下文；global 关系靠语义找对主题、实体跟随。

### 8.2 Stage 2 的排序依据：不变

Stage 2 只截断不排序，**完全沿用 Stage 1 的顺序**从前往后塞 token 预算，塞满就停。

### 8.3 Stage 3 的排序依据：位置级 round-robin

| 模式 | 合并对象 | 优先级 |
|---|---|---|
| hybrid | E + R | E 优先 |
| mix | C + E + R | C > E > R |

依据是"来源贴合度"：C 是用户原话直接查的最贴、E 是实体反查的次之、R 是关系反查的兜底。round-robin 让多路信号在位置维度均匀铺开，不是简单拼接。

### 8.4 Stage 4 的排序依据：rerank 模型的相关性分数

**rerank 一旦开启，前面所有排序全部被覆盖**。rerank 模型是 cross-encoder，把 query 和每个 chunk 拼一起做精细相关性判断，输出 `relevance_score`，按分数降序就是最终顺序。

依据是"独立模型在统一相关性维度上的判断"——比 cosine 相似度更准，比图结构更贴合查询意图。

### 8.5 一句话本质

> local/global 的排序是「存储内部」的排序（相似度或图结构），hybrid/mix 的排序是「序列间」的排序（round-robin 交织），Stage 4 的 rerank 是「相关性维度」的统一重排——把前面所有基于相似度/结构的排序全部压平。

---

## 附：常用默认值速查

来源：`lightrag/constants.py`

| 常量 | 默认值 | 含义 |
|---|---|---|
| `DEFAULT_TOP_K` | 40 | 实体/关系向量检索条数 |
| `DEFAULT_CHUNK_TOP_K` | 20 | chunk 初始检索 / rerank top_n / 硬上限 |
| `DEFAULT_MAX_ENTITY_TOKENS` | 6000 | 实体 token 预算 |
| `DEFAULT_MAX_RELATION_TOKENS` | 8000 | 关系 token 预算 |
| `DEFAULT_MAX_TOTAL_TOKENS` | 30000 | 总 token 预算 |
| `DEFAULT_RELATED_CHUNK_NUMBER` | 5 | 每个实体的 chunk 配额上限 |
| `DEFAULT_KG_CHUNK_PICK_METHOD` | `"VECTOR"` | chunk 选择策略 |
| `GRAPH_FIELD_SEP` | `"<SEP>"` | source_id 的 chunk_id 分隔符 |
| `min_rerank_score` | 0.5 | rerank 过滤阈值（1.0 视为通过） |
