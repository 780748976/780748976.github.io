---
title: RAG 评估入门：ragas 指标体系与接入方法
date: 2026-07-23 20:22:00
---

> 前面几篇我们聊了检索侧的 Query Rewrite、知识库的动态更新、生成侧的幻觉治理——这些方案都在「做」：让 RAG 更好。但有个问题一直没回答：**做了这么多改进，怎么知道效果好不好？** 这篇聊怎么「量」。
>
> 一句话版本：**ragas 是当前最主流的 RAG 评估开源框架，核心思路是「用 LLM 当裁判」，把 RAG 拆成「检索质量」和「生成质量」两个环节分别打分。四大经典指标各管一件事：Faithfulness 管生成有没有照着上下文说，Answer Relevancy 管有没有答非所问，Context Precision 管召回的排序对不对，Context Recall 管该找的内容找全没有。接入只需三步：准备一条评估数据（问题 / 答案 / 上下文 / 标准答案）→ 配一个评估用的 LLM → 调 `evaluate()` 跑出分数。**

---

## 一、为什么 RAG 不能用 BLEU / ROUGE 来评估

评估一个 RAG 系统，最直觉的做法是拿它的回答和标准答案比一比，像机器翻译那样算个 BLEU。但这套传统指标在 RAG 场景下基本失效，原因有两个：

**1. BLEU/ROUGE 比的是「文本长得像不像」，RAG 要的是「事实对不对」。**

标准答案写「退款政策是 7 天无理由」，RAG 回答「客户可在收货后 7 日内申请全额退款」。这句话和标准答案用词完全不同，BLEU 会给出很低的分数——但它**事实完全正确**。反过来，RAG 把标准答案原封不动抄一遍，但答错了问题（问 A 答 A 的模板），BLEU 反而给高分。RAG 的质量核心是**语义和事实**，不是字面重合度。

**2. RAG 是两段式结构，只评最终答案没法定位问题。**

```
用户提问
  → ① 检索层：从向量库找相关 chunk
  → ② 生成层：LLM 基于 chunk 生成答案
```

如果答案质量差，到底是**检索没找对**还是**生成没用好**？只看最终答案无法回答。RAG 评估必须拆开：**检索质量一套指标，生成质量一套指标**，出了问题才能定位到具体环节——这正好接上我们聊过的幻觉治理：治幻觉先治检索，但「检索到底烂不烂」，得有一个量化的尺子。

**ragas 就是这把尺子。** 它是一个开源评估框架，思路是：与其用 n-gram 算相似度，不如**让一个 LLM 来当裁判**，用「提问 + 推理」的方式给每个环节打分。指标本身是 LLM 判断出来的，所以能理解语义、能识别事实错误。代价是评估过程要消耗 LLM 调用——这个后文细说。

---

## 二、先认识评估数据的四个字段

任何 RAG 评估，数据本质上都是「一条问答记录」加上「检索和生成过程中的痕迹」。ragas 的最小评估数据长这样：

| 字段 | 含义 | 必须吗 |
|---|---|---|
| `question` | 用户问了什么 | 必须 |
| `answer` | RAG 系统给出的回答 | 必须 |
| `contexts` | 检索召回、喂给 LLM 的 chunk 列表 | 必须 |
| `ground_truth` | 人工标注的标准答案 | 看指标 |

**`ground_truth` 决定你能算哪些指标**：Faithfulness、Answer Relevancy 不需要标准答案（有 question + answer + contexts 就能算）；Context Precision、Context Recall、Answer Correctness 需要。这个映射在第三节讲指标时展开。

注意新旧版本的字段名：老版本叫 `question/answer/contexts/ground_truth`，新版本（v0.2+）统一改成了 `user_input/response/retrieved_contexts/reference`，**内容一样、只是改了名**。看老教程时别对不上。

---

## 三、四大经典指标：每管一件事

ragas 的指标很多，但核心是这四个，也是绝大多数评估的基本盘。逐个说清楚「它衡量什么 + 怎么算的」。

### 指标一：Faithfulness（忠实度）—— 生成层有没有幻觉

**衡量的是：回答里的每一句话，是不是都能在检索到的上下文里找到依据。** 这正是幻觉治理那篇说的「生成层超发挥」——上下文里没有的信息，LLM 自己脑补出来了。

计算原理分三步：

```
① 把答案拆成多个独立论断（statements）
   「这家店支持 7 天无理由退款」→ 论断1：「支持退款」论断2：「期限 7 天」
② 对每个论断单独问裁判 LLM：这个论断能否从上下文中推断出来？（是/否）
③ 得分 = 能被支持的论断数 / 论断总数
```

一句话理解：**答案里有多少比例的话是「有据可查」的。** 1.0 意味着每句话都出自上下文，0.5 意味着有一半论断是编的。

### 指标二：Answer Relevancy（答案相关性）—— 有没有答非所问

**衡量的是：回答和问题到底有多相关。** 一个回答可以很「忠实」（每句都有依据），但完全答非所问——比如用户问「退款政策」，系统答了「物流时效」。

计算原理是反着推：

```
① 把答案喂给裁判 LLM，让它反推：这个答案可能在回答什么问题？
   （会生成 N 个候选问题）
② 把 N 个候选问题和用户原始问题分别算 embedding 相似度
③ 取平均 → 得分
```

一句话理解：**系统说的和用户问的是不是一回事。** 有个常见坑：只问「某产品支不支持 A 功能」，RAG 答了 A 功能的完整介绍但没回答「支不支持」——embedding 相似度可能不低，但答案是无效的。所以这个指标适合粗筛，不适合当唯一标准。

### 指标三：Context Precision（上下文精度）—— 召回的排序对不对

**衡量的是：检索结果里，真正相关的 chunk 是不是排在了前面。** RAG 通常把 top-K 全部塞给 LLM，但排在前面（最被重视）的必须是相关的内容；如果相关 chunk 排在末尾、前面全是噪声，LLM 更容易被误导。

计算原理带位置加权：

```
① 对每个位置的 chunk，问裁判 LLM：这个 chunk 和问题 / 标准答案相关吗？
② 从位置 1 开始，逐个位置计算「已见 chunk 中的相关比例」再累加
③ 相关且排在前面的 chunk 贡献大，相关但排在后面的贡献小
```

**注意：这个指标需要 `ground_truth`**——因为「这个 chunk 到底相不相关」得有个参照标准。

一句话理解：**好的检索 = 相关的内容排在前面。** 如果你在调 embedding、调重排（Rerank），这个指标就是最直接的效果度量。

### 指标四：Context Recall（上下文召回）—— 该找的内容找全没

**衡量的是：标准答案里的信息点，检索结果覆盖了多少。** 排序再好，如果关键信息根本没被召回，LLM 也无米下锅——这就是幻觉治理里说的「检索层没找到，LLM 靠猜」。

```
① 把标准答案（ground_truth）拆成若干句子
② 对每个句子问裁判 LLM：这个句子的信息能否从检索上下文中推断出来？
③ 得分 = 能被覆盖的句子数 / 总句数
```

一句话理解：**知识库里明明有答案，检索器找到没有。** 同样需要 `ground_truth`。如果你怀疑「问题答不上来是因为检索没召回」，先看这个指标。

### 四个指标的分工，画一张图

```
                    RAG 评估
                       │
      ┌────────────────┴────────────────┐
      ▼                                 ▼
  检索质量（四步里①步）               生成质量（四步里②步）
      │                                 │
      ├─ Context Precision              ├─ Faithfulness
      │    相关 chunk 排序对不对？       │    回答有没有照着上下文说？
      ├─ Context Recall                 └─ Answer Relevancy
      │    该找的内容找全没？               回答有没有答非所问？
      └─ （需要 ground_truth）
```

| 指标 | 衡量环节 | 需要字段 | 高分意味着 |
|---|---|---|---|
| Faithfulness | 生成 | question + answer + contexts | 回答有据可查，不编 |
| Answer Relevancy | 生成 | question + answer + contexts | 回答切题，不跑偏 |
| Context Precision | 检索 | + ground_truth | 相关的排在前面 |
| Context Recall | 检索 | + ground_truth | 关键信息都召回了 |

---

## 四、扩展指标：按需加装

四大指标是基本盘，ragas 还提供了一批扩展指标，对应更具体的诉求：

| 指标 | 衡量什么 | 什么时候用 |
|---|---|---|
| Answer Correctness | 回答与标准答案的「事实一致性 + 语义相似度」 | 有标准答案，想综合评答案质量 |
| Context Entity Recall | 上下文覆盖了标准答案里多少「实体」（人名、数字等） | 关注实体级信息是否召回（如医疗、法务） |
| Noise Sensitivity | 往上下文里混入无关 chunk 后，答案质量下降多少 | 检验系统对噪声的鲁棒性 |
| Context Utilization | 检索到的上下文里，真正被答案用上的比例 | 判断召回是否「多而杂」 |
| LLM Context Recall | 基于 LLM 判定信息覆盖的召回变体 | 标准答案拆句效果不好的时候 |

另外还有 `AspectCritique`——**自定义评估标准**：比如「回答是否包含广告」「语气是否礼貌」，你可以自己定义一条评判标准，让裁判 LLM 按它打分。相当于给评估框架插你自己的维度。

> 一个朴素的经验：**第一轮评估只用四大指标就够了**，跑出分数、定位到是检索问题还是生成问题，再有针对性地加扩展指标。指标不是越多越好——每个指标都是一次 LLM 调用，是有成本的。

---

## 五、接入：三步跑通一次评估

说了这么多原理，落地只需要三步。

### 第一步：安装 + 配一个「评估用 LLM」

```bash
pip install ragas
```

关键认知：**ragas 的裁判是一个独立配置的 LLM，不是你 RAG 应用里那个生成模型。** 基本要求是「裁判要比被评的系统强」——用 7B 小模型去评 GPT 的输出，分数会失真。常见配置是让裁判用 `gpt-4o-mini` 级别以上的模型：

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# 裁判 LLM：评估过程中的提问、判断都靠它
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# 部分指标（如 Answer Relevancy）还需要 embedding 计算相似度
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
```

### 第二步：准备评估数据集

三种来源，按成本从低到高：

**① 手动构造**：适合起步。几十条就够了，字段就是第二节那张表：

```python
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

dataset = [
    SingleTurnSample(
        user_input="退款期限是多久？",                       # question
        response="支持 7 天无理由退款。",                     # answer
        retrieved_contexts=[                                 # contexts
            "本店支持 7 天无理由退款，以签收之日起算。",
            "发货时效为 48 小时内。",
        ],
        reference="7 天无理由退款。",                        # ground_truth
    ),
    # ... 再多来几十条
]

eval_dataset = EvaluationDataset(samples=dataset)
```

**② 从你的 RAG 应用采集**：把线上或测试时跑过的真实问题收集起来，`question` 和 `contexts` 直接从你的系统日志里拿（检索结果、喂给 LLM 的 chunk），`answer` 是系统当时的输出，`ground_truth` 人工标一下。**这是最推荐的来源——测的是真实分布。**

**③ TestsetGenerator 合成**：没有现成测试集时，让 ragas 根据你的知识库文档自动生成「问题 + 标准答案」：

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,          # 用来出题的模型
    embedding_model=generator_embeddings,
)
testset = generator.generate_with_langchain_docs(docs, testset_size=50)
testset.to_pandas()
```

它能按比例生成不同类型的题：`simple`（简单直问）、`multi_context`（需要跨多段拼接）、`reasoning`（需要推理）。**注意合成测试集 ≠ 真实分布**，只能作为冷启动，上线前务必用真实数据补一批。

### 第三步：跑评估

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

# 看分数
result.to_pandas()
# 每个指标一个 0~1 的分数，还有一个全局的 RAGAS 综合分
```

输出示例：

| | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| 样例1 | 1.0 | 0.86 | 0.72 | 0.83 |
| 样例2 | 0.50 | 0.91 | 0.64 | 1.0 |
| 全局 | **0.78** | **0.89** | **0.68** | **0.90** |

怎么解读？**别只盯综合分，要看指标之间的「分差」**——分差才是定位问题的地方：

- `faithfulness` 低但 `context_recall` 高 → **检索没问题，是生成没照着说** → 去调 Prompt 约束（幻觉治理的防线一）
- `context_recall` 低 → **该找的没找到** → 去调 embedding / chunk 切分 / 召回数（先治检索）
- `context_precision` 低但 `context_recall` 高 → **找全了但排序乱** → 上 Rerank 重排
- `answer_relevancy` 低 → **答非所问** → 检查问题理解和提示词

这和前几篇是闭环的：**治理方案告诉你「出了问题怎么改」，评估告诉你「问题到底出在哪」。**

---

## 六、更进一步：在线评估与新版写法

### 在线评估：把评估挂到线上流量上

离线评估是拿固定测试集跑一次。想持续监控线上质量，思路是**把每次线上请求的痕迹（question / contexts / answer）记下来，定时批量跑一轮指标**。现在主流链路是「trace 平台 + ragas」：

- **Langfuse / LangSmith**：RAG 应用接入后自动记录每次请求的检索结果和回答，ragas 直接消费这些 trace 批量打分，结果回写到平台看趋势
- **Arize Phoenix**：基于 OpenTelemetry，评估分数可以作为 span 注解回传，在链路视图里直接看到「哪一次检索拖了后腿」

在线评估的取舍：不需要 `ground_truth` 的指标（Faithfulness、Answer Relevancy）可以全量跑；需要标准答案的指标只能抽样子集人工标。

### 新版写法：@experiment

ragas v0.4 起官方推荐用 `@experiment` 装饰器替代 `evaluate()`（后者已标记废弃但仍在广泛使用）。好处是把「评估」定义成一个普通函数，对数据集自动迭代、自动落库，便于和实验管理打通：

```python
from ragas import experiment

@experiment
async def my_eval(user_input, response, retrieved_contexts, reference):
    # 自定义评估逻辑：跑指标、算分
    return {"faithfulness": score.faithfulness, ...}

await my_eval.arun(dataset=eval_dataset)
```

如果你刚上手，用 `evaluate()` 完全没问题，先跑通再说。

---

## 七、常见坑

**坑 1：裁判模型太弱。** 用和被评系统同级的模型当裁判，等于「让差生给差生改卷」——Faithfulness 该判「没依据」的判成「有依据」，分数虚高。原则：**裁判 ≥ 被评模型的水平**，必要时抽 20 条人工复核裁判的判分是否合理。

**坑 2：指标和字段不匹配。** 没有 `ground_truth` 却想算 Context Precision，跑出来的分数毫无意义。先对表：需要标准答案的三个指标（Context Precision / Context Recall / Answer Correctness）必须有 `reference` 字段。

**坑 3：测试集太小或太「干净」。** 20 条全是简单直问，分数好看但说明不了问题。至少覆盖：多跳问题、无答案问题（知识库里没有的）、带噪声干扰的问题——**无答案问题尤其重要**，它检验的是「不知道时会不会老老实实说不知道」（幻觉治理的防线二）。

**坑 4：把评估当一次性的。** 改一次配置跑一次，看不出退化。好的做法是固定一组评估集，每次改动（换 embedding、调 Prompt、换模型）都跑一遍，**对比分数变化而不是绝对值**——绝对值受测试集影响很大，同一测试集上的前后对比才有意义。

---

## 八、灵魂总结

回到开头的问题：前几篇聊的 Query Rewrite、动态更新、幻觉治理，都在「做改进」；但**没有度量，就没有改进**——「检索质量到底怎么样」「防线到底有没有生效」，靠感觉是不行的。

ragas 给了一套完整答案：

```
评估粒度：拆成 检索质量 + 生成质量 两个环节，各自独立打分
四大指标：Faithfulness（有据可查）/ Answer Relevancy（答即所问）
          Context Precision（排序正确）/ Context Recall（召回完整）
接入三步：准备数据（问题/答案/上下文/标准答案）
          → 配裁判 LLM（要比被评系统强）
          → evaluate() 跑分，看「指标分差」定位问题
进阶玩法：TestsetGenerator 合成测试集冷启动
          在线评估挂 trace 平台持续监控
          新版 @experiment 把评估代码化、可追溯
```

最后记住一条最重要的心法——**别盯综合分，盯分差**：

> 四个指标不是四个并列的数字，而是一套「定位系统」：`faithfulness` 低是生成在编，`context_recall` 低是检索没找到，`context_precision` 低是排序有问题，`answer_relevancy` 低是答非所问。**指标之间的相对关系，比任何绝对值都更有信息量。**

这和前面几篇是同一个闭环：治幻觉先治检索，治检索先能量化检索。评估不是 RAG 的最后一步，而是**每次迭代的第一步**。
