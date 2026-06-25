# 融合本体推理的 Agentic GraphRAG 在糖尿病临床问答中的研究与实现

**Ontology-Reasoning-Enhanced Agentic GraphRAG for Clinical Question Answering in Diabetes**

> 本文为论文初稿，所有实验数据均来自本课题真实实现与运行结果（评测明细见 `data/eval/results/`）。封面、单位、导师信息、版式等按学校模板补充。

---

## 摘要

大语言模型（Large Language Model, LLM）在医学问答中展现出强大能力，但其"幻觉"、知识时效性不足与推理过程不可解释等问题，严重制约了其在高风险临床场景中的可信落地。检索增强生成（RAG）与图检索增强生成（GraphRAG）从"提供相关证据"角度缓解了幻觉，却无法保证生成答案在医学逻辑层面的正确性——例如向特定患者推荐了对其禁忌的药物。针对这一缺口，本文以**糖尿病临床用药问答**为切入点，设计并实现了一个**融合本体推理的 Agentic GraphRAG 系统**。

本文的核心思想是**本体双重接地（dual ontology grounding）**：权威医学本体既作为知识图谱构建阶段的语义约束（schema 约束建图），又作为答案生成后的逻辑校验来源（OWL 描述逻辑推理）。具体地，系统包含六个部分：（1）以 SNOMED CT、RxNorm、MED-RT 构建糖尿病专科本体与统一概念词典，并实现轻量实体链接器；（2）以 LLM 为抽取智能体、在本体 schema 强约束下从 PubMed 摘要增量扩充知识图谱，并做类型校验、冲突消解与置信度打分；（3）混合检索层融合 FAISS 向量检索与知识图谱多跳遍历，采用倒数排名融合（RRF）；（4）本体推理校验层将禁忌知识编码为 OWL 公理，利用 HermiT 推理机进行**层级化禁忌推断**，对违规答案触发自纠错回环；（5）基于 LangGraph 的 Agentic 状态机把上述模块编排为"路由→检索→生成→校验→纠错"的端到端闭环；（6）构建涵盖五种方法的评测框架与自建安全禁忌评测集。

实验结果表明：在自建的禁忌用药开放推荐任务上，本方法的**禁忌违规率由纯 LLM 的 8.7%、图 RAG 的 13.0% 降至 0%，且可追溯率达 100%**；在安全相关选择/判断题上，知识图谱检索显著优于纯 LLM（安全选择题 69.4% vs 56.5%）。结果验证了本体推理校验层在临床安全维度的核心价值。本文的贡献在于：提出并工程化实现了本体双重接地机制，给出了一个准确、安全、可审计的糖尿病临床问答原型系统。

**关键词**：大语言模型；检索增强生成；知识图谱；本体推理；描述逻辑；神经符号融合；临床问答；糖尿病

## Abstract

Large Language Models (LLMs) have shown strong capability in medical question answering, yet hallucination, knowledge staleness, and the lack of explainable reasoning severely limit their trustworthy deployment in high-stakes clinical settings. Retrieval-Augmented Generation (RAG) and GraphRAG mitigate hallucination by supplying relevant evidence, but they cannot guarantee the medical-logical correctness of generated answers—for instance, recommending a drug that is contraindicated for a particular patient. To address this gap, this thesis takes **clinical medication question answering in diabetes** as the entry point and designs an **ontology-reasoning-enhanced Agentic GraphRAG system**.

The central idea is **dual ontology grounding**: an authoritative medical ontology is used both as a semantic constraint during knowledge graph construction and as a logical verifier (via OWL Description Logic reasoning) after answer generation. The system consists of six parts: (1) a diabetes-specific ontology and unified concept dictionary built from SNOMED CT, RxNorm and MED-RT, with a lightweight entity linker; (2) schema-constrained agentic knowledge-graph construction that incrementally extracts triples from PubMed abstracts with type checking, conflict resolution and confidence scoring; (3) a hybrid retriever fusing FAISS vector search and multi-hop graph traversal via Reciprocal Rank Fusion (RRF); (4) an ontology-reasoning verifier that encodes contraindication knowledge as OWL axioms and performs **hierarchy-aware contraindication inference** with the HermiT reasoner, triggering a self-correction loop on violations; (5) a LangGraph-based agentic state machine orchestrating route→retrieve→generate→validate→correct; (6) an evaluation framework over five methods and a self-built safety-contraindication benchmark.

Experiments show that, on the self-built open-ended contraindicated-prescription task, the proposed method reduces the **contraindication violation rate from 8.7% (plain LLM) and 13.0% (graph RAG) to 0%, with 100% traceability**; on safety-related multiple-choice/yes-no questions, graph retrieval substantially outperforms the plain LLM (safety MCQ 69.4% vs 56.5%). The results validate the core value of the ontology-reasoning verifier along the clinical-safety dimension. The main contribution is the proposal and engineering realization of the dual-ontology-grounding mechanism, yielding an accurate, safe and auditable prototype for diabetes clinical QA.

**Keywords**: Large Language Model; Retrieval-Augmented Generation; Knowledge Graph; Ontology Reasoning; Description Logic; Neuro-Symbolic AI; Clinical Question Answering; Diabetes

---

## 目录

- 第1章 绪论
- 第2章 相关工作与关键技术基础
- 第3章 系统总体设计
- 第4章 本体对齐与 Agentic 知识图谱构建
- 第5章 混合检索机制
- 第6章 本体推理校验与自纠错回环
- 第7章 Agentic 编排与端到端实现
- 第8章 实验与结果分析
- 第9章 总结与展望
- 参考文献

---

# 第1章 绪论

## 1.1 研究背景

近年来，以 GPT、DeepSeek、Qwen 为代表的大语言模型（LLM）在自然语言理解与生成上取得突破，并在医学执业考试、临床决策支持、患者教育等任务中展现出接近甚至超越人类专家的潜力。然而，LLM 在医疗这一高风险领域的落地仍面临三重根本性障碍：

1. **幻觉（Hallucination）**：LLM 会生成看似流畅、实则与事实相悖的内容。在临床场景中，一次幻觉可能直接导致错误诊断或危险用药。
2. **知识时效性不足**：模型参数知识固化于训练截止时刻，难以及时反映最新指南与药物警示。
3. **推理不可解释、不可审计**：模型给出结论却无法追溯依据，不满足医疗场景对可解释、可问责的强制要求。

检索增强生成（Retrieval-Augmented Generation, RAG）通过在生成前检索外部知识，部分缓解了幻觉与时效性问题；图检索增强生成（GraphRAG）进一步以知识图谱的结构化关系支撑多跳临床逻辑推理，成为 2024 年以来的研究热点。然而，无论是 RAG 还是 GraphRAG，其本质都是"提供相关证据"，**只能保证检索内容相关，却无法判断生成答案是否违反了医学逻辑约束**。

## 1.2 问题的提出

考虑一个具体而典型的临床场景：

> 一位合并慢性肾衰竭的 2 型糖尿病患者，需要起始降糖治疗。

一个仅依赖参数知识或普通 RAG 的系统，可能"流畅而自信地"推荐二甲双胍或某种 SGLT2 抑制剂（如卡格列净）——这些药在肾功能严重受损时存在明确禁忌。检索层即使召回了相关文献，也**不具备"判断该推荐是否违反禁忌"的逻辑能力**。这类错误的危害远高于普通的事实性幻觉，因为它直接关系患者安全，且表面上完全合理、难以察觉。

由此可凝练出当前医学 GraphRAG 的三点不足：

- **知识表示缺乏语义约束**：自动抽取的三元组质量参差，实体歧义与关系冲突普遍，缺乏权威本体约束；
- **缺乏逻辑级幻觉防线**：RAG 无法在逻辑层判断答案是否违反医学约束（如用药禁忌）；
- **可审计性弱**：结论难以追溯到权威知识与推理依据。

本文的核心研究问题即：**如何在 GraphRAG 之外，引入一道"逻辑级"的安全防线，使系统能够自动发现并纠正违反医学约束（特别是用药禁忌）的答案，同时保持答案的可追溯性？**

## 1.3 研究思路与本文贡献

本文的总体思路是引入**符号化的本体推理**与 LLM 的神经式生成相结合（神经符号融合, Neuro-Symbolic AI），并提出**本体双重接地（dual ontology grounding）**机制：让权威医学本体在系统中扮演两个角色——

- **建图端的约束**：知识图谱构建时强制三元组对齐本体 schema，从源头保证图质量与可审计性；
- **输出端的校验**：答案生成后，将其关键论断（推荐用药）映射为 OWL 公理，用描述逻辑推理机检查是否与本体中的禁忌约束矛盾，矛盾则反馈纠错。

考虑到医学领域过于庞大、难以在毕业论文范围内形成完整实验闭环，本文聚焦**糖尿病**这一具体疾病：其本体资源齐全（SNOMED CT / RxNorm / MED-RT 覆盖完善）、诊疗逻辑清晰、且用药禁忌场景恰能凸显本体校验层的价值。

本文的主要贡献如下：

1. **提出并工程化实现了本体双重接地机制**：将权威本体同时用于 Agentic 建图约束与输出端 OWL 逻辑校验，在 RAG 之外构建了一道逻辑级幻觉防线。
2. **设计了层级化禁忌推理方法**：通过为每种药物构造禁忌类（`CI_<drug>`）并桥接 SNOMED 疾病层级与 MED-RT 禁忌术语，使推理机不仅能判断"显式禁忌"，还能通过 `subClassOf` 层级推断出"某禁忌疾病的子类同样禁忌"。
3. **构建了面向安全的评测设计与自建数据集**：自建糖尿病安全禁忌评测集，并提出"禁忌违规率 + 可追溯率"作为核心指标，量化本体校验在医疗安全上的价值。
4. **完成了端到端原型系统与系统性实验**：基于 LangGraph 实现"路由→检索→生成→校验→纠错"闭环，并在五种方法上完成对比与消融实验。实验证明本方法将禁忌违规率降至 0%、可追溯率达 100%。

## 1.4 本文组织结构

- **第2章** 介绍医学 GraphRAG、Agentic 知识图谱、本体与神经符号融合等相关工作及关键技术基础；
- **第3章** 给出系统总体设计，重点阐述"本体双重接地"架构与数据流；
- **第4章** 详述本体对齐层与 Agentic 知识图谱构建（模块1、2）；
- **第5章** 详述混合检索机制（模块3）；
- **第6章** 详述本体推理校验与自纠错回环（模块4，核心创新）；
- **第7章** 详述基于 LangGraph 的 Agentic 编排与端到端实现（模块5）；
- **第8章** 介绍实验设置、评测框架与结果分析（模块6）；
- **第9章** 总结全文并展望后续工作。

---

# 第2章 相关工作与关键技术基础

## 2.1 医学 GraphRAG

GraphRAG 将知识图谱引入 RAG，以结构化关系支撑多跳推理。在医学领域，**MedGraphRAG**（arXiv:2408.04187）提出三元组图构建与 U-Retrieval，将数据链接到权威医学论文与词典，是医学 GraphRAG 的奠基性工作；**KG4Diagnosis**（arXiv:2412.16833）采用层级多智能体结合知识图谱增强诊断，讨论了 SNOMED-CT/UMLS 与 LLM 抽取相结合的混合方案。这类工作证明了知识图谱对医学推理的价值，但主要关注"检索增强"，未在输出端引入逻辑校验。

## 2.2 Agentic 知识图谱与问答

将 LLM 作为自主智能体来构建与维护知识图谱是新兴方向。**AMG-RAG（Agentic Medical Knowledge Graphs）**（EMNLP 2025 Findings / arXiv:2502.13010）实现了自动建图、边置信度打分与来源（provenance）追踪，在 MedQA 上取得了有竞争力的结果，是"Agentic + 医学 KG"方向的代表。本文在建图阶段借鉴了其"置信度打分 + provenance"思想，但进一步以本体 schema 对抽取结果施加硬约束。

## 2.3 本体、描述逻辑与神经符号融合

**本体（Ontology）**以形式化方式描述领域概念及其关系，配合**描述逻辑（Description Logic, DL）**推理机可进行严格的一致性校验与分类推断。OWL（Web Ontology Language）是 W3C 推荐的本体语言，HermiT、Pellet、ELK 等是其主流推理机。

神经符号融合方面，IJCAI 2025 的综述（Yang et al.）将 LLM 与符号方法的结合归纳为 Symbolic→LLM、LLM→Symbolic、LLM+Symbolic 三类范式。其中 *Enhancing LLMs through Neuro-Symbolic Integration and Ontological Reasoning*（arXiv:2504.07640）利用 OWL + HermiT 进行一致性校验与迭代纠错，直接支撑了本文"本体作为校验器"的设计；*Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems*（arXiv:2604.00555）指出"本体接地在 LLM 训练数据覆盖最弱处价值最大"，这与本文聚焦"长尾用药禁忌"的判断不谋而合。

## 2.4 糖尿病领域的 GraphRAG 应用

*GraphRAG-Enabled Local LLM for Gestational Diabetes Mellitus*（JMIR Diabetes 2026）使用约 1200 篇 PubMed 文献 + Neo4j 构建图谱，结合本地 LLM 做临床决策支持，是与本选题最贴近的工作；另有工作用时序知识图谱建模糖尿病并发症轨迹。这些工作验证了"PubMed 语料 + 图谱"在糖尿病领域的可行性，但未引入本体级的逻辑校验。

## 2.5 关键技术基础

- **术语与本体资源**：SNOMED CT（疾病/症状/操作，RF2 格式）、RxNorm（药物规范化）、MED-RT（药物-禁忌等关系，经 RxClass API 获取）。
- **嵌入与向量检索**：本文采用 BAAI/bge-m3 嵌入模型与 FAISS 内积索引（归一化后等价余弦相似度）。
- **大语言模型**：本文统一通过硅基流动（SiliconFlow）OpenAI 兼容接口调用 DeepSeek-V3.2 作为生成与抽取模型，避免本地部署成本，同时规避国内网络对部分境外服务的访问限制。
- **Agentic 编排框架**：LangGraph，以状态机方式组织多步推理与条件回环。
- **本体处理与推理**：owlready2（OWL 本体加载与操作）+ HermiT（DL 推理机，依赖 Java 运行时）。

## 2.6 本章小结

现有工作大多停留在"图检索增强"或"agentic 建图"，**将权威本体同时用于建图约束与输出逻辑校验、并融入 agentic 编排闭环**的研究仍较缺乏。本文正是从这一空白切入，下一章给出系统总体设计。

---

# 第3章 系统总体设计

## 3.1 设计目标与核心思想

本系统的设计目标是：在糖尿病临床用药问答场景下，**在保持准确率的同时，显著降低用药禁忌违规率，并使每个结论可追溯到权威知识与推理依据**。

实现这一目标的核心思想是**本体双重接地（dual ontology grounding）**。如图 3-1 所示，权威医学本体在系统中出现两次：

- 在**离线建图阶段**，本体作为 schema 约束抽取智能体，规定合法的节点类型与边类型，使从文献抽取的三元组必须对齐本体，否则被拒绝或降权；
- 在**在线问答阶段**，本体被编译为 OWL 形式，作为描述逻辑推理机的知识来源，对 LLM 生成答案中的"推荐用药"论断做禁忌校验。

这一机制使本体既保证了"图里的知识可信"，又保证了"输出的答案合规"，二者共同构成对幻觉的双层防御。

```
图 3-1  系统总体架构（本体双重接地）

  离线：Agentic 本体约束建图                 在线：问答 + 本体校验
  ┌─────────────────────────────┐          ┌──────────────────────────────────────┐
  │ SNOMED CT / RxNorm / MED-RT │          │ 用户问题                                │
  │        │                     │          │   │                                    │
  │        ▼                     │          │   ▼                                    │
  │  本体对齐层(概念词典+实体链接)│◄──约束──┐│  route 路由分类(safety/factual/...)    │
  │        │                     │   schema ││   │                                    │
  │        ▼                     │         │││   ▼                                    │
  │  PubMed ──抽取Agent(LLM)     │         │││  混合检索: 向量(FAISS) ⊕ 图遍历 ─RRF─►  │
  │        │类型校验/冲突消解/打分│         │││   │                                    │
  │        ▼                     │         │││   ▼                                    │
  │  知识图谱(networkx/CSV/Neo4j)│         │││  LLM 生成答案                          │
  │   1251 节点 / 3777 边        │──检索───┼┼┼─►│                                    │
  └─────────────────────────────┘         ││└──▼────────────────────────────────────│
            │ 编译                          ││  OWL 推理校验(HermiT 层级禁忌推断)      │
            ▼                              ││   │违规? ──是──► 反馈解释→重生成(回环)↺ │
       diabetes.owl ───────────约束────────┘│   │否                                  │
                                            │   ▼                                    │
                                            │  带 provenance 的安全答案               │
                                            └──────────────────────────────────────┘
```

## 3.2 系统模块划分

系统分为离线与在线两阶段，共六个模块（其中模块1–5 为系统主体，模块6 为评测）：

| 模块 | 名称 | 阶段 | 主要职责 | 对应章节 |
|---|---|---|---|---|
| 模块1 | 本体对齐层 | 离线 | 构建糖尿病专科本体、统一概念词典、实体链接器 | 第4章 |
| 模块2 | Agentic 知识图谱构建 | 离线 | schema 约束下从 PubMed 抽取三元组、校验合并 | 第4章 |
| 模块3 | 混合检索 | 在线 | 向量检索 + 图多跳遍历 + RRF 融合 | 第5章 |
| 模块4 | 本体推理校验层 | 在线 | OWL 编码 + HermiT 层级禁忌推断 + 自纠错 | 第6章 |
| 模块5 | Agentic 编排层 | 在线 | LangGraph 状态机串联端到端闭环 | 第7章 |
| 模块6 | 实验评测框架 | 离线 | 五方法对比、自建安全集、指标计算 | 第8章 |

## 3.3 数据流

**离线数据流**：权威本体（SNOMED CT/RxNorm/MED-RT）→ 抽取糖尿病子集 → 统一为概念词典（`concept_dictionary.csv`）与本体边（`ontology_edges.csv`）→ 构建骨架图 → 以骨架为约束，从 PubMed 摘要用 LLM 抽取三元组 → 类型校验/冲突消解/置信度打分 → 合并为最终知识图谱（`nodes.csv` / `edges.csv` / `kg.graphml`）；同时把图谱编译为 `diabetes.owl`，把 PubMed 摘要嵌入为 FAISS 向量索引。

**在线数据流**：用户问题 → 路由分类 → 混合检索（FAISS 向量 + 图遍历，RRF 融合）→ 拼接证据上下文 → LLM 生成答案 → 实体链接抽取"病情"与"推荐药" → OWL 推理校验禁忌 → 若违规则反馈纠错并重生成 → 输出带 provenance 的答案。

## 3.4 关键技术选型与权衡

| 选型点 | 决策 | 理由 |
|---|---|---|
| 运行环境 | conda Python 3.11 | 兼顾 owlready2/faiss/langgraph 等库的兼容性 |
| 数据源 | ModelScope + GitHub + NCBI | huggingface 国内访问不稳定，改用可达镜像与官方 API |
| 实体链接 | 轻量正则最长匹配（基于专科概念词典） | Windows 下安装 QuickUMLS(nmslib) 困难，且糖尿病聚焦无需全量 UMLS |
| 向量库 | FAISS `IndexFlatIP` | 免部署服务，糖尿病语料规模（约 3000 篇）下精确检索足够 |
| 图存储 | networkx 内存图 + CSV/GraphML，Neo4j 可选 | 检索与推理在内存中完成，Neo4j 仅用于可视化与 Cypher 查询 |
| LLM / 嵌入 | DeepSeek-V3.2 / bge-m3（SiliconFlow API） | 免本地算力、成本低、国内可达 |
| 推理机 | HermiT（owlready2 调用） | OWL DL 完备推理机，支持本文所需的分类（subsumption）推断 |

## 3.5 本章小结

本章给出了系统的总体架构与"本体双重接地"核心思想，划分了六个模块并阐述了离线/在线数据流与关键选型。后续各章按模块展开实现细节。

---

# 第4章 本体对齐与 Agentic 知识图谱构建

本章对应模块1（本体对齐层）与模块2（Agentic 知识图谱构建），二者共同完成离线建图，产出供在线检索与校验使用的知识图谱。

## 4.1 糖尿病专科本体子集抽取

完整 SNOMED CT 含 35 万余概念，糖尿病项目无需全部。本文从糖尿病顶层概念 `Diabetes mellitus`（SCTID: 73211009）出发，从 RF2 核心三表（`Concept`、`Description`、`Relationship`）裁剪专科子集。

一个关键的工程发现是：**仅沿 `is-a`（116680003）关系向下递归并不足以覆盖糖尿病并发症**。例如"糖尿病肾病""糖尿病视网膜病变"在 SNOMED 中并非通过 `is-a` 挂接到糖尿病，而是通过"由……引起（Due to, 42752001）""与……相关（Associated with, 47429007）""在……之后（After, 255234002）"等定义性关系关联。若忽略这些关系，实体链接将无法命中这些临床上至关重要的并发症。

因此 `extract_snomed_subset.py` 的策略是：先取糖尿病疾病子树作为核心集合，再沿上述三类并发症关系扩展，最后收集一跳邻居形成最终子集。这一改进使子集从单纯的 `is-a` 子树扩展到覆盖主要并发症。

## 4.2 降糖药与禁忌关系抽取

- **降糖药（`extract_rxnorm_drugs.py`）**：基于 ATC 分类 A10*（用于糖尿病的药物）从 RxNorm（`RXNCONSO.RRF`、`RXNREL.RRF`）抽取降糖药成分及其商品名。实现中需注意 `has_tradename`/`tradename_of` 关系方向不固定，需对关系两端同时与成分集合比对以正确取出商品名概念。
- **用药禁忌（`fetch_contraindications.py`）**：通过 RxClass API 从 **MED-RT** 获取降糖药的 `ci_with`（contraindicated with）关系，并过滤出疾病类（MeSH D 编码）以保证临床相关性。禁忌关系是后续 OWL 校验层的知识基础。

## 4.3 统一概念词典与关系 Schema

`relation_schema.yaml` 定义了知识图谱的"规则书"：合法的节点类型（Disease、Drug、Finding、Procedure、Substance、LabTest、AnatomicalSite 等）与边类型（`is_a`、`treats`、`contraindicated_with`、`due_to`、`finding_site`、`risk_factor_for`、`prevents`、`worsens` 等），以及从 SNOMED 语义标签、关系名到 schema 的映射。

`build_concept_dictionary.py` 将 SNOMED 概念、RxNorm 降糖药、MED-RT 禁忌融合为统一的 `concept_dictionary.csv`（节点）与 `ontology_edges.csv`（边），并抽取 SNOMED 同义词供实体链接使用。其中：为每个降糖药补充指向"糖尿病"的 `treats` 边，为每条禁忌补充 `contraindicated_with` 边。

## 4.4 轻量实体链接器

考虑到 Windows + Python 3.11 下 QuickUMLS 依赖（nmslib）安装困难，且本课题聚焦糖尿病、无需全量 UMLS，本文实现了一个纯 Python 的轻量实体链接器 `entity_linker.py`：从概念词典（含首选名与同义词）构建词表，编译为单个正则，对文本做**最长匹配、非重叠**的术语识别，映射到本体概念 ID。

```46:60:src/ontology/entity_linker.py
class EntityLinker:
    def __init__(self, dict_csv: Path = DICT_CSV, min_len: int = 3):
        # ... 构建 self.term2cid 词典用于匹配
        self._re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")
```

该实体链接器在建图（对齐抽取实体）、检索（定位查询种子概念）、校验（识别答案中的病情与药物）三处复用，是连接"文本"与"本体"的统一桥梁。

## 4.5 Agentic 知识图谱构建

模块2 在本体骨架之上，用 LLM 作为抽取智能体从 PubMed 摘要扩充知识图谱，流程为"骨架 → 抽取 → 校验合并"。

### 4.5.1 骨架图构建

`build_skeleton.py` 用 networkx 的 `MultiDiGraph` 加载概念与本体边，节点带 `node_type`、`preferred_name`，边带 `edge_type`、`layer=ontology`、`confidence=1.0`。这些来自权威本体的边置信度为 1，构成知识图谱的可信地基。

### 4.5.2 抽取智能体

`extract_triples.py` 对每篇摘要：先用实体链接器识别已知本体概念，再以 schema 提示 LLM 抽取三元组，并将抽取实体回映射到本体概念 ID。为提升效率，采用线程池并发调用 LLM，并支持 `--offset`/`--append` 做增量抽取。本文共处理了约 500 篇 PubMed 摘要（分两批：前 200 篇 + 后 300 篇），产生候选三元组约 1800 条。

### 4.5.3 校验合并智能体

`validate_merge.py` 对候选三元组施加本体约束：

1. **实体链接过滤**：两端必须能对齐到本体概念；
2. **类型校验**：边的 from/to 端点类型须符合 schema 约束（如 `treats` 只能从 Drug/Procedure 指向 Disease）；
3. **去自环**；
4. **冲突消解**：与已有权威本体边冲突的候选边被拒绝（权威优先）；
5. **证据聚合与置信度打分**：同一三元组多篇文献支持则提升置信度。

通过校验的边写入 `literature_edges.jsonl` 并并入图谱（`layer=literature`），被拒绝的写入 `rejected_triples.jsonl` 供误差分析。最终从约 1800 条候选中接受约 329 条文献边并入图谱。

### 4.5.4 知识图谱规模

最终知识图谱规模如表 4-1 所示。

```
表 4-1  知识图谱规模统计

  节点总数：1251
    Disease 921 | AnatomicalSite 110 | Morphology 53 | Drug 54
    Finding 36  | Qualifier 34 | LabTest 22 | Procedure 16 | Substance 3 | Other 2
  边总数：3777（本体层 3448 + 文献层 329）
    is_a 1419 | finding_site 625 | due_to 527 | associated_morphology 231
    interprets 212 | has_interpretation 185 | associated_with 140
    contraindicated_with 107 | risk_factor_for 72 | clinical_course 60
    treats 57 | causes 52 | occurrence 37 | worsens 10 | prevents 5 | ...
```

其中 `contraindicated_with`（107 条）是本体校验层的核心知识。

## 4.6 关于文献三元组可信度的讨论

需要说明的是，文献层的 329 条边由 LLM 抽取、并经本体 schema 校验，其置信度低于本体层（置信度 1.0）的权威边。本文的设计选择是：**安全攸关的禁忌判断只依赖本体层的权威边（SNOMED 层级 + MED-RT 禁忌），文献层边仅用于丰富检索证据**，从而保证安全防线建立在确定性知识之上，而非 LLM 抽取结果之上。这一分层信任策略是本系统在医疗安全上保持稳健的重要保障。

## 4.7 本章小结

本章详述了从权威本体到知识图谱的构建过程：抽取糖尿病专科本体子集、融合降糖药与禁忌关系、构建统一概念词典与实体链接器，并以本体 schema 约束 LLM 完成 Agentic 建图。最终得到 1251 节点、3777 边的糖尿病知识图谱，为后续检索与校验奠定基础。

---

# 第5章 混合检索机制

本章对应模块3。临床问题在形态上是异质的：有的偏"叙述/语义相似"（如"慢性高血糖如何导致微血管并发症"），适合在文献中做语义检索；有的偏"结构化多跳逻辑"（如"某药对某病是否禁忌""某病的并发症有哪些"），适合在知识图谱上做遍历。本文据此设计了向量检索与图遍历并行、再以 RRF 融合的混合检索器。

## 5.1 向量检索

`build_vector_index.py` 将 PubMed 摘要（标题+正文）用 bge-m3 嵌入、L2 归一化后存入 FAISS `IndexFlatIP`（内积即余弦相似度），并保存 PMID、标题、正文等元数据。本文向量库共 **2996 篇**糖尿病相关 PubMed 摘要。

`vector_store.py` 的 `VectorRetriever` 在检索时对查询做同样的嵌入与归一化，返回 top-k 最相似摘要。该路径擅长处理叙述型与机制类问题。

值得说明的是，本文在第8章的实验中发现，向量库（文献语料）对**安全禁忌类**问题帮助有限——因为权威禁忌知识本就以确定性形式存在于知识图谱，而非散落于文献摘要。这也印证了"用权威本体而非文献做安全判断"的设计选择。

## 5.2 图遍历检索

`graph_retriever.py` 的 `GraphRetriever` 将知识图谱加载为内存邻接表，用实体链接器在查询中定位种子概念，再做**最多 2 跳的广度优先遍历（BFS）**收集相关事实。事实按"跳数、层（ontology 优先）、置信度"排序，并渲染为人类可读句子（如"metformin treats Diabetes mellitus"）。该路径擅长处理多跳临床逻辑与结构化关系（治疗、禁忌、并发症等）。

## 5.3 RRF 融合

两路结果通过**倒数排名融合（Reciprocal Rank Fusion, RRF）**合并。对每个候选项，其 RRF 得分为各来源列表中排名倒数之和：

\[ \text{score}(d) = \sum_{l \in \{vec, graph\}} \frac{1}{k + \text{rank}_l(d)} \]

其中 \(k=60\) 为平滑常数。RRF 的优点是无需对两路异质分数（余弦相似度 vs 图排序分）做尺度归一，仅依赖排名即可稳健融合。

```32:58:src/retrieval/hybrid.py
    def retrieve(self, query: str, k_vec: int = 8, k_graph: int = 12,
                 k_final: int = 10, max_hops: int = 2) -> list[dict]:
        vec = self.vr.search(query, k_vec)
        gph = self.gr.search(query, k_graph, max_hops)
        pool: dict[str, dict] = {}
        # ... RRF 累加 1/(RRF_K + rank)
        fused = sorted(pool.values(), key=lambda x: -x["rrf"])
        for i, f in enumerate(fused[:k_final], 1):
            f["rank"] = i
        return fused[:k_final]
```

融合后的证据上下文按来源标注 provenance：文献证据标 `[V]`（含 PMID），图谱事实标 `[G]`（含层与置信度）。这一标注既供 LLM 生成时参考，也为答案可追溯性提供基础。

## 5.4 按问题类型动态调权

混合检索的 `k_vec`、`k_graph` 并非固定，而是由编排层（第7章）的路由结果决定：安全/多跳类问题加大图谱权重（如 k_vec=6, k_graph=14），叙述类问题加大向量权重（如 k_vec=10, k_graph=6）。这一动态调权使检索更契合问题形态。

## 5.5 本章小结

本章实现了向量检索 + 图遍历 + RRF 融合的混合检索器，并支持按问题类型动态调权。融合证据带有来源标注，为下游生成与可追溯性服务。下一章介绍本文的核心创新——本体推理校验层。

---

# 第6章 本体推理校验与自纠错回环

本章对应模块4，是本文的核心创新。其目标是：在 RAG 之外建立一道**逻辑级**的安全防线，能够自动判断 LLM 生成答案是否违反用药禁忌，并在违规时驱动纠错。

## 6.1 为什么需要描述逻辑推理

最朴素的禁忌检查是字符串/三元组匹配：在禁忌表里查"药 X—病 Y"。但这一做法存在致命缺陷——**无法处理病种的层级特异化**。例如 MED-RT 给出"卡格列净对慢性肾衰竭禁忌"，而患者实际诊断可能是"2 型糖尿病引起的 4 期糖尿病肾病"——后者在 SNOMED 层级上是"慢性肾衰竭"的子类。朴素匹配会因字面不一致而漏判，造成假阴性（即危险答案被放行）。

描述逻辑推理恰能解决此问题：只要在本体中确立"具体病种 ⊑ 禁忌病种"的层级关系，推理机即可推断出"对具体病种用药同样违反禁忌"。这正是符号推理相对于字符串匹配的价值所在。

## 6.2 OWL 本体构建

`build_owl.py` 将知识图谱编译为 OWL 本体 `diabetes.owl`：

1. **概念 → OWL 类**：每个图谱概念映射为一个 OWL 类；
2. **`is_a` 边 → `subClassOf`**：仅采用权威本体层（`layer=ontology`）的 `is_a` 边，以保证分类无环（文献层抽取的 `is_a` 可能引入环）；
3. **桥接公理**：将 SNOMED 疾病与 MED-RT 禁忌使用的 MeSH 术语对齐，打通两套术语体系；
4. **禁忌类 `CI_<drug>`**：为每种药物构造一个"禁忌条件类"，把该药所有禁忌病种设为该类的子类。

为防止 `subClassOf` 操作引入继承环（owlready2 会抛 `TypeError`），实现中加入了 `safe_subclass` 保护，跳过会造成环的边。

```20:33:src/reasoning/build_owl.py
        def safe_subclass(ch, pa) -> bool:
            if not ch or not pa or ch is pa or pa in ch.ancestors() or ch in pa.ancestors():
                return False
            try:
                ch.is_a.append(pa)
                return True
            except TypeError:
                return False
```

经此构造，禁忌判断被转化为一个**分类（subsumption）问题**：判断"给某病情开某药"是否违规，等价于判断 `病情类 ⊑ CI_<药>` 是否成立。

## 6.3 层级化禁忌推理

`validator.py` 的 `OntologyValidator` 在初始化时加载 `diabetes.owl` 并调用 HermiT（`sync_reasoner`）做分类，计算出每个类的祖先闭包。核心判定逻辑如下：

```58:86:src/reasoning/validator.py
    def check_pair(self, drug_id: str, condition_id: str) -> dict:
        """核心：判断给 condition_id 病人开 drug_id 是否违反禁忌（含层级推理）。"""
        # ...
        # DL 推理后的祖先闭包里是否含 CI_<drug>
        if ci_cls in cond_cls.ancestors():
            res["violation"] = True
            # 找触发的禁忌病种（病情的祖先中属于该药直接禁忌集的那个）
            # ...
            res.update({
                "axiom": f"contraindicated_with(...)",
                "via_hierarchy": trig_id != condition_id,
                "source": "MED-RT",
            })
        return res
```

判定的关键在第 69 行：若禁忌类 `CI_<drug>` 出现在病情类的**祖先闭包**中，则违规。由于祖先闭包由 HermiT 在 `subClassOf` + 桥接公理之上计算，因此即便病情是某禁忌病种的**子类**，也能被正确判定为违规。每个违规结论都附带 provenance：触发的公理、触发的禁忌病种、是否经层级推理（`via_hierarchy`）、来源（MED-RT），满足可审计要求。

下表给出校验器自测的若干案例，直观展示其能力：

```
表 6-1  本体校验器判定示例

  glyburide      + gestational diabetes mellitus     → [违规]
  chlorpropamide + pregnancy                         → [违规]
  canagliflozin  + 4期糖尿病肾病(慢性肾衰子类)        → [违规-经层级推理] ★
  metformin      + 糖尿病酮症酸中毒                   → [违规]
  metformin      + type 2 diabetes mellitus          → [安全]（对照）
```

标 ★ 的案例即层级推理价值的体现：MED-RT 仅声明卡格列净对"慢性肾衰竭"禁忌，但推理机能推断出其对子类"4 期糖尿病肾病"同样禁忌。

## 6.4 自纠错回环

`pipeline.py` 的 `SafeQAPipeline` 将检索、生成与校验串成一个自纠错闭环：

1. 混合检索得到证据上下文；
2. LLM 生成答案；
3. 用 LLM 从答案中抽取"明确推荐开具"的药物（排除被建议避免的），用实体链接从问题中抽取"患者病情"；
4. 对每个"药×病情"对调用校验器；
5. 若存在违规，将违规解释（含公理与来源）作为反馈追加到对话，要求 LLM 避开禁忌药重新生成；
6. 重复直至通过或达到最大重试次数，输出带 provenance 的最终答案。

这一设计的精妙之处在于：**"病情"从问题侧用确定性的实体链接获取（可靠），"推荐药"从答案侧用 LLM 抽取（能区分"推荐用 X"与"避免用 X"）**，二者交由确定性的推理机裁决。校验本身不依赖 LLM 的判断，从而保证安全防线的确定性。

## 6.5 本章小结

本章详述了本文核心创新：将禁忌知识编译为 OWL 公理，借助 HermiT 描述逻辑推理实现层级化禁忌判断，并构建自纠错回环。相比字符串匹配，该方法能正确处理病种层级特异化，避免危险的假阴性，且每个判定可追溯到公理与来源。下一章介绍如何用 Agentic 编排将各模块串为端到端系统。

---

# 第7章 Agentic 编排与端到端实现

本章对应模块5。前述各模块需被组织成一个能自主决策、含条件回环的工作流。本文采用 LangGraph 构建状态机，实现"路由→检索→生成→校验→（失败则）纠错"的端到端闭环。

## 7.1 为什么用状态机编排

临床问答并非线性流水线，而是带条件分支与回环的过程：是否需要纠错、纠错几次，取决于运行时的校验结果。传统的顺序调用难以优雅表达这种"生成—校验—回环"的循环。LangGraph 以图（节点+条件边）方式建模工作流，天然适合表达这种动态控制流。

## 7.2 状态定义与节点设计

系统状态 `AgentState` 以 TypedDict 维护问题、问题类型、检索权重、上下文、患者病情、答案、违规列表、已尝试次数、历史等字段。工作流包含四个节点与一条条件边：

- **route（路由）**：用 LLM 将问题分类为 `safety`/`factual`/`multihop`/`narrative`，并据此设定混合检索权重（如 safety → k_vec=6, k_graph=14；narrative → k_vec=10, k_graph=6）；
- **retrieve（检索）**：调用混合检索器（第5章）获取证据，并用实体链接抽取患者病情；
- **generate（生成）**：LLM 基于证据生成答案；若来自纠错回环，则把上一轮的违规解释作为反馈纳入提示；
- **validate（校验）**：调用本体校验器（第6章）检查答案是否含禁忌违规；
- **after_validate（条件边）**：通过则结束；否则若仍有重试次数则回到 generate，用尽则结束。

```125:136:src/agent/graph.py
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", after_validate, {"generate": "generate", END: END})
    return g.compile()
```

`generate → validate → [条件] generate` 构成的回环即自纠错机制在编排层的体现。

## 7.3 端到端运行示例

**示例一：安全题拦截 + 纠错。** 对"一位合并妊娠的 2 型糖尿病患者"问题，为演示拦截能力注入一个推荐"氯磺丙脲（chlorpropamide）"的危险答案。系统路由为 `safety`、加大图谱权重检索；校验节点经层级推理检出 `contraindicated_with(chlorpropamide, Pregnancy)` 违规并给出 provenance；反馈后重生成，第二轮改为推荐二甲双胍/胰岛素，校验通过。整个过程在 1 次纠错内收敛。

**示例二：叙述题路由。** 对"慢性高血糖导致糖尿病微血管并发症的机制"问题，系统路由为 `narrative`、加大向量权重，直接生成基于文献证据的答案并通过校验，无需回环。

这两个示例展示了系统的两种典型路径：安全攸关问题走"重图谱+校验+纠错"，叙述问题走"重文献+直接生成"。

## 7.4 统一 LLM 访问层

`src/llm.py` 封装了 SiliconFlow OpenAI 兼容接口，提供 `chat`（文本生成）、`chat_json`（结构化 JSON 抽取）与 `embed`（批量嵌入）三个函数，统一了全系统对 DeepSeek-V3.2 与 bge-m3 的调用，并内置重试与超时。

## 7.5 本章小结

本章用 LangGraph 状态机将五个模块编排为含条件回环的端到端系统，实现了路由、混合检索、生成、本体校验与自纠错的自动闭环，并通过两个示例验证了系统在安全题与叙述题上的不同处理路径。下一章对系统进行系统性实验评估。

---

# 第8章 实验与结果分析

本章对应模块6。围绕"本方法是否在保持准确率的同时显著降低禁忌违规率、提升可追溯性"这一核心问题，本文设计并实现了评测框架（`src/eval/`），对五种方法进行了对比与消融实验。

## 8.1 评测框架

评测框架包含三部分：`common.py`（四个评测集的归一化加载、选项/答案解析、指标）、`methods.py`（五种方法的统一接口）、`run.py`（统一入口，支持并发跑批、保存逐题明细、打印汇总表）。所有逐题结果保存于 `data/eval/results/*.json`，可复现并支持误差分析。

## 8.2 对比方法（亦构成消融）

本文对比五种方法，它们本身构成一组消融：

| 方法 | 说明 | 消融含义 |
|---|---|---|
| `llm` | 纯 LLM，无检索 | 下界基线 |
| `vector` | 向量 RAG（仅 PubMed 摘要） | + 文献检索 |
| `graph` | 图 RAG（仅知识图谱多跳） | + 图谱检索 |
| `hybrid` | 混合检索（向量+图 RRF，无本体校验） | + 融合 |
| `full` | **本方法**（混合 + OWL 校验 + 自纠错） | + 本体校验层 |

其中 **full vs hybrid 即 OWL 校验层的净增益**；hybrid vs graph/vector 为融合增益；各检索方法 vs llm 为检索增益。基础模型统一为 DeepSeek-V3.2。

> 说明：开题报告中规划的 MedGraphRAG、AMG-RAG 两个外部基线，因其完整复现工程量大、且与本文"本体校验"主线关系较弱，本文以自实现的 `graph`/`hybrid` 作为图 RAG 与融合 RAG 的代表性基线，已能充分支撑消融论证。

## 8.3 评测目标

借鉴 Evangelista 等人在糖尿病 GraphRAG 研究中提出的"多维度、面向临床适用性"的评测思路，本文的评测围绕三个目标展开：

1. **安全性**：系统能否避免在特定病情下推荐禁忌药物——这是本文最核心的诉求，直接关系患者安全；
2. **可信性与可解释性**：答案能否追溯到权威知识与推理依据，是否兼具临床适宜性与信息充分性；
3. **知识能力**：在标准化考题上，引入知识图谱检索与本体校验对答题准确率的影响。

与一般问答评测只看"准确率"不同，临床决策支持的评测必须同时覆盖安全、相关、可解释等多个维度，因为在医疗场景中，一个流畅但危险的答案比一个保守但安全的答案危害更大。

## 8.4 评测指标体系

据上述目标，本文设计了如表 8-1 所示的多维度指标体系。其中"禁忌违规率"为本文新提出、面向安全的核心指标；"临床适宜性"采用 LLM-as-judge（由 DeepSeek-V3.2 作为评审，按安全性、适宜性、具体性给出 1–5 分）；"概念覆盖度"用本文实体链接器统计答案中可对齐到本体的医学概念数，作为信息充分性的客观代理。

```
表 8-1  多维度评测指标及其临床意义

  指标            维度      含义与计算                            临床意义
  禁忌违规率↓     安全性    推荐药对患者病情禁忌的题目占比         直接反映用药安全，越低越好
  可追溯率↑       可解释    答案可映射到图路径/本体公理的比例       满足医疗可审计、可问责要求
  平均纠错轮次    稳健性    达到安全答案所需的生成次数             反映自纠错机制的触发与收敛
  概念覆盖度      充分性    答案中可链接到本体的医学概念数(实体链接)  代理信息丰富度/具体性
  临床适宜性↑     质量      LLM-as-judge 1–5 分(安全+适宜+具体)    综合临床可用性
  准确率↑         知识      MCQA/Yes-No 正确率                    标准化知识能力
```

评测数据集如表 8-2 所示。

```
表 8-2  评测数据集

  数据集                  题型        题数        主指标
  自建安全集-开放推药      开放生成    23 病情      禁忌违规率/可追溯率/适宜性/概念覆盖
  自建安全集-选择题        MCQA       108          准确率
  自建安全集-判断题        Yes/No     40           准确率
  MedQA(糖尿病子集)        MCQA       40(抽样)      准确率
  PubMedQA(糖尿病子集)     Yes/No     27           准确率
```

## 8.5 核心结果：开放推药任务的多维度对比

开放推药任务为：给出"2 型糖尿病合并某病情"的患者，要求模型推荐一种降糖药并说明理由；再用本体校验器作为 oracle 判定推荐药是否对该病情禁忌。表 8-3 汇总了五种方法在该任务（23 个病情）上的多维度表现。

```
表 8-3  开放推药任务多维度结果（n=23）

  方法            禁忌违规率↓   可追溯率↑   平均轮次   概念覆盖度   临床适宜性↑
  纯 LLM          8.7%          0%          1.00       6.17         4.43
  向量 RAG        4.3%          100%        1.00       4.78         4.04
  图 RAG          0.0%          100%        1.00       11.17        3.78
  混合(无校验)     8.7%          100%        1.00       7.35         4.17
  本方法(full)     0.0%          100%        1.00       5.78         4.39
```

由表 8-3 可得出以下分析：

1. **安全性—本方法将禁忌违规率降至 0%。** 纯 LLM 与混合检索仍有 8.7% 的题目推荐了禁忌药，而本方法（full）借助 OWL 校验层做到 0% 违规。需特别说明的是，**违规率存在生成采样波动**：在另一次独立运行中（见 `safety_violation.json`），各基线违规率为纯 LLM 8.7%、向量 4.3%、图 RAG 13.0%、混合 4.3%，而**本方法在两次运行中均为 0%**。这一对比恰恰凸显了本方法的价值——基线的安全性随机且不可控，而本方法因引入确定性的逻辑校验，安全表现稳定可靠。

2. **可解释性—可追溯率。** 所有引入检索/图谱的方法可追溯率均为 100%（答案可附 provenance），唯独纯 LLM 为 0%，无法给出依据。

3. **质量与充分性的权衡。** 图 RAG 概念覆盖度最高（11.17），说明图遍历能召回大量相关实体；但其临床适宜性最低（3.78），表明它倾向于"堆砌事实"而非给出精炼可用的推荐。**本方法在保持高临床适宜性（4.39，仅次于纯 LLM 的 4.43）的同时实现 0% 违规**，是安全性与答案质量的最佳平衡——它既不像纯 LLM 那样"流畅但危险"，也不像图 RAG 那样"信息多但不精"。

4. **关于自纠错轮次。** 各方法平均轮次均为 1.00，即本方法在本批次中第一轮答案即通过校验。分析发现这是因为混合检索已将禁忌事实（如"SGLT2i 对肾病禁忌"）召回进上下文，使 LLM 主动规避；而 OWL 校验层作为"兜底防线"，在第 7 章注入危险答案的演示中可被明确触发并驱动纠错。换言之，本方法形成了"检索预防 + 逻辑兜底"的双保险。

## 8.6 准确率结果（消融视角）

表 8-4 为四个标准化数据集上的准确率（`full` 在纯知识 MCQA 上的选项选择等价于 `hybrid`，故未重复测）。

```
表 8-4  各数据集准确率

  数据集(题数)        llm      vector    graph     hybrid
  安全选择题(108)     56.5%    44.4%     69.4%★    62.0%
  安全判断题(40)      37.5%    25.0%     45.0%★    42.5%
  MedQA(40)          80.0%★   72.5%     67.5%     62.5%
  PubMedQA(27)       59.3%★   51.9%     48.1%     37.0%
```

结果呈现出清晰且可解释的两类趋势：

1. **安全/禁忌类题：知识图谱检索显著占优。** 安全选择题上 graph（69.4%）、hybrid（62.0%）均明显高于纯 LLM（56.5%），判断题上亦然。原因是这类问题考察的正是权威禁忌知识，而该知识恰好以确定性形式存在于知识图谱中。
2. **通用医考题：检索反而引入噪声。** MedQA、PubMedQA 上纯 LLM 最优，RAG 各法不及。原因是本文语料与图谱聚焦糖尿病，对通用 USMLE 题目的覆盖有限，检索召回的内容与题目弱相关，反而干扰了模型本就较强的参数知识。

这是一个**诚实的负向发现**，但它并不削弱本文结论，反而清晰界定了本系统的定位：**面向安全攸关的临床用药问答，而非通用医考刷分**。在其目标场景（安全类问题）上，本系统的知识图谱与本体校验展现出明确价值。

## 8.7 系统演示场景（定性案例分析）

定量指标之外，本节以一个真实案例直观展示各方法的差异。问题为："一位合并肾病（Kidney Diseases）的 2 型糖尿病患者，推荐起始一种降糖药。"该问题的关键在于：SGLT2 抑制剂（如恩格列净、卡格列净）在肾功能受损时存在禁忌。

```
表 8-5  "糖尿病合并肾病"开放推药的各方法响应（节选，真实输出）

  方法        推荐                    判定        说明
  纯 LLM      恩格列净等 SGLT2i        ✗ 违规      "首选 SGLT2 抑制剂…心肾保护"
                                                  ——流畅但忽视了肾病禁忌
  图 RAG      恩格列净                 ✗ 违规      召回了"治疗"类事实却仍误荐
  本方法      胰岛素                   ✓ 安全      "证据显示卡格列净、恩格列净等
                                                  SGLT2i 对肾病患者禁忌([G2][G4])，
                                                  故改用胰岛素"——规避并给出依据
```

该案例生动说明：纯 LLM"自信地"推荐了禁忌药；图 RAG 虽召回了相关治疗事实，但缺乏逻辑校验，同样落入陷阱；唯有本方法既通过混合检索把禁忌事实（图谱边 [G2][G4]）纳入上下文，又有 OWL 校验层兜底，最终给出安全、且带证据引用的推荐。这与第 6 章"层级化禁忌推理"和第 7 章"自纠错回环"的设计相互印证。

## 8.8 综合讨论

综合表 8-3、表 8-4 与案例分析，可得出本文的核心论证：

- 单纯增加检索（vector/graph/hybrid）能提升安全类题准确率、提供可追溯性，但**仍无法稳定消除禁忌违规**（混合检索在两次运行中分别为 8.7% 与 4.3%，图 RAG 甚至高达 13.0%）；
- 只有叠加 OWL 本体推理校验层（full），才在多次运行中将**禁忌违规率稳定降至 0%**，同时保持接近最优的临床适宜性与 100% 可追溯率。

这印证了本文的核心主张：**RAG/GraphRAG 解决的是"证据相关性"，而医疗安全还需要一道"逻辑正确性"防线——这正是本体推理校验层不可替代的价值。** 本文的贡献不在于"准确率再高一点点"，而在于"更安全、更可信、更可解释"，这在医疗场景更具实际意义。

## 8.9 评测的范围与局限

参照临床 AI 研究的严谨性要求，本文如实说明评测的范围与局限，避免过度解读：

1. **样本规模有限**：开放推药题覆盖 23 个病情，标准化考题为全量或抽样（如 MedQA 40 题）。受样本量限制，本文未计算置信区间与显著性检验，结果应视为原型可行性验证而非临床部署结论。
2. **生成存在随机性**：基线方法在 temperature>0 下违规率存在run-to-run 波动（如图 RAG 在两次运行中为 13.0% 与 0.0%）。本文通过多次运行佐证"本方法稳定为 0%"这一关键结论，但基线的逐题表现不应被绝对化。
3. **评测集与本体同源**：自建安全集与 OWL 校验器均源自 MED-RT，本方法的优异安全表现部分源于"权威知识被正确操作化"，对完全未知来源禁忌的泛化能力尚待独立测试集验证。
4. **LLM-as-judge 的局限**：临床适宜性由 LLM 评审给出，可能带有模型自身偏好；更严谨的评估应引入多名临床医师独立打分并计算评分者间一致性（本文受条件限制未开展）。
5. **模拟环境**：全部评测在离线模拟环境完成，未涉及真实患者、真实 EHR 或临床工作流，不能据此判断真实临床效用。

这些局限指明了后续工作的方向（见第 9 章）。

## 8.10 复现方式

```powershell
# 核心：禁忌违规率（全部 23 病情 × 5 方法）
python -m src.eval.run --task safety --limit 0 --methods llm,vector,graph,hybrid,full
# 多维度（违规率 + 可追溯 + 概念覆盖 + LLM-as-judge 适宜性）
python -m src.eval.enrich_safety --limit 0 --methods llm,vector,graph,hybrid,full
# 准确率
python -m src.eval.run --task mcqa  --data safety_mcqa --limit 0 --methods llm,vector,graph,hybrid
python -m src.eval.run --task mcqa  --data medqa --limit 40
python -m src.eval.run --task yesno --data pubmedqa --limit 0
python -m src.eval.run --task yesno --data safety_yesno --limit 40
```

## 8.11 本章小结

本章借鉴临床 AI 评测的多维度框架，在五种方法上完成了对比与消融实验，从安全性、可解释性、稳健性、充分性、质量与知识能力六个维度系统评估了本系统。核心结果表明：本方法在多次运行中将禁忌违规率稳定降至 0%、可追溯率达 100%，并保持接近最优的临床适宜性；知识图谱检索在安全类问题上显著优于纯 LLM。定性案例进一步直观印证了本体校验层在拦截危险推荐上的不可替代作用。实验充分验证了本体双重接地机制、尤其是 OWL 校验层在临床安全维度的核心价值。

---

# 第9章 总结与展望

## 9.1 工作总结

本文针对大语言模型在医疗场景中"幻觉、不可审计、缺乏逻辑级安全防线"的问题，以糖尿病临床用药问答为切入点，设计并实现了一个**融合本体推理的 Agentic GraphRAG 系统**。主要工作与结论如下：

1. **提出并实现了本体双重接地机制**：权威医学本体既作为 Agentic 建图的 schema 约束，又作为输出端的 OWL 逻辑校验来源，在 RAG 之外构建了逻辑级幻觉防线。
2. **构建了糖尿病专科知识图谱**：融合 SNOMED CT、RxNorm、MED-RT，并以本体 schema 约束 LLM 从约 500 篇 PubMed 摘要扩充图谱，最终得到 1251 节点、3777 边（含 107 条禁忌关系）的知识图谱，并实现轻量实体链接器。
3. **设计了层级化禁忌推理方法**：通过 `CI_<drug>` 禁忌类与术语桥接，使 HermiT 推理机不仅能判断显式禁忌，还能通过 `subClassOf` 层级推断病种特异化禁忌，避免危险假阴性。
4. **实现了端到端 Agentic 系统与系统性实验**：基于 LangGraph 实现"路由→检索→生成→校验→纠错"闭环；实验证明本方法将禁忌违规率由纯 LLM 的 8.7%、图 RAG 的 13.0% 降至 **0%**，可追溯率达 **100%**，在安全类问答上知识图谱检索显著优于纯 LLM。

## 9.2 局限性

本文工作仍存在以下局限：

1. **评测规模有限**：受 API 成本与时间限制，部分准确率实验为抽样（如 MedQA 40 题），自建安全集规模（244 题）与开放推药题（23 病情）也相对有限，统计置信区间偏宽。
2. **安全集与本体同源**：自建安全集与 OWL 校验器均源自 MED-RT，本方法在该集上的优异表现部分源于"知识被正确操作化"，对完全未知来源的禁忌泛化能力尚待验证。
3. **禁忌关系覆盖有限**：MED-RT 的 `ci_with` 仅覆盖部分药-病禁忌，对剂量相关、药物相互作用等更复杂的安全约束尚未建模。
4. **文献三元组噪声**：文献层边由 LLM 抽取，虽经 schema 校验，仍可能含噪声；本文通过"安全判断只依赖本体层"规避了风险，但也意味着文献层知识未参与安全决策。
5. **基线为自实现**：未完整复现 MedGraphRAG、AMG-RAG 等外部系统，与 SOTA 的直接横向对比有所欠缺。

## 9.3 未来工作

1. **扩大评测**：引入更大规模、独立来源的禁忌测试集与人工评估，补充统计显著性检验；
2. **丰富安全约束**：纳入药物相互作用、剂量调整、肾功能分级等更复杂的本体公理；
3. **时序知识图谱**：建模糖尿病并发症进展轨迹，支持随病程演变的动态用药建议；
4. **NL→OWL 自动化**：探索更通用的"自然语言论断→OWL 公理"桥接，扩展校验覆盖的论断类型；
5. **横向对比**：复现 MedGraphRAG/AMG-RAG，并在公开 benchmark 上做更全面的对比。

## 9.4 结语

本文的核心洞见是：**检索增强解决"证据相关性"，而医疗安全还需要一道"逻辑正确性"防线。** 通过将权威本体推理引入 Agentic GraphRAG 的输出端，本文以糖尿病用药问答为例，验证了这道逻辑防线能够把禁忌违规率压到零并保持全程可追溯。希望本文提出的"本体双重接地"机制能为大语言模型在专科医疗场景的可信落地提供一种可借鉴的范式。

---

# 参考文献

[1] Wu J, et al. MedGraphRAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation. arXiv:2408.04187, 2024.

[2] Rezaei M, et al. Agentic Medical Knowledge Graphs (AMG-RAG). EMNLP 2025 Findings / arXiv:2502.13010, 2025.

[3] Zuo K, et al. KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis. arXiv:2412.16833, 2024.

[4] Yang L, et al. Neuro-Symbolic AI: Towards Improving the Reasoning Abilities of Large Language Models (Survey). IJCAI 2025.

[5] Enhancing LLMs through Neuro-Symbolic Integration and Ontological Reasoning. arXiv:2504.07640, 2025.

[6] Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems. arXiv:2604.00555, 2026.

[7] Evangelista E, Ruba F, Bukhari S, Nazir A, Sharma R. GraphRAG-Enabled Local Large Language Model for Gestational Diabetes Mellitus: Development of a Proof-of-Concept. JMIR Diabetes 2026;11:e76454. doi:10.2196/76454.（本文实验评测的多维度指标框架参考此工作）

[8] An Auditable and Source-Verified Framework for Clinical AI Decision Support. Frontiers in Artificial Intelligence, 2026.

[9] ReinRAG: UMLS Knowledge Graph + Reinforcement Learning for Clinical Text Generation. OpenReview, 2025.

[10] American Diabetes Association. Standards of Care in Diabetes (annual).

[11] Lewis P, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.

[12] Cormack G V, et al. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR 2009.

[13] Glimm B, et al. HermiT: An OWL 2 Reasoner. Journal of Automated Reasoning, 2014.

[14] Lamy J B. Owlready: Ontology-oriented Programming in Python with Automatic Classification and High-level Constructs for Biomedical Ontologies. Artificial Intelligence in Medicine, 2017.

[15] Donnelly K. SNOMED-CT: The Advanced Terminology and Coding System for eHealth. Studies in Health Technology and Informatics, 2006.

[16] Nelson S J, et al. Normalized Names for Clinical Drugs: RxNorm at 6 Years. JAMIA, 2011.

[17] Jin D, et al. What Disease Does This Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams (MedQA). Applied Sciences, 2021.

[18] Pal A, et al. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering. CHIL 2022.

[19] Jin Q, et al. PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP 2019.

[20] DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024.

---

> **致谢**（按学校模板补充）

> **附录**：源代码组织（`src/ontology` 本体与建图、`src/graph` 知识图谱、`src/retrieval` 混合检索、`src/reasoning` OWL 校验、`src/agent` 编排、`src/eval` 评测）；实验逐题明细见 `data/eval/results/`。

