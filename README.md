# 融合本体推理的 Agentic GraphRAG（糖尿病临床问答）

> Ontology-Reasoning-Enhanced Agentic GraphRAG for Clinical Question Answering in Diabetes

毕业论文配套代码仓库。开题报告见 [`开题报告.md`](开题报告.md)。

## 环境

- Python 3.11（conda 环境名 `graphrag`）
- 创建环境：

```bash
conda create -n graphrag python=3.11 -y
conda activate graphrag
pip install -r requirements.txt   # 当前仅启用阶段0(数据获取)依赖
```

> 注意：本机默认 Python 为 3.14，过新，scispaCy/torch/faiss 等暂无对应轮子，故统一用 conda 的 3.11 环境。

## 目录结构

```
data/
  raw/        umls/ snomed/ pubmed/ guidelines/   # 原始数据(不入库)
  processed/  ontology/ kg/                        # 加工产物
  eval/       medqa/ medmcqa/ pubmedqa/ custom/    # 评测集
src/
  data_acquisition/   数据获取脚本(本阶段)
  ontology/           模块1 本体对齐层
  kg_construction/    模块2 Agentic 建图
  retrieval/          模块3 混合检索
  reasoning/          模块4 OWL 本体校验(核心创新)
  orchestration/      模块5 LangGraph 编排
configs/              配置
```

## 数据获取（阶段0，当前进度）

1. **评测集**（无需 License，可立即运行）：

```bash
python src/data_acquisition/download_eval_datasets.py
```

下载 MedQA / MedMCQA / PubMedQA，并在各自目录生成 `diabetes_subset.jsonl`。

2. **PubMed 糖尿病语料**（建议先在 `.env` 填 `NCBI_API_KEY`）：

```bash
cp .env.example .env   # 然后编辑填入 NCBI_EMAIL / NCBI_API_KEY
python src/data_acquisition/fetch_pubmed.py --max 2000
```

3. **UMLS / SNOMED CT**（需 UMLS License）：
   - 下载 `umls 2026AA-full.zip` 后用自带 MetamorphoSys 安装，**仅勾选** `SNOMEDCT_US` / `RXNORM` / `ICD10CM`；
   - 安装产物（`MRCONSO.RRF` 等）放到 `data/raw/umls/2026AA/META`。

## 后续阶段

见 `开题报告.md` 第三章五大模块与第七章进度计划。
