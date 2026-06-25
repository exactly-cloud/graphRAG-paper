"""
模块3 step 1：构建 PubMed 摘要的 FAISS 向量索引

对每篇摘要 (title + abstract) 用硅基流动 bge-m3 做 embedding，归一化后建
FAISS 内积索引（等价余弦相似度），供向量检索（叙述型问题）使用。

输出:
  data/processed/retrieval/pubmed.index   (FAISS 索引)
  data/processed/retrieval/pubmed_meta.jsonl  (pmid/title/text，与索引行号对齐)

用法:
    python src/retrieval/build_vector_index.py
    python src/retrieval/build_vector_index.py --limit 500 --batch 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from src.llm import embed

PUBMED = Path("data/raw/pubmed/diabetes_pubmed.jsonl")
OUT = Path("data/processed/retrieval")


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 PubMed 向量索引")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇 (0=全部)")
    ap.add_argument("--batch", type=int, default=64, help="每批 embedding 文本数")
    args = ap.parse_args()

    rows = []
    with PUBMED.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get("title", "") + ". " + r.get("abstract", "")).strip()
            if len(text) < 30:
                continue
            rows.append({"pmid": r.get("pmid", ""), "title": r.get("title", ""), "text": text})
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[向量化] {len(rows)} 篇摘要, 批大小 {args.batch} ...")

    vecs = []
    for i in range(0, len(rows), args.batch):
        batch = [r["text"][:4000] for r in rows[i:i + args.batch]]
        vecs.extend(embed(batch))
        if (i // args.batch) % 5 == 0:
            print(f"  进度 {min(i + args.batch, len(rows))}/{len(rows)}")

    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)               # 归一化 -> 内积=余弦
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)

    OUT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT / "pubmed.index"))
    with (OUT / "pubmed_meta.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[完成] 索引 {index.ntotal} 条, 维度 {arr.shape[1]} -> {OUT}")


if __name__ == "__main__":
    main()
