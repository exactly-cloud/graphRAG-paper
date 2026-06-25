"""
模块3 step 2：向量检索器

加载 FAISS 索引 + 元数据，对查询做 embedding 后返回最相似的 top-k 篇 PubMed 摘要。
擅长"叙述型/语义相似"问题。

用法（库）:
    from src.retrieval.vector_store import VectorRetriever
    vr = VectorRetriever()
    vr.search("management of diabetic nephropathy", k=5)
用法（命令行）:
    python src/retrieval/vector_store.py --q "treatment of diabetic retinopathy"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from src.llm import embed

DIR = Path("data/processed/retrieval")


class VectorRetriever:
    def __init__(self, index_dir: Path = DIR):
        idx = index_dir / "pubmed.index"
        meta = index_dir / "pubmed_meta.jsonl"
        if not idx.exists():
            raise FileNotFoundError(f"缺少 {idx}，请先运行 build_vector_index.py")
        self.index = faiss.read_index(str(idx))
        self.meta = [json.loads(l) for l in meta.open(encoding="utf-8")]

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = np.array(embed([query]), dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        out = []
        for rank, (i, sc) in enumerate(zip(ids[0], scores[0]), 1):
            if i < 0:
                continue
            m = self.meta[i]
            out.append({"rank": rank, "score": float(sc), "pmid": m["pmid"],
                        "title": m["title"], "text": m["text"], "source": "vector"})
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="向量检索 PubMed 摘要")
    ap.add_argument("--q", required=True, help="查询文本")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    vr = VectorRetriever()
    for r in vr.search(args.q, args.k):
        print(f"[{r['rank']}] sim={r['score']:.3f} PMID={r['pmid']}  {r['title'][:90]}")


if __name__ == "__main__":
    main()
