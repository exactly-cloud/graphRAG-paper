"""
模块3 step 4：混合检索（向量 + 图）+ RRF 融合

并行跑向量检索（PubMed 摘要）与图检索（KG 多跳事实），用 Reciprocal Rank
Fusion (RRF) 把两路排序结果融合成统一证据上下文，供下游 LLM 生成 + 本体校验使用。

RRF: score(item) = Σ_list 1 / (rrf_k + rank_in_list)，rrf_k 默认 60。

用法（库）:
    from src.retrieval.hybrid import HybridRetriever
    hr = HybridRetriever()
    ctx = hr.retrieve("Is glyburide safe in pregnancy?")
    print(hr.format_context(ctx))
用法（命令行）:
    python src/retrieval/hybrid.py --q "Which antidiabetic should be avoided in pregnancy?"
"""
from __future__ import annotations

import argparse

from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.vector_store import VectorRetriever

RRF_K = 60


class HybridRetriever:
    def __init__(self):
        self.vr = VectorRetriever()
        self.gr = GraphRetriever()

    def retrieve(self, query: str, k_vec: int = 8, k_graph: int = 12,
                 k_final: int = 10, max_hops: int = 2) -> list[dict]:
        vec = self.vr.search(query, k_vec)
        gph = self.gr.search(query, k_graph, max_hops)

        pool: dict[str, dict] = {}

        def add(items):
            for it in items:
                if it["source"] == "vector":
                    key = f"V:{it['pmid']}"
                    content = f"{it['title']} (PMID {it['pmid']})"
                    prov = {"pmid": it["pmid"], "score": round(it["score"], 3)}
                else:
                    key = f"G:{it['head']}|{it['edge_type']}|{it['tail']}"
                    content = it["fact"]
                    prov = {"layer": it["layer"], "confidence": it["confidence"], "hop": it["hop"]}
                node = pool.setdefault(key, {"source": it["source"], "content": content,
                                             "prov": prov, "rrf": 0.0, "raw": it})
                node["rrf"] += 1.0 / (RRF_K + it["rank"])

        add(vec)
        add(gph)
        fused = sorted(pool.values(), key=lambda x: -x["rrf"])
        for i, f in enumerate(fused[:k_final], 1):
            f["rank"] = i
        return fused[:k_final]

    @staticmethod
    def format_context(fused: list[dict]) -> str:
        """渲染成给 LLM 的证据上下文（区分文献证据 [V] 与图谱事实 [G]）。"""
        lines = []
        for f in fused:
            if f["source"] == "vector":
                lines.append(f"[V{f['rank']}] (文献) {f['content']}")
            else:
                p = f["prov"]
                lines.append(f"[G{f['rank']}] (图谱·{p['layer']},置信{p['confidence']}) {f['content']}")
        return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="混合检索 + RRF 融合")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    hr = HybridRetriever()
    fused = hr.retrieve(args.q, k_final=args.k)
    print(f"查询: {args.q}\n种子实体:", [s["preferred_name"] for s in hr.gr.seeds(args.q)] or "(无)")
    print("\n=== RRF 融合证据 ===")
    print(hr.format_context(fused))


if __name__ == "__main__":
    main()
