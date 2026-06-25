"""
模块3 step 3：图检索器

问题 → 实体链接命中本体节点（种子）→ 在知识图谱上做 1~2 跳遍历 →
把路径上的边转成"事实句"返回。擅长多跳临床逻辑（如禁忌、并发症因果链）。

事实排序优先级：跳数近 > 本体层(ontology) > 高置信。

用法（库）:
    from src.retrieval.graph_retriever import GraphRetriever
    gr = GraphRetriever()
    gr.search("Is metformin safe in renal failure?", k=10)
用法（命令行）:
    python src/retrieval/graph_retriever.py --q "metformin contraindication renal failure"
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.ontology.entity_linker import EntityLinker

GRAPH = Path("data/processed/graph")
LAYER_RANK = {"ontology": 0, "literature": 1}


def humanize(etype: str) -> str:
    return etype.replace("_", " ")


class GraphRetriever:
    def __init__(self, graph_dir: Path = GRAPH):
        self.name: dict[str, str] = {}
        self.ntype: dict[str, str] = {}
        with (graph_dir / "nodes.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.name[r["concept_id"]] = r["preferred_name"]
                self.ntype[r["concept_id"]] = r["node_type"]
        # 无向邻接：node -> [(neighbor, etype, layer, conf, forward?)]
        self.adj: dict[str, list] = defaultdict(list)
        with (graph_dir / "edges.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                u, v, et = r["source_id"], r["target_id"], r["edge_type"]
                lay = r.get("layer", ""); conf = float(r.get("confidence") or 0)
                self.adj[u].append((v, et, lay, conf, True))
                self.adj[v].append((u, et, lay, conf, False))
        self.el = EntityLinker()

    def seeds(self, query: str) -> list[dict]:
        seen, out = set(), []
        for e in self.el.link(query):
            if e["concept_id"] not in seen and e["concept_id"] in self.name:
                seen.add(e["concept_id"]); out.append(e)
        return out

    def search(self, query: str, k: int = 10, max_hops: int = 2) -> list[dict]:
        seeds = [s["concept_id"] for s in self.seeds(query)]
        if not seeds:
            return []
        visited = set(seeds)
        frontier = set(seeds)
        facts: dict[tuple, dict] = {}
        for hop in range(1, max_hops + 1):
            nxt = set()
            for node in frontier:
                for (other, et, lay, conf, fwd) in self.adj.get(node, []):
                    h, t = (node, other) if fwd else (other, node)
                    key = (h, et, t)
                    if key not in facts:
                        facts[key] = {
                            "head": self.name.get(h, h), "edge_type": et,
                            "tail": self.name.get(t, t), "layer": lay,
                            "confidence": conf, "hop": hop, "source": "graph",
                            "fact": f"{self.name.get(h, h)} [{humanize(et)}] {self.name.get(t, t)}",
                        }
                    if other not in visited:
                        nxt.add(other)
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        ranked = sorted(facts.values(),
                        key=lambda x: (x["hop"], LAYER_RANK.get(x["layer"], 9), -x["confidence"]))
        for i, f in enumerate(ranked[:k], 1):
            f["rank"] = i
        return ranked[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description="图检索（多跳事实）")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()
    gr = GraphRetriever()
    s = gr.seeds(args.q)
    print("种子实体:", [f"{x['preferred_name']}" for x in s] or "(无)")
    for r in gr.search(args.q, args.k, args.hops):
        print(f"[{r['rank']}] (hop{r['hop']},{r['layer']},c={r['confidence']}) {r['fact']}")


if __name__ == "__main__":
    main()
