"""
建图 step 1：本体骨架入库

把模块1产出的统一概念词典与本体边载入图结构（networkx），作为知识图谱的
"可信骨架层"（layer=ontology）。后续 Agentic 文献抽取的三元组（layer=literature）
会在此基础上增量合并。

输入:
  data/processed/ontology/concept_dictionary.csv
  data/processed/ontology/ontology_edges.csv
输出:
  data/processed/graph/nodes.csv
  data/processed/graph/edges.csv
  data/processed/graph/kg.graphml      (networkx 持久化，供检索/可视化)
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import networkx as nx

ONT = Path("data/processed/ontology")
OUT = Path("data/processed/graph")


def load_skeleton() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    with (ONT / "concept_dictionary.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g.add_node(
                r["concept_id"],
                node_type=r["node_type"],
                preferred_name=r["preferred_name"],
                source=r["source"],
                category=r.get("category", ""),
                synonyms=r.get("synonyms", ""),
            )
    with (ONT / "ontology_edges.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # 保证两端节点存在（边表里偶有未在词典登记的目标，补一个占位节点）
            for nid, nm in ((r["source_id"], r["source_name"]), (r["target_id"], r["target_name"])):
                if nid and nid not in g:
                    g.add_node(nid, node_type="Unknown", preferred_name=nm,
                               source=r.get("vocab", ""), category="", synonyms="")
            g.add_edge(
                r["source_id"], r["target_id"], key=r["edge_type"],
                edge_type=r["edge_type"], layer="ontology",
                confidence=1.0, evidence="", vocab=r.get("vocab", ""),
            )
    return g


def save_graph(g: nx.MultiDiGraph) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "nodes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept_id", "node_type", "preferred_name", "source", "category", "synonyms"])
        for nid, d in g.nodes(data=True):
            w.writerow([nid, d.get("node_type", ""), d.get("preferred_name", ""),
                        d.get("source", ""), d.get("category", ""), d.get("synonyms", "")])
    with (OUT / "edges.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "edge_type", "target_id", "layer", "confidence", "evidence", "vocab"])
        for u, v, d in g.edges(data=True):
            w.writerow([u, d.get("edge_type", ""), v, d.get("layer", ""),
                        d.get("confidence", ""), d.get("evidence", ""), d.get("vocab", "")])
    nx.write_graphml(g, OUT / "kg.graphml")


def main() -> None:
    g = load_skeleton()
    save_graph(g)
    print(f"[完成] 本体骨架图 -> {OUT}")
    print(f"  节点: {g.number_of_nodes():,}  边: {g.number_of_edges():,}")
    print("  节点类型:", dict(Counter(d["node_type"] for _, d in g.nodes(data=True))))
    print("  边类型:", dict(Counter(d["edge_type"] for *_, d in g.edges(data=True)).most_common(10)))


if __name__ == "__main__":
    main()
