"""
建图 step 3：Agentic 本体校验 + 合并入库

对 step2 抽出的候选三元组做本体约束校验，再合并进知识图谱:
  R1 实体对齐   : 要求 head/tail 均已对齐到本体概念（--keep-unlinked 可放宽）
  R2 类型约束   : 关系的 from/to 端点类型须符合 schema（如 treats 只能 Drug->Disease）
  R3 自环去除   : head==tail 丢弃
  R4 冲突消解   : 与本体骨架冲突的丢弃（如文献说 treats，本体说 contraindicated_with
                  → 本体权威，丢弃文献边并记录）
  R5 去重/强化  : 同一三元组多次出现则聚合证据并提升置信度；与骨架已有边一致则视为"加强"

通过的边以 layer=literature 合并入图，写出更新后的 nodes/edges/graphml，
并产出校验报告与被拒样本（供论文做误差分析）。

用法:
    python src/graph/validate_merge.py
    python src/graph/validate_merge.py --keep-unlinked   # 也纳入未对齐的新文献实体
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.graph.build_skeleton import load_skeleton, save_graph
from src.graph.schema import relation_allowed

GRAPH = Path("data/processed/graph")
CAND = GRAPH / "candidate_triples.jsonl"

# 互斥关系对（同一对端点上不能共存 -> 冲突）
MUTEX = {("treats", "contraindicated_with"), ("contraindicated_with", "treats"),
         ("prevents", "causes"), ("causes", "prevents")}
BASE_CONF = 0.70   # 文献边基础置信度
REINFORCE = 0.20   # 与本体一致 / 多证据的加成


def main() -> None:
    ap = argparse.ArgumentParser(description="本体校验 + 合并入库")
    ap.add_argument("--keep-unlinked", action="store_true", help="纳入未对齐到本体的新实体")
    args = ap.parse_args()

    g = load_skeleton()
    # 骨架已有边: (u,v) -> set(rel)
    existing = defaultdict(set)
    for u, v, d in g.edges(data=True):
        existing[(u, v)].add(d["edge_type"])

    cands = [json.loads(l) for l in CAND.open(encoding="utf-8")]
    print(f"[校验] 候选三元组 {len(cands)} 条")

    # 聚合同一三元组（跨摘要）的证据
    agg: dict[tuple, dict] = {}
    rejected = []

    def reject(c, reason):
        rejected.append({**c, "reject_reason": reason})

    for c in cands:
        h, t, rel = c.get("head_id"), c.get("tail_id"), c["relation"]
        # R1 对齐
        if not (h and t):
            if not args.keep_unlinked:
                reject(c, "unlinked")
                continue
            # 放宽：为未对齐端创建文献节点
            h = h or f"LIT:{c['head_text'][:40].lower().strip()}"
            t = t or f"LIT:{c['tail_text'][:40].lower().strip()}"
            for nid, name, typ, txt in ((c.get('head_id'), c.get('head_name'), c.get('head_type'), c['head_text']),
                                        (c.get('tail_id'), c.get('tail_name'), c.get('tail_type'), c['tail_text'])):
                lid = nid or f"LIT:{txt[:40].lower().strip()}"
                if lid not in g:
                    g.add_node(lid, node_type=typ or "Unknown", preferred_name=name or txt,
                               source="LITERATURE", category="", synonyms="")
        # R3 自环
        if h == t:
            reject(c, "self_loop")
            continue
        h_type = g.nodes[h]["node_type"] if h in g else (c.get("head_type") or "Unknown")
        t_type = g.nodes[t]["node_type"] if t in g else (c.get("tail_type") or "Unknown")
        # R2 类型约束
        if not relation_allowed(rel, h_type, t_type):
            reject(c, f"type_violation({h_type}-{rel}->{t_type})")
            continue
        # R4 冲突消解：与骨架互斥
        conflict = any((rel, er) in MUTEX for er in existing.get((h, t), ()))
        if conflict:
            reject(c, "conflict_with_ontology")
            continue
        # 通过 -> 聚合
        key = (h, rel, t)
        if key not in agg:
            agg[key] = {"head_id": h, "relation": rel, "tail_id": t,
                        "head_name": g.nodes[h]["preferred_name"] if h in g else c.get("head_name"),
                        "tail_name": g.nodes[t]["preferred_name"] if t in g else c.get("tail_name"),
                        "pmids": set(), "evidences": [], "in_ontology": rel in existing.get((h, t), ())}
        a = agg[key]
        a["pmids"].add(c.get("pmid", ""))
        if c.get("evidence"):
            a["evidences"].append(c["evidence"])

    # 计算置信度并合并入图
    lit_edges = []
    added = 0
    for (h, rel, t), a in agg.items():
        conf = BASE_CONF
        if a["in_ontology"]:
            conf += REINFORCE          # 与本体一致 -> 加强
        if len(a["pmids"]) > 1:
            conf += 0.10               # 多文献佐证
        conf = round(min(conf, 0.95), 2)
        a["confidence"] = conf
        a["pmids"] = sorted(p for p in a["pmids"] if p)
        a["n_evidence"] = len(a["evidences"])
        # 已在本体中的关系不重复加边，仅记录"被文献加强"
        if not a["in_ontology"]:
            g.add_edge(h, t, key=rel, edge_type=rel, layer="literature",
                       confidence=conf, evidence=" || ".join(a["evidences"][:3]),
                       pmids=",".join(a["pmids"]), vocab="PubMed")
            added += 1
        lit_edges.append({**a, "evidences": a["evidences"][:5]})

    save_graph(g)
    with (GRAPH / "literature_edges.jsonl").open("w", encoding="utf-8") as f:
        for e in lit_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (GRAPH / "rejected_triples.jsonl").open("w", encoding="utf-8") as f:
        for e in rejected:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\n[完成] 校验合并")
    print(f"  通过(去重后唯一三元组): {len(agg)}  其中新增文献边: {added}  加强已有本体边: {len(agg)-added}")
    print(f"  被拒: {len(rejected)}  原因:", dict(Counter(r['reject_reason'].split('(')[0] for r in rejected)))
    print(f"  合并后图: 节点 {g.number_of_nodes():,}, 边 {g.number_of_edges():,}")
    print(f"  产出: literature_edges.jsonl / rejected_triples.jsonl / kg.graphml")


if __name__ == "__main__":
    main()
