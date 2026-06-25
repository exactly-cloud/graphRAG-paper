"""
建图 step 4（可选）：把图文件导入 Neo4j

读取 data/processed/graph/{nodes.csv, edges.csv}，写入 Neo4j：
  - 节点 label = node_type，属性 concept_id/preferred_name/source/...
  - 关系 type = edge_type（大写），属性 layer/confidence/evidence/pmids
连接信息取自 .env（NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD）。

前置：本地需有运行中的 Neo4j（Desktop 或 Docker:
    docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/yourpass neo4j:5）

用法:
    python src/graph/load_neo4j.py
    python src/graph/load_neo4j.py --wipe     # 先清空再导入
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GRAPH = Path("data/processed/graph")


def read_csv(p: Path):
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="导入知识图谱到 Neo4j")
    ap.add_argument("--wipe", action="store_true", help="导入前清空数据库")
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "")
    if not pwd:
        raise RuntimeError("请在 .env 设置 NEO4J_PASSWORD")

    nodes = read_csv(GRAPH / "nodes.csv")
    edges = read_csv(GRAPH / "edges.csv")
    print(f"读取 {len(nodes)} 节点, {len(edges)} 边")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as s:
        if args.wipe:
            s.run("MATCH (n) DETACH DELETE n")
            print("已清空数据库")
        s.run("CREATE CONSTRAINT concept_id IF NOT EXISTS "
              "FOR (n:Concept) REQUIRE n.concept_id IS UNIQUE")

        # 节点（统一 :Concept 标签 + 动态二级标签 node_type）
        q_node = (
            "UNWIND $rows AS r "
            "MERGE (n:Concept {concept_id: r.concept_id}) "
            "SET n += {preferred_name:r.preferred_name, node_type:r.node_type, "
            "source:r.source, category:r.category, synonyms:r.synonyms}"
        )
        for i in range(0, len(nodes), args.batch):
            s.run(q_node, rows=nodes[i:i + args.batch])
        print("节点导入完成")

        # 关系：用真实 edge_type 作为 Neo4j 关系类型（Browser 直接显示关系名）。
        # 关系类型不能参数化，故按 edge_type 分组拼接（edge_type 取值均为安全标识符）。
        by_type: dict[str, list] = {}
        for e in edges:
            by_type.setdefault(e["edge_type"].upper(), []).append(e)
        ok = 0
        for etype, rows in by_type.items():
            safe = "".join(c for c in etype if c.isalnum() or c == "_") or "REL"
            q_edge = (
                "UNWIND $rows AS r "
                "MATCH (a:Concept {concept_id:r.source_id}) "
                "MATCH (b:Concept {concept_id:r.target_id}) "
                f"MERGE (a)-[e:`{safe}` {{layer:r.layer}}]->(b) "
                "SET e.edge_type=r.edge_type, e.confidence=toFloat(r.confidence), "
                "e.evidence=r.evidence, e.vocab=r.vocab"
            )
            for i in range(0, len(rows), args.batch):
                s.run(q_edge, rows=rows[i:i + args.batch]).consume()
                ok += len(rows[i:i + args.batch])
        print(f"关系导入完成 ({ok}), 关系类型: {sorted(by_type)}")

    driver.close()
    print("[完成] 已导入 Neo4j：", uri)
    print("提示：Neo4j Browser 查询  MATCH (a)-[e]->(b) RETURN a,e,b LIMIT 100")


if __name__ == "__main__":
    main()
