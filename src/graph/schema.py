"""
本体 Schema 加载与校验工具（供 Agentic 抽取/校验复用）

读取 configs/relation_schema.yaml，提供:
  EDGE_TYPES   : {relation: {"from": set, "to": set}}
  NODE_TYPES   : set
  relation_allowed(rel, from_type, to_type) -> bool
  CORE_RELATIONS : 给 LLM 抽取用的关系清单（带中文说明）
"""
from __future__ import annotations

from pathlib import Path

import yaml

SCHEMA_PATH = Path("configs/relation_schema.yaml")

# 供 LLM 抽取的关系定义（控制抽取范围，避免发散）
CORE_RELATIONS = {
    "treats": "drug/therapy treats a disease",
    "contraindicated_with": "drug must be avoided in a disease/finding (safety)",
    "causes": "a factor causes a disease/complication",
    "due_to": "a disease is caused by / due to another condition",
    "risk_factor_for": "a factor increases risk of a disease",
    "prevents": "an intervention prevents a disease/complication",
    "worsens": "a factor worsens / accelerates a disease",
    "symptom_of": "a clinical finding is a symptom of a disease",
    "associated_with": "two conditions are statistically/clinically associated",
    "is_a": "subtype / is a kind of",
}


def _split_key(k: str):
    return [p.strip() for p in k.split("/")]


def load_schema():
    raw = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    node_types = set(raw.get("node_types", {}).keys())
    node_types.update({"Complication", "Unknown", "Other"})  # 兼容子类/占位
    edge_types: dict[str, dict] = {}
    for k, v in raw.get("edge_types", {}).items():
        froms = set(v.get("from", [])) if isinstance(v, dict) else set()
        tos = set(v.get("to", [])) if isinstance(v, dict) else set()
        for name in _split_key(k):
            edge_types[name] = {"from": froms, "to": tos}
    return node_types, edge_types, raw


NODE_TYPES, EDGE_TYPES, RAW_SCHEMA = load_schema()


def relation_allowed(rel: str, from_type: str | None = None, to_type: str | None = None) -> bool:
    """校验关系是否合法；给定端点类型时还校验 from/to 约束。"""
    if rel not in EDGE_TYPES:
        return False
    spec = EDGE_TYPES[rel]
    # 占位/未知类型放行（文献新实体常无精确类型），仅做关系名校验
    if from_type and from_type not in ("Unknown", "Other") and spec["from"]:
        if from_type not in spec["from"]:
            return False
    if to_type and to_type not in ("Unknown", "Other") and spec["to"]:
        if to_type not in spec["to"]:
            return False
    return True


if __name__ == "__main__":
    print("节点类型:", sorted(NODE_TYPES))
    print("关系类型:", sorted(EDGE_TYPES))
    print("treats 合法(Drug->Disease):", relation_allowed("treats", "Drug", "Disease"))
    print("treats 非法(Disease->Drug):", relation_allowed("treats", "Disease", "Drug"))
