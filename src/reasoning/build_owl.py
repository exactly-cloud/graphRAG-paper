"""
模块4 step 1：知识图谱 → OWL 本体

把图谱转成 OWL（owlready2），供 HermiT 做 DL 推理。核心设计：
  - 每个概念 → 一个 OWL 类（label=preferred_name）
  - is_a 边 → subClassOf（构建疾病层级，HermiT 可推理上下位闭包）
  - 本体对齐桥接：把 MED-RT 的 MeSH 禁忌病种连到 SNOMED 子层级
    （如"4期糖尿病肾病" ⊑ Kidney Diseases），让禁忌沿层级向下传播
  - 禁忌标记类 CI_<drug>：把"药物 d 的所有禁忌病种"设为 CI_<drug> 的子类，
    于是 HermiT 能推出"任何更具体的病种 ⊑ CI_<drug>" → 校验时一次 subsumption 查询即可
  - treats / contraindicated_with 也作为对象属性写入（保证 OWL 语义完整、可导出）

输出:
  data/processed/reasoning/diabetes.owl          (RDF/XML)
  data/processed/reasoning/owl_index.json        (concept_id↔类名 + 各药禁忌信息, 供校验器)

用法:
    python src/reasoning/build_owl.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from owlready2 import ObjectProperty, Thing, get_ontology, types

GRAPH = Path("data/processed/graph")
OUT = Path("data/processed/reasoning")
IRI = "http://example.org/diabetes.owl"

# 本体对齐桥接规则：SNOMED 病名关键词 → MeSH 禁忌病种 concept_id
BRIDGE_RULES = [
    (["kidney", "nephropath", "glomerulosclerosis", "renal", "glomerular"], ["MESH:D007674"]),
    (["chronic kidney disease stage 3", "chronic kidney disease stage 4",
      "chronic kidney disease stage 5", "stage 5 on dialysis"], ["MESH:D051437", "MESH:D007676"]),
    (["pregnan", "gestational", "puerperium", "childbirth", "postpartum"], ["MESH:D011247"]),
    (["ketoacidosis"], ["MESH:D016883"]),
    (["lactic acidosis"], ["MESH:D000138"]),
]


def cls_name(cid: str) -> str:
    return "C_" + re.sub(r"[^A-Za-z0-9]", "_", cid)


def main() -> None:
    nodes = {r["concept_id"]: r for r in
             csv.DictReader((GRAPH / "nodes.csv").open(encoding="utf-8"))}
    edges = list(csv.DictReader((GRAPH / "edges.csv").open(encoding="utf-8")))

    onto = get_ontology(IRI)
    cmap = {}  # concept_id -> owl class

    with onto:
        # 1) 概念 → 类
        for cid, r in nodes.items():
            c = types.new_class(cls_name(cid), (Thing,))
            c.label = [r["preferred_name"]]
            cmap[cid] = c

        # 2) is_a → subClassOf（仅用权威本体层，SNOMED 分类无环；防环保护）
        def safe_subclass(ch, pa) -> bool:
            if not ch or not pa or ch is pa or pa in ch.ancestors() or ch in pa.ancestors():
                return False
            try:
                ch.is_a.append(pa)
                return True
            except TypeError:
                return False

        n_isa = 0
        for e in edges:
            if e["edge_type"] == "is_a" and e.get("layer") == "ontology":
                if safe_subclass(cmap.get(e["source_id"]), cmap.get(e["target_id"])):
                    n_isa += 1

        # 3) 本体对齐桥接：SNOMED 病种 → MeSH 禁忌病种
        n_bridge = 0
        bridges = []
        for cid, r in nodes.items():
            if r["source"] != "SNOMEDCT" or r["node_type"] != "Disease":
                continue
            name = r["preferred_name"].lower()
            for kws, targets in BRIDGE_RULES:
                if any(k in name for k in kws):
                    for t in targets:
                        if t in cmap and safe_subclass(cmap[cid], cmap[t]):
                            n_bridge += 1
                            bridges.append({"snomed": cid, "snomed_name": r["preferred_name"],
                                            "mesh": t, "mesh_name": nodes[t]["preferred_name"]})

        # 4) 对象属性 treats / contraindicated_with（语义完整性）
        class treats(ObjectProperty):
            pass

        class contraindicated_with(ObjectProperty):
            pass

        # 5) 禁忌标记类 CI_<drug>：药物的禁忌病种都设为其子类
        drug_ci = defaultdict(list)  # drug_id -> [ci disease ids]
        for e in edges:
            if e["edge_type"] == "contraindicated_with":
                drug_ci[e["source_id"]].append(e["target_id"])
            if e["edge_type"] == "treats":
                d, t = cmap.get(e["source_id"]), cmap.get(e["target_id"])
                if d and t:
                    d.is_a.append(treats.some(t))

        ci_class_of = {}
        for did, cis in drug_ci.items():
            if did not in cmap:
                continue
            ci_cls = types.new_class("CI_" + re.sub(r"[^A-Za-z0-9]", "_", did), (Thing,))
            ci_cls.label = [f"ContraindicatedCondition_of_{nodes[did]['preferred_name']}"]
            ci_class_of[did] = ci_cls
            for c in cis:
                if c in cmap and safe_subclass(cmap[c], ci_cls):  # 禁忌病种 ⊑ CI_<drug>
                    cmap[did].is_a.append(contraindicated_with.some(cmap[c]))

    OUT.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(OUT / "diabetes.owl"), format="rdfxml")

    index = {
        "iri": IRI,
        "cls_of": {cid: cls_name(cid) for cid in nodes},
        "name_of": {cid: nodes[cid]["preferred_name"] for cid in nodes},
        "ci_class_of": {did: ("CI_" + re.sub(r"[^A-Za-z0-9]", "_", did)) for did in ci_class_of},
        "drug_ci": {did: cis for did, cis in drug_ci.items() if did in cmap},
        "bridges": bridges,
    }
    (OUT / "owl_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[完成] OWL 本体 -> {OUT/'diabetes.owl'}")
    print(f"  类(概念): {len(cmap):,}")
    print(f"  subClassOf(is_a): {n_isa:,}")
    print(f"  对齐桥接边: {n_bridge}  (SNOMED病种→MeSH禁忌病种)")
    print(f"  禁忌标记类 CI_<drug>: {len(ci_class_of)}")


if __name__ == "__main__":
    main()
