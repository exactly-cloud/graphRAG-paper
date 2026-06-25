"""
构建统一概念词典 + 本体边表（模块1 收口）

合并三份产物为知识图谱的"节点 + 边"规范层：
  - diabetes_concepts.csv        (SNOMED 糖尿病概念)
  - diabetes_drugs.csv           (RxNorm 降糖药)
  - drug_contraindications.csv   (MED-RT 用药禁忌)
  - diabetes_relationships.csv   (SNOMED 定义性关系)

并依据 configs/relation_schema.yaml 把节点/关系归入统一 schema；
为 SNOMED 概念补充同义词（供后续实体链接）。

输出 (data/processed/ontology/):
  - concept_dictionary.csv : concept_id, source, node_type, preferred_name, category, synonyms
  - ontology_edges.csv     : source_id, source_name, edge_type, target_id, target_name, vocab

用法:
    python src/ontology/build_concept_dictionary.py
"""
from __future__ import annotations

import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import yaml

ONT = Path("data/processed/ontology")
SNOMED_BASE = "data/raw/snomed"
SCHEMA = Path("configs/relation_schema.yaml")
DM_ROOT = "73211009"  # Diabetes mellitus

FSN_TYPE = "900000000000003001"
SYN_TYPE = "900000000000013009"


def load_schema():
    s = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    return s["snomed_tag_map"], s["snomed_rel_map"]


def load_snomed_synonyms(subset: set[str]) -> dict[str, set[str]]:
    """为子集概念收集活跃同义词（含 FSN 去括号名）。"""
    desc = glob.glob(f"{SNOMED_BASE}/**/sct2_Description_Snapshot*.txt", recursive=True)[0]
    syn = defaultdict(set)
    with open(desc, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            # id,eff,active,mod,conceptId,lang,typeId,term,caseSig
            if c[2] == "1" and c[4] in subset and c[6] in (FSN_TYPE, SYN_TYPE):
                term = c[7]
                if c[6] == FSN_TYPE:
                    term = re.sub(r"\s*\([^)]*\)$", "", term)  # 去语义标签
                syn[c[4]].add(term)
    return syn


def main() -> None:
    tag_map, rel_map = load_schema()
    concepts_path = ONT / "diabetes_concepts.csv"
    subset = {r["sctid"] for r in csv.DictReader(concepts_path.open(encoding="utf-8"))}

    print(f"[1/4] 读取 SNOMED 同义词（{len(subset)} 概念）...")
    syn = load_snomed_synonyms(subset)

    nodes = {}   # concept_id -> dict
    edges = []   # dict

    print("[2/4] SNOMED 概念 -> 节点 ...")
    for r in csv.DictReader(concepts_path.open(encoding="utf-8")):
        sctid, fsn, tag = r["sctid"], r["fsn"], r["semantic_tag"]
        ntype = tag_map.get(tag, "Other")
        pref = re.sub(r"\s*\([^)]*\)$", "", fsn)
        is_compl = ntype == "Disease" and ("diabetic" in pref.lower() or "diabetes" in pref.lower()) and sctid != DM_ROOT
        nodes[f"SCTID:{sctid}"] = {
            "concept_id": f"SCTID:{sctid}", "source": "SNOMEDCT", "node_type": ntype,
            "preferred_name": pref,
            "category": "Complication" if is_compl else tag,
            "synonyms": " | ".join(sorted(syn.get(sctid, {pref}))),
        }

    print("[3/4] RxNorm 降糖药 -> Drug 节点; 加 treats 边 ...")
    for r in csv.DictReader((ONT / "diabetes_drugs.csv").open(encoding="utf-8")):
        if not r["ingredient"]:
            continue
        cid = f"RXCUI:{r['rxcui']}"
        brands = [b.strip() for b in r["brand_names"].split(";") if b.strip()]
        nodes[cid] = {
            "concept_id": cid, "source": "RXNORM", "node_type": "Drug",
            "preferred_name": r["ingredient"], "category": r["atc_class"],
            "synonyms": " | ".join([r["ingredient"]] + brands),
        }
        # 降糖药 treats 糖尿病
        edges.append({"source_id": cid, "source_name": r["ingredient"], "edge_type": "treats",
                      "target_id": f"SCTID:{DM_ROOT}", "target_name": "Diabetes mellitus", "vocab": "ATC"})

    print("[4/4] 禁忌 -> contraindicated_with 边 + MeSH 疾病节点; SNOMED 关系映射 ...")
    for r in csv.DictReader((ONT / "drug_contraindications.csv").open(encoding="utf-8")):
        if not r["ingredient"]:
            continue
        drug_id = f"RXCUI:{r['rxcui']}"
        dis_id = f"MESH:{r['ci_code']}"
        if dis_id not in nodes:
            nodes[dis_id] = {"concept_id": dis_id, "source": "MEDRT", "node_type": "Disease",
                             "preferred_name": r["ci_disease"], "category": "ci_target",
                             "synonyms": r["ci_disease"]}
        edges.append({"source_id": drug_id, "source_name": r["ingredient"], "edge_type": "contraindicated_with",
                      "target_id": dis_id, "target_name": r["ci_disease"], "vocab": "MEDRT"})

    # SNOMED 定义性关系
    for r in csv.DictReader((ONT / "diabetes_relationships.csv").open(encoding="utf-8")):
        rel = re.sub(r"\s*\([^)]*\)$", "", r["type_fsn"])
        etype = rel_map.get(rel, rel.lower().replace(" ", "_"))
        edges.append({"source_id": f"SCTID:{r['source_id']}", "source_name": re.sub(r'\s*\([^)]*\)$','',r['source_fsn']),
                      "edge_type": etype, "target_id": f"SCTID:{r['dest_id']}",
                      "target_name": re.sub(r'\s*\([^)]*\)$','',r['dest_fsn']), "vocab": "SNOMEDCT"})

    # 写出
    with (ONT / "concept_dictionary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept_id", "source", "node_type", "preferred_name", "category", "synonyms"])
        w.writeheader()
        w.writerows(nodes.values())
    with (ONT / "ontology_edges.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "source_name", "edge_type", "target_id", "target_name", "vocab"])
        w.writeheader()
        w.writerows(edges)

    # 统计
    from collections import Counter
    nt = Counter(n["node_type"] for n in nodes.values())
    et = Counter(e["edge_type"] for e in edges)
    print(f"\n[完成] 概念词典 {len(nodes)} 节点, 边表 {len(edges)} 条 -> {ONT}")
    print("  节点类型:", dict(nt))
    print("  边类型(前12):")
    for e, n in et.most_common(12):
        print(f"    {e:<26} {n}")


if __name__ == "__main__":
    main()
