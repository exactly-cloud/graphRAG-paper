"""
SNOMED CT 糖尿病专科本体子集裁剪 (对应开题报告 4.3)

从糖尿病顶层概念 Diabetes mellitus (SCTID: 73211009) 出发, 沿 is-a 关系
(typeId=116680003) 递归向下抽出整棵子树, 得到"糖尿病专科本体子集"。

输入: SNOMED CT International RF2 Snapshot 三表
  - sct2_Concept_Snapshot      概念(id/active/...)
  - sct2_Description_Snapshot   描述(conceptId/typeId/term, FSN=全限定名)
  - sct2_Relationship_Snapshot  关系(sourceId/destinationId/typeId)

输出 (data/processed/ontology/):
  - diabetes_concepts.csv        子集概念: sctid, fsn, semantic_tag
  - diabetes_relationships.csv   子集内部关系: source/type/dest (含 FSN)
  - diabetes_subset_sctids.txt   子集概念 ID 列表

用法:
    python src/ontology/extract_snomed_subset.py
    python src/ontology/extract_snomed_subset.py --root 73211009 --root 73211009
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

IS_A = "116680003"             # is-a 关系
# 并发症/因果关系类型: 这些"指向糖尿病"的概念本身就是糖尿病并发症
COMPLICATION_TYPES = {
    "42752001": "Due to",
    "47429007": "Associated with",
    "255234002": "After",
}
FSN_TYPE = "900000000000003001"  # Fully Specified Name
SNOMED_BASE = "data/raw/snomed"
OUT_DIR = Path("data/processed/ontology")

DIABETES_ROOT = "73211009"     # Diabetes mellitus (disorder)


def _find(pattern: str) -> str:
    hits = glob.glob(f"{SNOMED_BASE}/**/{pattern}", recursive=True)
    if not hits:
        sys.exit(f"[错误] 找不到 RF2 文件: {pattern} (确认 SNOMED 已解压到 {SNOMED_BASE})")
    return hits[0]


def load_active_concepts(path: str) -> set[str]:
    active = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[2] == "1":  # active
                active.add(c[0])
    print(f"  活跃概念: {len(active):,}")
    return active


def load_isa_children(path: str) -> dict[str, list[str]]:
    """返回 parent_id -> [child_id], 仅活跃 is-a 关系 (source is-a destination)。"""
    children = defaultdict(list)
    n = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[2] == "1" and c[7] == IS_A:  # active & is-a
                children[c[5]].append(c[4])  # destination(parent) -> source(child)
                n += 1
    print(f"  活跃 is-a 关系: {n:,}")
    return children


def collect_descendants(roots: list[str], children: dict[str, list[str]]) -> set[str]:
    subset = set(roots)
    dq = deque(roots)
    while dq:
        cur = dq.popleft()
        for ch in children.get(cur, []):
            if ch not in subset:
                subset.add(ch)
                dq.append(ch)
    return subset


def load_fsn(path: str, needed: set[str] | None = None) -> dict[str, str]:
    """conceptId -> FSN 术语 (仅活跃 FSN 描述)。needed 为 None 时加载全部。"""
    fsn = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            # id,effectiveTime,active,moduleId,conceptId,languageCode,typeId,term,caseSig
            if c[2] == "1" and c[6] == FSN_TYPE:
                cid = c[4]
                if needed is None or cid in needed:
                    fsn[cid] = c[7]
    return fsn


def semantic_tag(fsn: str) -> str:
    m = re.search(r"\(([^()]+)\)\s*$", fsn or "")
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="裁剪 SNOMED 糖尿病本体子集")
    ap.add_argument("--root", action="append", default=None,
                    help="子树根 SCTID, 可多次指定; 默认 73211009 (Diabetes mellitus)")
    args = ap.parse_args()
    roots = args.root or [DIABETES_ROOT]

    concept_f = _find("sct2_Concept_Snapshot*.txt")
    desc_f = _find("sct2_Description_Snapshot*.txt")
    rel_f = _find("sct2_Relationship_Snapshot*.txt")

    print("[1/5] 读取活跃概念 ...")
    active = load_active_concepts(concept_f)

    print("[2/5] 读取 is-a 关系并构建层级 ...")
    children = load_isa_children(rel_f)

    print(f"[3/5] 从根 {roots} 递归抽取 is-a 子树 ...")
    core = collect_descendants(roots, children)
    core = {c for c in core if c in active}  # 核心: 糖尿病疾病子树
    print(f"  核心概念(疾病子树): {len(core):,}")

    print("[4/5] 纳入并发症(Due to/Associated with→糖尿病)与定义性关系邻居 ...")
    # 4a: 通过 Due to/Associated with/After 指向核心疾病的概念 = 糖尿病并发症
    complications = set()
    with open(rel_f, encoding="utf-8") as fin:
        next(fin)
        for line in fin:
            c = line.rstrip("\n").split("\t")
            if c[2] != "1":
                continue
            src, dst, typ = c[4], c[5], c[7]
            if typ in COMPLICATION_TYPES and dst in core and src in active and src not in core:
                complications.add(src)
    expanded = core | complications
    print(f"  并发症概念: {len(complications):,}")

    # 4b: 保留所有"源在 expanded 内"的活跃关系; 目标概念作为邻居纳入(病灶/形态等)
    edges = []  # (src, typ, dst, grp)
    neighbors = set()
    with open(rel_f, encoding="utf-8") as fin:
        next(fin)
        for line in fin:
            c = line.rstrip("\n").split("\t")
            if c[2] != "1":
                continue
            src, dst, grp, typ = c[4], c[5], c[6], c[7]
            if src in expanded and dst in active:
                edges.append((src, typ, dst, grp))
                if dst not in expanded:
                    neighbors.add(dst)
    subset = expanded | neighbors
    print(f"  邻居概念(病灶/形态等): {len(neighbors):,}")
    print(f"  扩展后子集总数: {len(subset):,}, 关系: {len(edges):,}")
    core = expanded  # 把并发症并入核心(in_core=1)

    print("[5/5] 读取 FSN 术语并写出 ...")
    fsn_all = load_fsn(desc_f)  # 全量, 便于给关系类型与目标命名
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 概念表 (含 in_core 标记: 是否为糖尿病疾病本体核心)
    with (OUT_DIR / "diabetes_concepts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sctid", "fsn", "semantic_tag", "in_core"])
        for cid in sorted(subset, key=lambda x: (x not in core, x)):
            name = fsn_all.get(cid, "")
            w.writerow([cid, name, semantic_tag(name), int(cid in core)])

    (OUT_DIR / "diabetes_subset_sctids.txt").write_text(
        "\n".join(sorted(subset)), encoding="utf-8")

    # 关系表
    tag_counter = defaultdict(int)
    with (OUT_DIR / "diabetes_relationships.csv").open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["source_id", "source_fsn", "type_id", "type_fsn", "dest_id", "dest_fsn", "group"])
        for src, typ, dst, grp in edges:
            tname = fsn_all.get(typ, typ)
            w.writerow([src, fsn_all.get(src, ""), typ, tname, dst, fsn_all.get(dst, ""), grp])
            tag_counter[re.sub(r"\s*\([^)]*\)$", "", tname)] += 1

    # 统计输出
    print(f"\n[完成] 输出目录: {OUT_DIR}")
    print(f"  概念: {len(subset):,} 个 (核心疾病 {len(core)}, 邻居 {len(neighbors)})")
    print(f"  关系: {len(edges):,} 条")
    tags = defaultdict(int)
    for cid in subset:
        tags[semantic_tag(fsn_all.get(cid, ""))] += 1
    print("  概念语义类别(前10):")
    for t, n in sorted(tags.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t or '(无)':<24} {n}")
    print("  关系类型(前10):")
    for t, n in sorted(tag_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t:<32} {n}")


if __name__ == "__main__":
    main()
