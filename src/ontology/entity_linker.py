"""
轻量实体链接器（模块1，替代 QuickUMLS）

把文本中的医学术语链接到糖尿病本体子集的概念 ID。
基于 concept_dictionary.csv 的 preferred_name + synonyms 构建词典，做
归一化的整词匹配（最长优先、不重叠）。无需 UMLS / nmslib，纯 Python。

用法（库）:
    from src.ontology.entity_linker import EntityLinker
    el = EntityLinker()
    el.link("A pregnant woman was prescribed glyburide for her diabetes.")
    # -> [{concept_id, preferred_name, node_type, matched, start, end}, ...]

用法（命令行 / 演示）:
    python src/ontology/entity_linker.py
    python src/ontology/entity_linker.py --text "Metformin is contraindicated in renal failure."
    python src/ontology/entity_linker.py --annotate data/raw/pubmed/diabetes_pubmed.jsonl --limit 200
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ONT = Path("data/processed/ontology")
DICT_CSV = ONT / "concept_dictionary.csv"

# 过于宽泛/易误报的词，作为整词链接时忽略（仍可在更长术语中出现）
STOP_TERMS = {
    "disease", "diseases", "disorder", "disorders", "syndrome", "finding",
    "combinations", "other", "drug", "drugs", "injury", "structure",
    "agent", "process", "value", "entity", "type", "human",
}


def _norm(text: str) -> str:
    """归一化：小写、非字母数字转空格、压缩空格。"""
    t = text.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class EntityLinker:
    def __init__(self, dict_csv: Path = DICT_CSV, min_len: int = 3):
        if not dict_csv.exists():
            raise FileNotFoundError(f"缺少 {dict_csv}，请先运行 build_concept_dictionary.py")
        self.concepts: dict[str, dict] = {}
        self.term2cid: dict[str, str] = {}     # 归一化词 -> concept_id
        self._build(dict_csv, min_len)
        # 单条大正则（按长度降序，保证最长优先）
        terms = sorted(self.term2cid.keys(), key=len, reverse=True)
        self._re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")

    def _build(self, dict_csv: Path, min_len: int) -> None:
        for r in csv.DictReader(dict_csv.open(encoding="utf-8")):
            cid = r["concept_id"]
            self.concepts[cid] = r
            names = [r["preferred_name"]] + r["synonyms"].split(" | ")
            for name in names:
                nt = _norm(name)
                if len(nt) < min_len or nt in STOP_TERMS or nt.isdigit():
                    continue
                # 已存在则保留更短术语对应的概念（更具体），否则首次写入
                self.term2cid.setdefault(nt, cid)

    def link(self, text: str) -> list[dict]:
        """返回文本中链接到的概念（基于归一化文本，start/end 为归一化文本下标）。"""
        norm = _norm(text)
        out, seen_spans = [], []
        for m in self._re.finditer(norm):
            s, e = m.span()
            if any(s < pe and ps < e for ps, pe in seen_spans):  # 跳过重叠
                continue
            cid = self.term2cid.get(m.group(0))
            if not cid:
                continue
            c = self.concepts[cid]
            out.append({
                "concept_id": cid, "preferred_name": c["preferred_name"],
                "node_type": c["node_type"], "matched": m.group(0), "start": s, "end": e,
            })
            seen_spans.append((s, e))
        return out


def _demo(el: EntityLinker) -> None:
    samples = [
        "A pregnant woman with gestational diabetes was prescribed glyburide.",
        "Metformin is contraindicated in patients with renal failure and lactic acidosis.",
        "The patient developed diabetic nephropathy and diabetic retinopathy over 10 years.",
        "Empagliflozin (Jardiance) reduced HbA1c in type 2 diabetes mellitus.",
    ]
    for s in samples:
        print("文本:", s)
        for h in el.link(s):
            print(f"   [{h['node_type']:<14}] {h['matched']:<28} -> {h['concept_id']} ({h['preferred_name']})")
        print()


def _annotate(el: EntityLinker, path: str, limit: int) -> None:
    src = Path(path)
    out = src.with_name(src.stem + "_linked.jsonl")
    n, total_ents = 0, 0
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if n >= limit:
                break
            rec = json.loads(line)
            text = (rec.get("title", "") + ". " + rec.get("abstract", "")).strip()
            ents = el.link(text)
            total_ents += len(ents)
            rec["entities"] = ents
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[annotate] {n} 篇, 共链接 {total_ents} 个实体 (均 {total_ents/max(1,n):.1f}/篇) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="轻量实体链接器")
    ap.add_argument("--text", default=None, help="链接单条文本")
    ap.add_argument("--annotate", default=None, help="标注一个 jsonl 文件(title+abstract)")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    el = EntityLinker()
    print(f"词典: {len(el.concepts)} 概念, {len(el.term2cid)} 个可匹配术语\n")
    if args.text:
        for h in el.link(args.text):
            print(f"[{h['node_type']}] {h['matched']} -> {h['concept_id']} ({h['preferred_name']})")
    elif args.annotate:
        _annotate(el, args.annotate, args.limit)
    else:
        _demo(el)


if __name__ == "__main__":
    main()
