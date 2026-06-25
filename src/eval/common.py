"""
评测公共层：数据集归一化加载、选项/答案解析、指标。

把 4 个评测集统一成两种题型：
  - MCQA: {id, question, options:{A..D}, gold:'A', kind, meta}
  - YESNO: {id, question, gold:'yes'/'no', context, kind, meta}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EVAL = Path("data/eval")
LETTERS = ["A", "B", "C", "D", "E"]


def _read(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def load_medqa(limit: int = 0) -> list[dict]:
    rows = _read(EVAL / "medqa" / "diabetes_refined.jsonl")
    out = []
    for i, r in enumerate(rows):
        opts = r.get("options", {})
        if not isinstance(opts, dict) or not opts:
            continue
        out.append({"id": f"medqa_{i}", "kind": "mcqa", "question": r["question"],
                    "options": opts, "gold": r.get("answer_idx", "")})
        if limit and len(out) >= limit:
            break
    return out


def load_medmcqa(limit: int = 0) -> list[dict]:
    rows = _read(EVAL / "medmcqa" / "diabetes_refined.jsonl")
    out = []
    for i, r in enumerate(rows):
        ch = r.get("choices", [])
        if not ch:
            continue
        opts = {LETTERS[j]: str(c) for j, c in enumerate(ch)}
        gold = r.get("answer", "") or (LETTERS[r["answer_index"]] if "answer_index" in r else "")
        out.append({"id": f"medmcqa_{i}", "kind": "mcqa", "question": r["question"],
                    "options": opts, "gold": gold})
        if limit and len(out) >= limit:
            break
    return out


def load_pubmedqa(limit: int = 0) -> list[dict]:
    rows = _read(EVAL / "pubmedqa" / "diabetes_refined.jsonl")
    out = []
    for i, r in enumerate(rows):
        out.append({"id": f"pubmedqa_{i}", "kind": "yesno", "question": r["question"],
                    "context": r.get("context", ""), "gold": (r.get("final_decision", "") or "").lower()})
        if limit and len(out) >= limit:
            break
    return out


def load_safety_mcqa(limit: int = 0) -> list[dict]:
    rows = _read(EVAL / "custom" / "safety_contraindication.jsonl")
    out = []
    for r in rows:
        if r["subtype"] in ("mcqa_avoid", "vignette_avoid") and isinstance(r.get("options"), dict):
            out.append({"id": r["id"], "kind": "mcqa", "question": r["question"],
                        "options": r["options"], "gold": r["answer"],
                        "meta": {"condition": r["condition"], "drug": r["drug"],
                                 "axiom": r.get("violated_axiom")}})
        if limit and len(out) >= limit:
            break
    return out


def load_safety_yesno(limit: int = 0) -> list[dict]:
    rows = _read(EVAL / "custom" / "safety_contraindication.jsonl")
    out = []
    for r in rows:
        if r["subtype"] == "yesno":
            out.append({"id": r["id"], "kind": "yesno", "question": r["question"],
                        "context": "", "gold": r["answer"].lower(),
                        "meta": {"condition": r["condition"], "drug": r["drug"]}})
        if limit and len(out) >= limit:
            break
    return out


LOADERS = {"medqa": load_medqa, "medmcqa": load_medmcqa, "pubmedqa": load_pubmedqa,
           "safety_mcqa": load_safety_mcqa, "safety_yesno": load_safety_yesno}


def parse_letter(text: str, options: dict) -> str:
    """从模型输出解析选项字母。"""
    t = text.strip()
    m = re.search(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-E])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"\(?([A-E])[\).:]", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", t)
    if m and m.group(1).upper() in options:
        return m.group(1).upper()
    return ""


def parse_yesno(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(no|not appropriate|contraindicat|should not|avoid)\b", t[:80]):
        return "no"
    if re.search(r"\b(yes|appropriate|safe|acceptable)\b", t[:80]):
        return "yes"
    if "no" in t[:20]:
        return "no"
    if "yes" in t[:20]:
        return "yes"
    return "maybe" if "maybe" in t else ""


def options_text(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in options.items())
