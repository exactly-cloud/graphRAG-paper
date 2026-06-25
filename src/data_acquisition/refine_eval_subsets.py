"""
公开评测集二次精筛脚本

问题: 一次筛选(关键词召回)会混入"只是顺带提糖尿病"的噪声题, 例如:
  - 患者既往史有"糖尿病", 但考点其实是泌尿系感染处理;
  - 选项里含"retinopathy"但题目问的是其他眼科病。

本脚本对 data/eval/<name>/diabetes_subset.jsonl 做精筛, 用"糖尿病强相关词"
并结合"出现位置"判断该题是否真正以糖尿病为考点, 输出 diabetes_refined.jsonl。

判定逻辑 (MCQA: MedQA / MedMCQA):
  保留当且仅当满足任一:
    (a) 选项中出现强相关词 (答案选项围绕糖尿病 -> 考点必然是糖尿病);
    (b) 题干"最后一句(真正发问句)"出现强相关词;
    (c) 题干中强相关词出现 >= 2 次 (糖尿病被反复提及, 多为主题)。
PubMedQA:
    研究问题(question)本身出现强相关词才保留 (上下文出现不算)。

用法:
    python src/data_acquisition/refine_eval_subsets.py
    python src/data_acquisition/refine_eval_subsets.py --only medqa --dump-dropped
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_DIR = Path("data/eval")

# 糖尿病"强相关"词 (高精度: 出现基本可断定与糖尿病直接相关)
STRONG_TERMS = [
    r"diabet",                       # diabetes / diabetic
    r"insulin", r"metformin",
    r"sulfonylurea", r"sulphonylurea", r"glipizide", r"glyburide",
    r"glibenclamide", r"glimepiride", r"gliclazide",
    r"pioglitazone", r"rosiglitazone", r"thiazolidinedione",
    r"sitagliptin", r"saxagliptin", r"linagliptin", r"vildagliptin",
    r"dpp-?4",
    r"empagliflozin", r"dapagliflozin", r"canagliflozin", r"sglt-?2",
    r"liraglutide", r"semaglutide", r"dulaglutide", r"exenatide",
    r"glp-?1", r"acarbose",
    r"hba1c", r"glycated h[ae]moglobin", r"glycosylated h[ae]moglobin",
    r"glyc[ae]mi", r"hyperglyc", r"hypoglyc",
    r"ketoacidosis", r"\bdka\b", r"hyperosmolar hyperglyc", r"\bhhs\b",
    r"prediabet", r"pre-diabet",
    r"impaired glucose tolerance", r"impaired fasting glucose",
    r"\bislet", r"c-peptide",
    r"gestational diabet",
    # 并发症: 仅当带 "diabetic" 前缀才算强相关
    r"diabetic retinopath", r"diabetic nephropath", r"diabetic neuropath",
    r"diabetic foot", r"diabetic macul", r"diabetic kidney",
]
_STRONG_RE = re.compile("|".join(STRONG_TERMS), re.IGNORECASE)


def count_strong(text: str) -> int:
    return len(_STRONG_RE.findall(text or ""))


def last_question_sentence(question: str) -> str:
    """取题干中真正发问的句子(优先含 '?' 的最后一句)。"""
    q = (question or "").strip()
    parts = re.split(r"(?<=[.?!])\s+", q)
    parts = [p for p in parts if p.strip()]
    if not parts:
        return q
    for p in reversed(parts):
        if "?" in p:
            return p
    return parts[-1]


def keep_mcqa(question: str, options_text: str) -> tuple[bool, str]:
    strong_q = count_strong(question)
    strong_opt = count_strong(options_text)
    strong_last = count_strong(last_question_sentence(question))
    if strong_opt >= 1:
        return True, "option_hit"
    if strong_last >= 1:
        return True, "question_sentence_hit"
    if strong_q >= 2:
        return True, "multi_mention"
    return False, "incidental_mention"


def keep_pubmedqa(question: str) -> tuple[bool, str]:
    return (count_strong(question) >= 1), ("question_hit" if count_strong(question) >= 1 else "context_only")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def refine_dataset(name: str, dump_dropped: bool) -> None:
    sub = EVAL_DIR / name / "diabetes_subset.jsonl"
    if not sub.exists():
        print(f"  [skip] 缺少 {sub}")
        return
    rows = _read_jsonl(sub)
    kept, dropped = [], []
    for r in rows:
        if name == "medqa":
            opt = r.get("options", {})
            opt_text = " ".join(str(v) for v in opt.values()) if isinstance(opt, dict) else str(opt)
            ok, reason = keep_mcqa(r.get("question", ""), opt_text)
        elif name == "medmcqa":
            opt_text = " ".join(str(c) for c in r.get("choices", []))
            ok, reason = keep_mcqa(r.get("question", ""), opt_text)
        elif name == "pubmedqa":
            ok, reason = keep_pubmedqa(r.get("question", ""))
        else:
            ok, reason = True, "na"
        r2 = dict(r)
        r2["_refine_reason"] = reason
        (kept if ok else dropped).append(r2)

    _write_jsonl(kept, EVAL_DIR / name / "diabetes_refined.jsonl")
    if dump_dropped:
        _write_jsonl(dropped, EVAL_DIR / name / "diabetes_dropped.jsonl")
    rate = 100 * len(kept) / max(1, len(rows))
    print(f"  {name:9s}: {len(rows):5d} -> 保留 {len(kept):5d} ({rate:.0f}%), 剔除 {len(dropped)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="评测集糖尿病子集二次精筛")
    ap.add_argument("--only", choices=["medqa", "medmcqa", "pubmedqa"], default=None)
    ap.add_argument("--dump-dropped", action="store_true", help="同时导出被剔除的题(便于人工抽检)")
    args = ap.parse_args()

    names = [args.only] if args.only else ["medqa", "medmcqa", "pubmedqa"]
    print("精筛结果 (diabetes_subset -> diabetes_refined):")
    for n in names:
        refine_dataset(n, args.dump_dropped)
    print("\n[done] 精筛集见 data/eval/<name>/diabetes_refined.jsonl")


if __name__ == "__main__":
    main()
