"""
评测集下载与糖尿病子集筛选脚本 (国内网络可用版)

由于 huggingface.co 在国内不可访问、hf-mirror 当前失效, 本脚本改用:
  - MedMCQA  : ModelScope  extraordinarylab/medmcqa  (parquet)
  - PubMedQA : GitHub       pubmedqa/pubmedqa  ori_pqal.json (官方标注版, 1000题)
  - MedQA    : ModelScope  AI-ModelScope/med_qa  data_clean.zip (USMLE)

输出: data/eval/<name>/all.jsonl 与 diabetes_subset.jsonl

用法:
    python src/data_acquisition/download_eval_datasets.py
    python src/data_acquisition/download_eval_datasets.py --only medmcqa
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import requests

EVAL_DIR = Path("data/eval")
RAW_DIR = Path("data/raw/eval_src")

MS_BASE = "https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"

# 糖尿病/内分泌相关筛选关键词 (小写匹配)
DIABETES_KEYWORDS = [
    "diabet", "insulin", "glycemi", "glycaemi", "hyperglyc", "hypoglyc",
    "hba1c", "hemoglobin a1c", "metformin", "sulfonylurea", "glp-1", "glp1",
    "sglt2", "sglt-2", "dpp-4", "dpp4", "ketoacidosis", " dka", "hhs",
    "gestational diabetes", "type 1 diabetes", "type 2 diabetes",
    "retinopathy", "nephropathy", "neuropathy", "islet of langerhans",
    "blood glucose", "blood sugar", "glucagon", "glucose tolerance",
]
_KW_RE = re.compile("|".join(re.escape(k) for k in DIABETES_KEYWORDS), re.IGNORECASE)


def is_diabetes_related(text: str) -> bool:
    return bool(_KW_RE.search(text or ""))


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] 已存在 {dest.name}")
        return dest
    print(f"  [get] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  [ok] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def _ms_url(repo: str, path: str) -> str:
    return MS_BASE.format(repo=repo, path=path)


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- MedMCQA (ModelScope parquet) ----------------
def process_medmcqa() -> None:
    print("\n=== MedMCQA (ModelScope: extraordinarylab/medmcqa) ===")
    import pandas as pd

    repo = "extraordinarylab/medmcqa"
    out_dir = EVAL_DIR / "medmcqa"
    sub_rows, total = [], 0
    # 该 parquet 列: question / choices(4选项数组) / answer(字母) / answer_index
    for split, fname in [("train", "data/train-00000-of-00001.parquet"),
                         ("validation", "data/validation-00000-of-00001.parquet")]:
        dest = RAW_DIR / "medmcqa" / Path(fname).name
        _download(_ms_url(repo, fname), dest)
        df = pd.read_parquet(dest)
        total += len(df)
        for _, ex in df.iterrows():
            _ch = ex["choices"] if "choices" in ex else None
            choices = list(_ch) if _ch is not None else []
            blob = str(ex.get("question", "")) + " " + " ".join(str(c) for c in choices)
            if is_diabetes_related(blob):
                sub_rows.append({
                    "split": split,
                    "question": str(ex.get("question", "")),
                    "choices": [str(c) for c in choices],
                    "answer": str(ex.get("answer", "")),
                    "answer_index": int(ex["answer_index"]) if "answer_index" in ex and str(ex["answer_index"]).lstrip("-").isdigit() else ex.get("answer_index", -1),
                })
    _save_jsonl(sub_rows, out_dir / "diabetes_subset.jsonl")
    print(f"  全量 {total} 题, 糖尿病子集 {len(sub_rows)} 题")


# ---------------- PubMedQA (GitHub 官方标注版) ----------------
def process_pubmedqa() -> None:
    print("\n=== PubMedQA (GitHub: pubmedqa/pubmedqa ori_pqal.json) ===")
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    dest = RAW_DIR / "pubmedqa" / "ori_pqal.json"
    _download(url, dest)
    data = json.loads(dest.read_text(encoding="utf-8"))
    out_dir = EVAL_DIR / "pubmedqa"
    all_rows, sub_rows = [], []
    for pubid, ex in data.items():
        contexts = ex.get("CONTEXTS", [])
        ctx_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
        row = {
            "pubid": pubid,
            "question": ex.get("QUESTION", ""),
            "context": ctx_text,
            "long_answer": ex.get("LONG_ANSWER", ""),
            "final_decision": ex.get("final_decision", ""),
        }
        all_rows.append(row)
        if is_diabetes_related(row["question"] + " " + ctx_text):
            sub_rows.append(row)
    _save_jsonl(all_rows, out_dir / "all.jsonl")
    _save_jsonl(sub_rows, out_dir / "diabetes_subset.jsonl")
    print(f"  全量 {len(all_rows)} 题, 糖尿病子集 {len(sub_rows)} 题")


# ---------------- MedQA (ModelScope data_clean.zip) ----------------
def process_medqa() -> None:
    print("\n=== MedQA (ModelScope: AI-ModelScope/med_qa data_clean.zip) ===")
    repo = "AI-ModelScope/med_qa"
    dest = RAW_DIR / "medqa" / "data_clean.zip"
    _download(_ms_url(repo, "data_clean.zip"), dest)

    extract_dir = RAW_DIR / "medqa" / "data_clean"
    if not extract_dir.exists():
        with zipfile.ZipFile(dest) as z:
            z.extractall(RAW_DIR / "medqa")
        print(f"  [unzip] -> {extract_dir}")

    # 优先用美国 USMLE 4 选项版本; 过滤 macOS 的 ._ 元数据垃圾文件
    def _clean(paths):
        return [p for p in paths if not p.name.startswith("._")]

    candidates = _clean((RAW_DIR / "medqa").rglob("US/4_options/*.jsonl"))
    if not candidates:
        candidates = _clean((RAW_DIR / "medqa").rglob("US/*.jsonl"))
    if not candidates:
        candidates = _clean((RAW_DIR / "medqa").rglob("*.jsonl"))
    print(f"  找到 {len(candidates)} 个 jsonl: {[c.name for c in candidates][:6]}")

    out_dir = EVAL_DIR / "medqa"
    all_rows, sub_rows = [], []
    for jf in candidates:
        # 跳过中文 (Mainland/Taiwan)
        if "Mainland" in str(jf) or "Taiwan" in str(jf):
            continue
        split = jf.stem
        with jf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                options = ex.get("options", {})
                row = {
                    "split": split,
                    "question": ex.get("question", ""),
                    "options": options,
                    "answer": ex.get("answer", ""),
                    "answer_idx": ex.get("answer_idx", ""),
                }
                all_rows.append(row)
                opt_text = " ".join(str(v) for v in options.values()) if isinstance(options, dict) else str(options)
                if is_diabetes_related(row["question"] + " " + opt_text):
                    sub_rows.append(row)
    _save_jsonl(all_rows, out_dir / "all.jsonl")
    _save_jsonl(sub_rows, out_dir / "diabetes_subset.jsonl")
    print(f"  全量 {len(all_rows)} 题(含各 split), 糖尿病子集 {len(sub_rows)} 题")


def main() -> None:
    ap = argparse.ArgumentParser(description="下载医学 QA 评测集并筛糖尿病子集(国内源)")
    ap.add_argument("--only", choices=["medqa", "medmcqa", "pubmedqa"], default=None)
    args = ap.parse_args()

    tasks = {"medqa": process_medqa, "medmcqa": process_medmcqa, "pubmedqa": process_pubmedqa}
    if args.only:
        tasks[args.only]()
    else:
        for fn in tasks.values():
            try:
                fn()
            except Exception as e:
                import traceback
                print(f"  [error] {fn.__name__} 失败: {e}")
                traceback.print_exc()
    print("\n[done] 评测集处理完成, 见 data/eval/")


if __name__ == "__main__":
    main()
