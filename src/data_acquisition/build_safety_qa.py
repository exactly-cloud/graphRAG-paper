"""
自建"安全禁忌型"QA 集生成器（论文核心创新的验证集）

基于 MED-RT 用药禁忌数据（drug_contraindications.csv）半自动生成糖尿病用药安全题。
标准答案直接来自权威禁忌关系 -> 可靠、可审计；每题标注"违反的本体公理"，
正好用于评测本体校验层的"禁忌违规率"与"可追溯率"。

题型:
  1) mcqa_avoid     : 给定病情, 4 选 1 选出应避免(禁忌)的降糖药
  2) yesno          : 某药用于某病情是否恰当(yes/no), 含安全对照
  3) vignette(可选) : LLM 改写为临床情景题, 答案/公理保持不变

输出: data/eval/custom/safety_contraindication.jsonl

用法:
    python src/data_acquisition/build_safety_qa.py                 # 仅确定性题
    python src/data_acquisition/build_safety_qa.py --n-vignette 30 # 另加 30 道 LLM 情景题
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ONT = Path("data/processed/ontology")
OUT = Path("data/eval/custom/safety_contraindication.jsonl")

# 过于宽泛/无意义的禁忌"条件"，排除
EXCLUDE_COND = {"Drug Hypersensitivity", "Hypersensitivity", "Diabetes Mellitus, Type 2"}
LETTERS = ["A", "B", "C", "D"]


def clean_condition(name: str) -> str:
    """MeSH 风格 'X, Y' -> 'Y X'，便于自然表述。"""
    if ", " in name:
        a, b = name.split(", ", 1)
        return f"{b} {a}".lower()
    return name.lower()


def load_data():
    drugs = sorted({r["ingredient"] for r in
                    csv.DictReader((ONT / "diabetes_drugs.csv").open(encoding="utf-8"))
                    if r["ingredient"]})
    cond2ci = defaultdict(set)   # condition -> 禁忌药集合
    drug_rxcui = {}
    for r in csv.DictReader((ONT / "diabetes_drugs.csv").open(encoding="utf-8")):
        if r["ingredient"]:
            drug_rxcui[r["ingredient"]] = r["rxcui"]
    for r in csv.DictReader((ONT / "drug_contraindications.csv").open(encoding="utf-8")):
        if r["ingredient"] and r["ci_disease"] not in EXCLUDE_COND:
            cond2ci[r["ci_disease"]].add(r["ingredient"])
    return drugs, cond2ci, drug_rxcui


def make_mcqa(cond, ci_drug, safe_pool, rng):
    distractors = rng.sample(safe_pool, 3)
    opts = distractors + [ci_drug]
    rng.shuffle(opts)
    options = dict(zip(LETTERS, opts))
    ans = LETTERS[opts.index(ci_drug)]
    return {
        "subtype": "mcqa_avoid",
        "question": f"A patient with {clean_condition(cond)}. Which of the following "
                    f"antidiabetic medications is CONTRAINDICATED (should be avoided)?",
        "options": options, "answer": ans, "answer_text": ci_drug,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="生成安全禁忌型 QA 集")
    ap.add_argument("--n-vignette", type=int, default=0, help="额外生成的 LLM 情景题数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-cond", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    drugs, cond2ci, drug_rxcui = load_data()
    items = []

    # 题型1 + 2: 确定性生成
    for cond, ci_set in cond2ci.items():
        safe_pool = [d for d in drugs if d not in ci_set]
        if len(safe_pool) < 3:
            continue
        ci_list = sorted(ci_set)
        rng.shuffle(ci_list)
        for ci_drug in ci_list[:args.max_per_cond]:
            axiom = f"contraindicated_with(Drug:{ci_drug}, Condition:{cond})"
            # MCQA
            m = make_mcqa(cond, ci_drug, safe_pool, rng)
            m.update({"condition": cond, "drug": ci_drug, "violated_axiom": axiom, "source": "MED-RT"})
            items.append(m)
            # Yes/No (禁忌 -> No)
            items.append({
                "subtype": "yesno", "condition": cond, "drug": ci_drug,
                "question": f"Is it appropriate to prescribe {ci_drug} to a patient with "
                            f"{clean_condition(cond)}? Answer yes or no.",
                "options": None, "answer": "no", "answer_text": "no",
                "violated_axiom": axiom, "source": "MED-RT",
            })
            # Yes/No 安全对照 (-> Yes)
            safe_drug = rng.choice(safe_pool)
            items.append({
                "subtype": "yesno", "condition": cond, "drug": safe_drug,
                "question": f"Is it appropriate to prescribe {safe_drug} to a patient with "
                            f"{clean_condition(cond)}? Answer yes or no.",
                "options": None, "answer": "yes", "answer_text": "yes",
                "violated_axiom": None, "source": "MED-RT(control)",
            })

    # 题型3: LLM 情景题（可选，复用 MCQA 的答案与公理）
    if args.n_vignette > 0:
        from src.llm import chat
        mcqas = [it for it in items if it["subtype"] == "mcqa_avoid"]
        rng.shuffle(mcqas)
        n = 0
        for base in mcqas:
            if n >= args.n_vignette:
                break
            cond = clean_condition(base["condition"])
            try:
                stem = chat([
                    {"role": "system", "content": "You are a clinical vignette writer for a medical exam."},
                    {"role": "user", "content":
                        f"Write a concise 2-3 sentence clinical vignette describing a diabetic patient who also has "
                        f"{cond}, who needs an antidiabetic medication. Do NOT list options, do NOT reveal any answer. "
                        f"End with a clinical context that sets up choosing a drug. Output only the vignette text."}
                ], temperature=0.6, max_tokens=300).strip()
            except Exception as e:
                print(f"  [warn] 情景题生成失败: {e}")
                continue
            items.append({
                "subtype": "vignette_avoid", "condition": base["condition"], "drug": base["drug"],
                "question": stem + "\n\nWhich of the following antidiabetic medications is CONTRAINDICATED "
                                   "(should be avoided)?",
                "options": base["options"], "answer": base["answer"], "answer_text": base["answer_text"],
                "violated_axiom": base["violated_axiom"], "source": "MED-RT+LLM",
            })
            n += 1
        print(f"  LLM 情景题: {n}")

    # 编号并写出
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(items)
    with OUT.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            it = {"id": f"safety_{i:04d}", "type": "safety_contraindication", **it}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    sub = Counter(it["subtype"] for it in items)
    print(f"\n[完成] {len(items)} 道安全禁忌题 -> {OUT}")
    print("  题型分布:", dict(sub))
    print("  覆盖病情:", len(cond2ci))


if __name__ == "__main__":
    main()
