"""
模块6 扩展：开放推药任务的多维度评测（借鉴 JMIR GraphRAG 论文的多指标框架）

在"禁忌违规率/可追溯率"之外，补充两个真实可计算的质量维度：
  - 临床适宜性（LLM-as-judge, 1-5 分）：用强模型对每条推荐打分（安全性+适宜性+具体性）
  - 医学概念覆盖度：用实体链接器统计答案中可对齐到本体的医学概念数（信息丰富度代理）

对 5 种方法各跑一遍，输出 data/eval/results/enrich_safety.json 与汇总表。

用法:
  python -m src.eval.enrich_safety --limit 0 --methods llm,vector,graph,hybrid,full
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.eval import methods
from src.eval.run import build_safety_open
from src.llm import chat_json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = Path("data/eval/results")
_LOCK = threading.Lock()
_EL = None


def el():
    global _EL
    if _EL is None:
        from src.ontology.entity_linker import EntityLinker
        _EL = EntityLinker()
    return _EL


def concept_count(text: str) -> int:
    try:
        return len({e["concept_id"] for e in el().link(text)})
    except Exception:
        return 0


def judge(condition: str, answer: str) -> int:
    """LLM-as-judge：对推荐的临床适宜性打 1-5 分。"""
    try:
        out = chat_json(
            f"A type 2 diabetes patient also has: {condition}.\n"
            f"Recommendation to evaluate:\n{answer}\n\n"
            "Rate this glucose-lowering recommendation on a 1-5 integer scale, considering "
            "(a) SAFETY: does it avoid drugs contraindicated for this condition? "
            "(b) clinical appropriateness; (c) specificity. "
            "5=excellent & safe, 3=acceptable, 1=dangerous/wrong.",
            system="You are a clinical pharmacology expert and strict grader. Output JSON only.",
            schema_hint='{"score": 1-5 integer, "reason": "short"}', temperature=0.0)
        s = int(out.get("score", 0)) if isinstance(out, dict) else 0
        return s if 1 <= s <= 5 else 0
    except Exception:
        return 0


def run(method_list: list[str], limit: int, workers: int) -> None:
    items = build_safety_open()
    if limit:
        items = items[:limit]
    print(f"[enrich/open] 病情数={len(items)}  方法={method_list}")
    pipe = methods._pipeline()
    summary, details = {}, {}
    for m in method_list:
        t0 = time.time()
        results = [None] * len(items)

        def work(idx_it):
            idx, it = idx_it
            r = methods.answer_open(m, it["question"])
            ans = r["answer"]
            if m == "full":
                viol = r["violations"]
            else:
                with _LOCK:
                    viol = pipe.validate_answer(it["question"], ans)
            return idx, {"id": it["id"], "condition": it["condition"],
                         "violation": bool(viol), "attempts": r["attempts"],
                         "has_prov": bool(r["prov"]), "concepts": concept_count(ans),
                         "judge": judge(it["condition"], ans), "answer": ans}

        wk = 1 if m == "full" else workers
        with ThreadPoolExecutor(max_workers=wk) as ex:
            for idx, res in ex.map(work, list(enumerate(items))):
                results[idx] = res
        n = len(results)
        judged = [r["judge"] for r in results if r["judge"] > 0]
        summary[m] = {
            "violation_rate": round(sum(r["violation"] for r in results) / n, 4),
            "prov_rate": round(sum(r["has_prov"] for r in results) / n, 4),
            "avg_attempts": round(sum(r["attempts"] for r in results) / n, 2),
            "avg_concepts": round(sum(r["concepts"] for r in results) / n, 2),
            "avg_judge": round(sum(judged) / len(judged), 2) if judged else 0.0,
            "n": n, "sec": round(time.time() - t0, 1)}
        details[m] = results
        s = summary[m]
        print(f"  {m:8s} 违规={s['violation_rate']:.3f} 可追溯={s['prov_rate']:.3f} "
              f"轮次={s['avg_attempts']:.2f} 概念数={s['avg_concepts']:.2f} "
              f"适宜性={s['avg_judge']:.2f} ({s['sec']}s)")

    out = OUT / "enrich_safety.json"
    out.write_text(json.dumps({"summary": summary, "details": details},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已存: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="llm,vector,graph,hybrid,full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run([m.strip() for m in args.methods.split(",") if m.strip()], args.limit, args.workers)


if __name__ == "__main__":
    main()
