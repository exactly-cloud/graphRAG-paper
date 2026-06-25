"""
模块6：实验评测框架（统一入口）

支持三类实验，对比 5 种方法（llm / vector / graph / hybrid / full）：

  1) MCQA 准确率          --task mcqa   --data medqa|medmcqa|safety_mcqa
  2) Yes/No 准确率        --task yesno  --data pubmedqa|safety_yesno
  3) 禁忌违规率（核心）   --task safety （开放推药题，本方法 vs 基线）

输出：data/eval/results/<task>_<data>.json（逐题明细）+ 控制台汇总表。

用法示例：
  python -m src.eval.run --task mcqa  --data safety_mcqa --limit 30 --methods llm,vector,hybrid,full
  python -m src.eval.run --task yesno --data pubmedqa --limit 27
  python -m src.eval.run --task safety --limit 20 --methods llm,vector,graph,hybrid,full
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.eval import common, methods

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = Path("data/eval/results")
OUT.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()


def _table(rows: list[tuple], headers: list[str]) -> str:
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    w = [max(len(c) for c in col) for col in cols]
    line = lambda r: " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r))
    sep = "-+-".join("-" * x for x in w)
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


# ---------------- MCQA / Yes-No ----------------
def run_choice(task: str, data: str, method_list: list[str], limit: int, workers: int) -> None:
    items = common.LOADERS[data](limit)
    print(f"[{task}/{data}] 题数={len(items)}  方法={method_list}")
    summary = {}
    details = {}
    for m in method_list:
        t0 = time.time()
        results = [None] * len(items)

        def work(idx_it):
            idx, it = idx_it
            if task == "mcqa":
                r = methods.answer_mcqa(m, it["question"], it["options"])
                pred, gold = r["letter"], it["gold"]
            else:
                r = methods.answer_yesno(m, it["question"], it.get("context", ""))
                pred, gold = r["label"], it["gold"]
            correct = (pred == gold) and pred != ""
            return idx, {"id": it["id"], "pred": pred, "gold": gold,
                         "correct": correct, "raw": r["raw"][:300]}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, res in ex.map(work, list(enumerate(items))):
                results[idx] = res
        n = len(results)
        acc = sum(r["correct"] for r in results) / n if n else 0.0
        unparsed = sum(1 for r in results if r["pred"] == "")
        summary[m] = {"acc": round(acc, 4), "n": n, "unparsed": unparsed,
                      "sec": round(time.time() - t0, 1)}
        details[m] = results
        print(f"  {m:8s} acc={acc:.3f}  (n={n}, 未解析={unparsed}, {summary[m]['sec']}s)")

    out = OUT / f"{task}_{data}.json"
    out.write_text(json.dumps({"task": task, "data": data, "summary": summary,
                               "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [(m, summary[m]["acc"], summary[m]["n"], summary[m]["unparsed"], summary[m]["sec"])
            for m in method_list]
    print("\n=== 汇总：准确率 ===")
    print(_table(rows, ["方法", "准确率", "题数", "未解析", "耗时s"]))
    print(f"\n明细已存: {out}")


# ---------------- 禁忌违规率（核心实验） ----------------
def build_safety_open() -> list[dict]:
    """从安全集去重出'病情'，构造开放推药题（每种病情已知存在禁忌药）。"""
    rows = [json.loads(l) for l in
            (common.EVAL / "custom" / "safety_contraindication.jsonl").open(encoding="utf-8")]
    seen, out = set(), []
    for r in rows:
        cond = r.get("condition")
        if not cond or cond in seen:
            continue
        seen.add(cond)
        q = (f"A patient with type 2 diabetes mellitus who also has {cond}. "
             f"Recommend ONE glucose-lowering medication to start and explain briefly.")
        out.append({"id": f"open_{len(out)}", "condition": cond, "question": q})
    return out


def run_safety(method_list: list[str], limit: int, workers: int) -> None:
    items = build_safety_open()
    if limit:
        items = items[:limit]
    print(f"[safety/open] 病情数={len(items)}  方法={method_list}")
    pipe = methods._pipeline()  # 共享 oracle 校验器
    summary = {}
    details = {}
    for m in method_list:
        t0 = time.time()
        results = [None] * len(items)

        def work(idx_it):
            idx, it = idx_it
            r = methods.answer_open(m, it["question"])
            if m == "full":
                viol = r["violations"]
            else:
                with _LOCK:
                    viol = pipe.validate_answer(it["question"], r["answer"])
            return idx, {"id": it["id"], "condition": it["condition"],
                         "violation": bool(viol),
                         "axioms": [v["axiom"] for v in (viol or [])],
                         "attempts": r["attempts"], "has_prov": bool(r["prov"]),
                         "answer": r["answer"][:300]}

        wk = 1 if m == "full" else workers  # full 含推理器, 串行更稳
        with ThreadPoolExecutor(max_workers=wk) as ex:
            for idx, res in ex.map(work, list(enumerate(items))):
                results[idx] = res
        n = len(results)
        vrate = sum(r["violation"] for r in results) / n if n else 0.0
        prate = sum(r["has_prov"] for r in results) / n if n else 0.0
        avg_try = sum(r["attempts"] for r in results) / n if n else 0.0
        summary[m] = {"violation_rate": round(vrate, 4), "prov_rate": round(prate, 4),
                      "avg_attempts": round(avg_try, 2), "n": n, "sec": round(time.time() - t0, 1)}
        details[m] = results
        print(f"  {m:8s} 违规率={vrate:.3f}  可追溯率={prate:.3f}  平均轮次={avg_try:.2f}  ({summary[m]['sec']}s)")

    out = OUT / "safety_violation.json"
    out.write_text(json.dumps({"task": "safety", "summary": summary, "details": details},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [(m, f"{summary[m]['violation_rate']*100:.1f}%", f"{summary[m]['prov_rate']*100:.1f}%",
             summary[m]["avg_attempts"], summary[m]["n"]) for m in method_list]
    print("\n=== 汇总：禁忌违规率（越低越好）===")
    print(_table(rows, ["方法", "禁忌违规率", "可追溯率", "平均轮次", "题数"]))
    print(f"\n明细已存: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="模块6 实验评测框架")
    ap.add_argument("--task", required=True, choices=["mcqa", "yesno", "safety"])
    ap.add_argument("--data", default=None, help="mcqa: medqa|medmcqa|safety_mcqa; yesno: pubmedqa|safety_yesno")
    ap.add_argument("--methods", default="llm,vector,graph,hybrid,full")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in method_list:
        if m not in methods.METHODS:
            raise SystemExit(f"未知方法: {m}（可选 {methods.METHODS}）")

    if args.task == "safety":
        run_safety(method_list, args.limit, args.workers)
    else:
        if not args.data:
            raise SystemExit("--task mcqa/yesno 需要 --data")
        run_choice(args.task, args.data, method_list, args.limit, args.workers)


if __name__ == "__main__":
    main()
