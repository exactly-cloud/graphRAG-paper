"""
模块4 step 3：本体校验 + 自纠错回环（端到端安全问答）

流程（论文核心创新的可演示形态）：
  问题 → (模块3)混合检索 → LLM 生成答案 → 解析答案"推荐用药"论断
       → (模块4)本体禁忌校验(DL 层级推理) → 若违规：反馈解释→LLM 重生成(回环)
       → 输出带 provenance 的安全答案

解析策略：病人病情用实体链接从"问题"取（可靠）；推荐药用 LLM 从"答案"抽
（能正确区分"避免用 X"与"推荐用 X"），再用实体链接落到概念 ID 后交校验器。

用法:
    python src/reasoning/pipeline.py --q "A pregnant woman with GDM needs glucose-lowering medication. Which drug do you recommend?"
"""
from __future__ import annotations

import argparse

from src.llm import chat, chat_json
from src.ontology.entity_linker import EntityLinker
from src.reasoning.validator import OntologyValidator
from src.retrieval.hybrid import HybridRetriever


def extract_recommended_drugs(question: str, answer: str) -> list[str]:
    """从答案中抽取'明确推荐开具'的药物名（排除被建议避免的）。"""
    try:
        out = chat_json(
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
            "List ONLY the medications that the answer RECOMMENDS prescribing/using. "
            "Exclude any drug the answer says to AVOID or that is contraindicated.",
            system="You extract structured info. Output JSON only.",
            schema_hint='{"recommended_drugs": ["drug name", ...]}', temperature=0.0)
        return out.get("recommended_drugs", []) if isinstance(out, dict) else []
    except Exception:
        return []


class SafeQAPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.validator = OntologyValidator()
        self.el = EntityLinker()

    def patient_conditions(self, question: str) -> list[str]:
        seen, out = set(), []
        for e in self.el.link(question):
            if e["node_type"] in ("Disease", "Finding") and e["concept_id"] not in seen:
                seen.add(e["concept_id"]); out.append((e["preferred_name"], e["concept_id"]))
        return out

    def validate_answer(self, question: str, answer: str) -> list[dict]:
        drugs = extract_recommended_drugs(question, answer)
        conds = self.patient_conditions(question)
        violations = []
        for d in drugs:
            for cname, cid in conds:
                r = self.validator.check_texts(d, cname)
                if r.get("violation"):
                    violations.append(r)
        return violations

    def answer(self, question: str, max_retries: int = 2, inject: str | None = None) -> dict:
        ctx = self.retriever.retrieve(question, k_final=10)
        ctx_text = self.retriever.format_context(ctx)
        base_sys = ("You are a clinical assistant for diabetes care. Answer concisely using the "
                    "evidence. Recommend safe glucose-lowering therapy.")
        msgs = [{"role": "system", "content": base_sys},
                {"role": "user", "content": f"Evidence:\n{ctx_text}\n\nQuestion: {question}"}]
        history = []
        for attempt in range(max_retries + 1):
            # inject: 第1轮用注入的(模拟较弱模型的)危险答案, 演示拦截+纠错
            if attempt == 0 and inject:
                ans = inject
            else:
                ans = chat(msgs, temperature=0.3)
            viol = self.validate_answer(question, ans)
            history.append({"attempt": attempt + 1, "answer": ans, "violations": viol})
            if not viol:
                return {"final_answer": ans, "ok": True, "attempts": attempt + 1,
                        "history": history, "context": ctx}
            # 构造反馈，要求避开禁忌药并重生成
            fb = "; ".join(f"{v['drug']} is contraindicated in {v['condition']} "
                           f"(axiom: {v['axiom']}, source {v['source']})" for v in viol)
            msgs.append({"role": "assistant", "content": ans})
            msgs.append({"role": "user", "content":
                         f"SAFETY VIOLATION detected by ontology reasoner: {fb}. "
                         f"Revise the recommendation to avoid these contraindicated drugs. "
                         f"Suggest a safer alternative."})
        return {"final_answer": history[-1]["answer"], "ok": False,
                "attempts": len(history), "history": history, "context": ctx}


def main() -> None:
    ap = argparse.ArgumentParser(description="本体校验 + 自纠错安全问答")
    ap.add_argument("--q", required=True)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--inject", default=None, help="注入第1轮危险答案以演示拦截+纠错")
    args = ap.parse_args()
    pipe = SafeQAPipeline()
    print("病人病情(实体链接):", [c[0] for c in pipe.patient_conditions(args.q)] or "(无)")
    res = pipe.answer(args.q, args.retries, inject=args.inject)
    print("\n" + "=" * 60)
    for h in res["history"]:
        print(f"\n--- 第 {h['attempt']} 次生成 ---")
        print(h["answer"][:600])
        if h["violations"]:
            for v in h["violations"]:
                tag = "层级推理" if v.get("via_hierarchy") else "直接"
                print(f"  [本体拦截-{tag}] {v['axiom']} (来源 {v['source']})")
        else:
            print("  [本体校验] 通过, 无禁忌违规")
    print("\n" + "=" * 60)
    print(f"最终: {'安全通过' if res['ok'] else '仍有违规(已尽力纠错)'}, 共 {res['attempts']} 次")


if __name__ == "__main__":
    main()
