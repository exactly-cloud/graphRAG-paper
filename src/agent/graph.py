"""
模块5：LangGraph Agentic 编排（端到端系统）

把模块1-4 串成一个 Agentic 状态机：

  START → route(问题路由分类) → retrieve(混合检索, 按类型调权重)
        → generate(LLM 生成) → validate(本体禁忌校验)
        →[条件] 有违规且还有重试 → generate(带反馈重生成)  ↺
        →[条件] 通过/重试用尽 → END(输出带 provenance 的答案)

复用：HybridRetriever(模块3) + OntologyValidator(模块4) + EntityLinker(模块1)。

用法:
    python src/agent/graph.py --q "..."                 # 真实端到端
    python src/agent/graph.py --q "..." --seed "危险答案" # 演示拦截+纠错回环
"""
from __future__ import annotations

import argparse
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.llm import chat, chat_json
from src.reasoning.pipeline import SafeQAPipeline

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 按问题类型设检索权重 (k_vec, k_graph)
WEIGHTS = {"safety": (6, 14), "multihop": (6, 14), "narrative": (10, 6), "factual": (8, 10)}

_pipe = None


def pipe() -> SafeQAPipeline:
    global _pipe
    if _pipe is None:
        _pipe = SafeQAPipeline()
    return _pipe


class AgentState(TypedDict, total=False):
    question: str
    qtype: str
    k_vec: int
    k_graph: int
    context: list
    context_text: str
    conditions: list
    answer: str
    violations: list
    attempts: int
    max_retries: int
    history: list
    ok: bool
    seed_answer: str


def route_node(state: AgentState) -> dict:
    q = state["question"]
    try:
        out = chat_json(
            f"Classify the clinical question into one type.\nQuestion: {q}",
            system="You are a router. Output JSON only.",
            schema_hint='{"qtype":"safety|factual|multihop|narrative"}', temperature=0.0)
        qt = out.get("qtype", "factual") if isinstance(out, dict) else "factual"
    except Exception:
        qt = "factual"
    if qt not in WEIGHTS:
        qt = "factual"
    kv, kg = WEIGHTS[qt]
    return {"qtype": qt, "k_vec": kv, "k_graph": kg}


def retrieve_node(state: AgentState) -> dict:
    p = pipe()
    fused = p.retriever.retrieve(state["question"], k_vec=state["k_vec"],
                                 k_graph=state["k_graph"], k_final=10)
    conds = p.patient_conditions(state["question"])
    return {"context": fused, "context_text": p.retriever.format_context(fused), "conditions": conds}


def generate_node(state: AgentState) -> dict:
    attempts = state.get("attempts", 0)
    # 演示用：第1轮可注入危险答案
    if attempts == 0 and state.get("seed_answer"):
        ans = state["seed_answer"]
    else:
        sys_p = ("You are a clinical assistant for diabetes care. Answer concisely using the "
                 "evidence and recommend safe glucose-lowering therapy.")
        user = f"Evidence:\n{state['context_text']}\n\nQuestion: {state['question']}"
        viol = state.get("violations") or []
        if viol:
            fb = "; ".join(f"{v['drug']} is contraindicated in {v['condition']} "
                           f"(axiom: {v['axiom']}, source {v['source']})" for v in viol)
            user += (f"\n\nSAFETY VIOLATION found by the ontology reasoner in your previous answer: "
                     f"{fb}. Avoid these contraindicated drugs and give a safer alternative.")
        ans = chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                   temperature=0.3)
    hist = list(state.get("history", [])) + [{"attempt": attempts + 1, "answer": ans}]
    return {"answer": ans, "attempts": attempts + 1, "history": hist}


def validate_node(state: AgentState) -> dict:
    p = pipe()
    viol = p.validate_answer(state["question"], state["answer"])
    hist = list(state.get("history", []))
    if hist:
        hist[-1] = {**hist[-1], "violations": viol}
    return {"violations": viol, "ok": not viol, "history": hist}


def after_validate(state: AgentState) -> str:
    if state.get("ok"):
        return END
    if state.get("attempts", 0) <= state.get("max_retries", 2):
        return "generate"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", after_validate, {"generate": "generate", END: END})
    return g.compile()


_APP = None


def run(question: str, max_retries: int = 2, seed_answer: str | None = None) -> dict:
    global _APP
    if _APP is None:
        _APP = build_graph()
    init: AgentState = {"question": question, "attempts": 0, "max_retries": max_retries,
                        "history": [], "violations": []}
    if seed_answer:
        init["seed_answer"] = seed_answer
    return _APP.invoke(init, config={"recursion_limit": 50})


def main() -> None:
    ap = argparse.ArgumentParser(description="模块5 LangGraph Agentic 端到端问答")
    ap.add_argument("--q", required=True)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed", default=None, help="注入第1轮危险答案(演示拦截+纠错)")
    args = ap.parse_args()

    res = run(args.q, args.retries, seed_answer=args.seed)
    print(f"问题类型(路由): {res.get('qtype')}  | 检索权重 k_vec={res.get('k_vec')}, k_graph={res.get('k_graph')}")
    print(f"病人病情(实体链接): {[c[0] for c in res.get('conditions', [])] or '(无)'}")
    print("=" * 60)
    for h in res["history"]:
        print(f"\n--- 第 {h['attempt']} 次生成 ---")
        print(h["answer"][:550])
        v = h.get("violations") or []
        if v:
            for vi in v:
                tag = "层级推理" if vi.get("via_hierarchy") else "直接"
                print(f"  [本体拦截-{tag}] {vi['axiom']} (来源 {vi['source']})")
        else:
            print("  [本体校验] 通过, 无禁忌违规")
    print("\n" + "=" * 60)
    print(f"最终: {'安全通过' if res.get('ok') else '仍有违规(已尽力纠错)'}, 共 {res.get('attempts')} 次生成")


if __name__ == "__main__":
    main()
