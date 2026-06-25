"""
评测方法层：把 5 种方法统一成相同调用接口，用于消融对比。

  llm    : 纯 LLM（无检索）—— 下界基线
  vector : 向量 RAG（仅 PubMed 摘要）
  graph  : 图 RAG（仅知识图谱多跳事实）
  hybrid : 混合检索（向量+图 RRF 融合，无本体校验）
  full   : 本方法（混合检索 + 本体禁忌校验 + 自纠错回环）

三类题型接口：
  answer_mcqa(method, question, options) -> {"letter", "raw", "prov"}
  answer_yesno(method, question, context) -> {"label", "raw", "prov"}
  answer_open(method, question)          -> {"answer", "drugs", "violations", "attempts", "prov"}
"""
from __future__ import annotations

from src.eval.common import options_text, parse_letter, parse_yesno
from src.llm import chat, chat_json

METHODS = ["llm", "vector", "graph", "hybrid", "full"]

_vr = _gr = _hr = _pipe = None


def _retrievers():
    global _vr, _gr, _hr
    if _hr is None:
        from src.retrieval.hybrid import HybridRetriever
        _hr = HybridRetriever()
        _vr, _gr = _hr.vr, _hr.gr
    return _vr, _gr, _hr


def _pipeline():
    global _pipe
    if _pipe is None:
        from src.reasoning.pipeline import SafeQAPipeline
        _pipe = SafeQAPipeline()
    return _pipe


def build_context(method: str, query: str) -> tuple[str, list]:
    """按方法构造证据上下文，返回 (context_text, provenance_list)。"""
    if method == "llm":
        return "", []
    vr, gr, hr = _retrievers()
    if method == "vector":
        items = vr.search(query, 6)
        ctx = "\n".join(f"[V{i+1}] {it['title']}: {it.get('text','')[:280]}"
                        for i, it in enumerate(items))
        return ctx, [{"pmid": it["pmid"], "score": round(it["score"], 3)} for it in items]
    if method == "graph":
        items = gr.search(query, 12, max_hops=2)
        ctx = "\n".join(f"[G{i+1}] {it['fact']}" for i, it in enumerate(items))
        return ctx, [{"fact": it["fact"], "layer": it["layer"], "conf": it["confidence"]} for it in items]
    # hybrid / full 共用混合检索证据
    fused = hr.retrieve(query, k_final=10)
    return hr.format_context(fused), [{"src": f["source"], **f["prov"]} for f in fused]


def answer_mcqa(method: str, question: str, options: dict) -> dict:
    ctx, prov = build_context(method, question)
    sys_p = ("You are a medical exam assistant. Choose the single best option. "
             "Respond with the option letter only, e.g. 'Answer: B'.")
    ev = f"Evidence:\n{ctx}\n\n" if ctx else ""
    user = f"{ev}Question: {question}\nOptions:\n{options_text(options)}\n\nAnswer with the letter."
    raw = chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
               temperature=0.0, max_tokens=200)
    return {"letter": parse_letter(raw, options), "raw": raw, "prov": prov}


def answer_yesno(method: str, question: str, context: str = "") -> dict:
    ctx, prov = build_context(method, question)
    extra = f"\nResearch context: {context[:1200]}" if context else ""
    sys_p = ("You are a biomedical QA assistant. Answer the question with yes or no "
             "(use maybe only if truly uncertain). Start your answer with the label.")
    ev = f"Evidence:\n{ctx}\n" if ctx else ""
    user = f"{ev}{extra}\n\nQuestion: {question}\nAnswer (yes/no/maybe):"
    raw = chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
               temperature=0.0, max_tokens=200)
    return {"label": parse_yesno(raw), "raw": raw, "prov": prov}


def answer_open(method: str, question: str) -> dict:
    """开放推药题：用于禁忌违规率实验。full 走自纠错；其余单轮生成。"""
    if method == "full":
        p = _pipeline()
        res = p.answer(question, max_retries=2)
        ans = res["final_answer"]
        viol = res["history"][-1]["violations"]
        return {"answer": ans, "violations": viol, "attempts": res["attempts"],
                "prov": [{"src": f["source"], **f["prov"]} for f in res["context"]],
                "ok": res["ok"]}
    ctx, prov = build_context(method, question)
    sys_p = ("You are a clinical assistant for diabetes care. Recommend a specific "
             "glucose-lowering medication for the patient and give a brief rationale.")
    ev = f"Evidence:\n{ctx}\n\n" if ctx else ""
    ans = chat([{"role": "system", "content": sys_p},
                {"role": "user", "content": f"{ev}Question: {question}"}],
               temperature=0.3, max_tokens=400)
    return {"answer": ans, "violations": None, "attempts": 1, "prov": prov, "ok": None}
