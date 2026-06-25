"""
建图 step 2：Agentic 文献三元组抽取

对每篇 PubMed 摘要:
  1) 实体链接：先用本体词典找出文中已知的本体概念（提供给 LLM 做锚点）
  2) LLM 抽取：在 schema 允许的关系范围内抽取"文中明确陈述"的临床关系三元组
  3) 本体对齐：把抽出的 head/tail 表层文本对齐回本体概念 ID（对不齐则标为新文献实体）

输出候选三元组（未校验）: data/processed/graph/candidate_triples.jsonl
校验与合并在 step 3 (validate_merge.py)。

用法:
    python src/graph/extract_triples.py --limit 30          # 抽取前30篇(并发)
    python src/graph/extract_triples.py --limit 200 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.graph.schema import CORE_RELATIONS, relation_allowed
from src.llm import chat_json
from src.ontology.entity_linker import EntityLinker, _norm

PUBMED = Path("data/raw/pubmed/diabetes_pubmed.jsonl")
OUT = Path("data/processed/graph/candidate_triples.jsonl")

_REL_DOC = "\n".join(f"  - {k}: {v}" for k, v in CORE_RELATIONS.items())
_SCHEMA_HINT = ('[{"head":"...", "head_type":"Disease|Drug|Finding|LabTest|Substance|Procedure",'
                ' "relation":"<one of allowed>", "tail":"...", "tail_type":"...",'
                ' "evidence":"<=20-word quote"}]')


def map_concept(el: EntityLinker, text: str):
    """把表层文本对齐到本体概念 ID；对不齐返回 (None, text, None)。"""
    if not text:
        return None, text, None
    cid = el.term2cid.get(_norm(text))
    if cid:
        c = el.concepts[cid]
        return cid, c["preferred_name"], c["node_type"]
    links = el.link(text)
    if links:
        best = max(links, key=lambda x: x["end"] - x["start"])
        return best["concept_id"], best["preferred_name"], best["node_type"]
    return None, text, None


def extract_one(el: EntityLinker, rec: dict) -> list[dict]:
    pmid = rec.get("pmid", "")
    text = (rec.get("title", "") + ". " + rec.get("abstract", "")).strip()
    if len(text) < 40:
        return []
    ents = el.link(text)
    ent_hint = ", ".join(sorted({f"{e['preferred_name']} ({e['node_type']})" for e in ents})[:40])
    prompt = (
        f"Allowed relations (use ONLY these names):\n{_REL_DOC}\n\n"
        f"Recognized ontology entities in the text:\n{ent_hint or '(none)'}\n\n"
        f"Text:\n{text}\n\n"
        "Extract clinically-relevant relations that the text EXPLICITLY asserts, among medical "
        "concepts (diseases, drugs, findings, lab tests, substances). Prefer the recognized "
        "entities as head/tail when applicable. Do not infer beyond the text. "
        "If none, return []."
    )
    try:
        triples = chat_json(prompt, schema_hint=_SCHEMA_HINT, temperature=0.1, max_tokens=1200)
    except Exception as e:
        print(f"  [warn] PMID {pmid} 抽取失败: {e}")
        return []
    if not isinstance(triples, list):
        return []

    out = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        rel = str(t.get("relation", "")).strip()
        if rel not in CORE_RELATIONS:
            continue
        h_id, h_name, h_type = map_concept(el, str(t.get("head", "")).strip())
        t_id, t_name, t_type = map_concept(el, str(t.get("tail", "")).strip())
        if not t.get("head") or not t.get("tail"):
            continue
        out.append({
            "pmid": pmid, "relation": rel,
            "head_text": str(t.get("head", "")).strip(), "head_id": h_id,
            "head_name": h_name, "head_type": h_type or t.get("head_type"),
            "tail_text": str(t.get("tail", "")).strip(), "tail_id": t_id,
            "tail_name": t_name, "tail_type": t_type or t.get("tail_type"),
            "evidence": str(t.get("evidence", ""))[:200],
            "both_linked": bool(h_id and t_id),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic 文献三元组抽取")
    ap.add_argument("--limit", type=int, default=30, help="抽取摘要篇数")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 篇（增量抽取用）")
    ap.add_argument("--append", action="store_true", help="追加写入而非覆盖候选文件")
    ap.add_argument("--workers", type=int, default=8, help="并发数")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    el = EntityLinker()
    recs = []
    with PUBMED.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < args.offset:
                continue
            if len(recs) >= args.limit:
                break
            recs.append(json.loads(line))
    print(f"[抽取] 第 {args.offset+1}~{args.offset+len(recs)} 篇, 共 {len(recs)} 篇, 并发 {args.workers} ...")

    all_triples, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, el, r): r for r in recs}
        for fut in as_completed(futs):
            all_triples.extend(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"  进度 {done}/{len(recs)}, 累计三元组 {len(all_triples)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a" if args.append else "w", encoding="utf-8") as f:
        for t in all_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    linked = sum(1 for t in all_triples if t["both_linked"])
    from collections import Counter
    print(f"\n[完成] 候选三元组 {len(all_triples)} 条 -> {out_path}")
    print(f"  两端都对齐本体: {linked} ({linked/max(1,len(all_triples))*100:.0f}%)")
    print("  关系分布:", dict(Counter(t["relation"] for t in all_triples).most_common()))


if __name__ == "__main__":
    main()
