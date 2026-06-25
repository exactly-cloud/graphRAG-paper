"""
PubMed 糖尿病文献抓取脚本 (基于 NCBI E-utilities)

用途: 为 Agentic GraphRAG 的离线建图阶段抓取糖尿病相关 PubMed 摘要语料。
输出: data/raw/pubmed/diabetes_pubmed.jsonl  (每行一篇: pmid/title/abstract/journal/year/mesh)

用法示例:
    python src/data_acquisition/fetch_pubmed.py --max 2000
    python src/data_acquisition/fetch_pubmed.py --query "gestational diabetes" --max 1000 --email you@example.com

说明:
- NCBI 无 API key 时限速 3 请求/秒; 提供 NCBI_API_KEY 后可到 10 请求/秒 (强烈建议申请, 免费)。
- 通过环境变量或 .env 配置 NCBI_EMAIL / NCBI_API_KEY。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from Bio import Entrez
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 默认检索式: 糖尿病主干 + 主要并发症/用药, 限定近年、英文、有摘要
DEFAULT_QUERY = (
    '("diabetes mellitus"[MeSH Terms] OR "diabetes mellitus, type 2"[MeSH Terms] '
    'OR "diabetes mellitus, type 1"[MeSH Terms] OR "diabetes, gestational"[MeSH Terms] '
    'OR "diabetic nephropathies"[MeSH Terms] OR "diabetic retinopathy"[MeSH Terms] '
    'OR "diabetic neuropathies"[MeSH Terms]) '
    'AND ("2015"[Date - Publication] : "3000"[Date - Publication]) '
    'AND English[Language] AND hasabstract'
)

OUT_DIR = Path("data/raw/pubmed")


def setup_entrez(email: str | None, api_key: str | None) -> None:
    Entrez.email = email or os.getenv("NCBI_EMAIL") or "anonymous@example.com"
    key = api_key or os.getenv("NCBI_API_KEY")
    if key:
        Entrez.api_key = key
    Entrez.tool = "diabetes-graphrag"


def search_pmids(query: str, max_results: int, sort: str = "relevance") -> list[str]:
    """用 esearch 配合 history server 收集 PMID 列表。

    sort: relevance(按相关性, 拿核心代表文献) / pub_date(按发表日期, 拿最新)
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0,
                            usehistory="y", sort=sort)
    res = Entrez.read(handle)
    handle.close()
    total = int(res["Count"])
    n = min(total, max_results)
    print(f"[esearch] 命中 {total} 篇, 计划抓取 {n} 篇 (排序: {sort})")

    pmids: list[str] = []
    batch = 5000
    for start in range(0, n, batch):
        h = Entrez.esearch(
            db="pubmed", term=query, retstart=start,
            retmax=min(batch, n - start), sort=sort,
            webenv=res["WebEnv"], query_key=res["QueryKey"], usehistory="y",
        )
        r = Entrez.read(h)
        h.close()
        pmids.extend(r["IdList"])
        time.sleep(0.34)
    return pmids


def _parse_article(art: dict) -> dict | None:
    try:
        medline = art["MedlineCitation"]
        pmid = str(medline["PMID"])
        article = medline["Article"]
        title = article.get("ArticleTitle", "")

        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(p) for p in abstract_parts)
        else:
            abstract = str(abstract_parts)
        if not abstract.strip():
            return None

        journal = article.get("Journal", {}).get("Title", "")
        year = ""
        try:
            year = article["Journal"]["JournalIssue"]["PubDate"].get("Year", "")
        except Exception:
            pass

        mesh = []
        for mh in medline.get("MeshHeadingList", []):
            mesh.append(str(mh["DescriptorName"]))

        return {
            "pmid": pmid,
            "title": str(title),
            "abstract": abstract,
            "journal": str(journal),
            "year": str(year),
            "mesh": mesh,
        }
    except Exception:
        return None


def fetch_abstracts(pmids: list[str], out_path: Path, batch: int = 200) -> int:
    """用 efetch 分批抓取摘要并写入 JSONL。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(pmids), batch), desc="efetch"):
            chunk = pmids[start:start + batch]
            for attempt in range(3):
                try:
                    h = Entrez.efetch(db="pubmed", id=",".join(chunk),
                                      rettype="medline", retmode="xml")
                    records = Entrez.read(h)
                    h.close()
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  [warn] 批次 {start} 失败: {e}")
                        records = {"PubmedArticle": []}
                    time.sleep(2 ** attempt)
            for art in records.get("PubmedArticle", []):
                rec = _parse_article(art)
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
            time.sleep(0.12 if Entrez.api_key else 0.34)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="抓取糖尿病 PubMed 摘要语料")
    ap.add_argument("--query", default=DEFAULT_QUERY, help="PubMed 检索式")
    ap.add_argument("--max", type=int, default=2000, help="最多抓取篇数")
    ap.add_argument("--sort", default="relevance",
                    choices=["relevance", "pub_date"],
                    help="relevance=核心代表文献(推荐), pub_date=最新")
    ap.add_argument("--out", default=str(OUT_DIR / "diabetes_pubmed.jsonl"))
    ap.add_argument("--email", default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    setup_entrez(args.email, args.api_key)
    pmids = search_pmids(args.query, args.max, sort=args.sort)
    out_path = Path(args.out)
    n = fetch_abstracts(pmids, out_path)
    print(f"[done] 已写入 {n} 篇到 {out_path}")


if __name__ == "__main__":
    main()
