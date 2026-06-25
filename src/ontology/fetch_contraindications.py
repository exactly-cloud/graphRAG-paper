"""
抓取降糖药的"用药禁忌"关系 (contraindicated_with)

背景: RxNorm 本地文件不含禁忌关系; 权威来源是 MED-RT, 经 NLM 的 RxClass API 获取。
对 diabetes_drugs.csv 中每个降糖药成分, 查询其 MED-RT ci_with(禁忌)疾病/状态。

输出 (data/processed/ontology/):
  - drug_contraindications.csv: rxcui, ingredient, ci_disease, ci_code, source

这是本体"安全校验层"(模块4)与自建"安全禁忌型"QA 集的数据基础。

用法:
    python src/ontology/fetch_contraindications.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import requests

OUT_DIR = Path("data/processed/ontology")
DRUGS_CSV = OUT_DIR / "diabetes_drugs.csv"
API = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"


def fetch_ci(rxcui: str) -> list[dict]:
    """返回某药的 ci_with 禁忌类(疾病)列表。"""
    try:
        r = requests.get(API, params={"rxcui": rxcui, "relaSource": "MEDRT", "rela": "ci_with"}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [warn] rxcui={rxcui} 失败: {e}")
        return []
    out = []
    for it in data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
        if it.get("rela") != "ci_with":      # 仅保留禁忌关系
            continue
        cls = it.get("rxclassMinConceptItem", {})
        # 仅保留疾病类(MeSH D-编码), 排除机制/生理类(N-编码)
        cid = cls.get("classId", "")
        if not cid.startswith("D"):
            continue
        out.append({"ci_disease": cls.get("className", ""), "ci_code": cid,
                    "source": cls.get("classType", "MEDRT")})
    return out


def main() -> None:
    if not DRUGS_CSV.exists():
        raise SystemExit(f"缺少 {DRUGS_CSV}, 请先运行 extract_rxnorm_drugs.py")
    drugs = list(csv.DictReader(DRUGS_CSV.open(encoding="utf-8")))
    print(f"对 {len(drugs)} 个降糖药查询 MED-RT 禁忌 ...")

    rows, seen = [], set()
    for i, d in enumerate(drugs, 1):
        rxcui, ing = d["rxcui"], d["ingredient"]
        for ci in fetch_ci(rxcui):
            key = (rxcui, ci["ci_code"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"rxcui": rxcui, "ingredient": ing, **ci})
        if i % 20 == 0:
            print(f"  进度 {i}/{len(drugs)}")
        time.sleep(0.12)

    with (OUT_DIR / "drug_contraindications.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rxcui", "ingredient", "ci_disease", "ci_code", "source"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    n_drugs = len({r["rxcui"] for r in rows})
    top = Counter(r["ci_disease"] for r in rows).most_common(15)
    print(f"\n[完成] {len(rows)} 条禁忌关系 (覆盖 {n_drugs} 个药) -> {OUT_DIR/'drug_contraindications.csv'}")
    print("  最常见禁忌(前15):")
    for dis, n in top:
        print(f"    {dis:<40} {n}")


if __name__ == "__main__":
    main()
