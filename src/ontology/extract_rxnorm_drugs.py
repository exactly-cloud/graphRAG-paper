"""
从 RxNorm 抽取降糖药清单 (对应模块1: 补充本体的 Drug 节点)

入口: ATC 药物分类 A10*（"DRUGS USED IN DIABETES"），权威覆盖全部降糖药。
  - A10A  胰岛素及类似物
  - A10B  口服降糖药（双胍/磺脲/格列奈/TZD/DPP-4/SGLT2/GLP-1/α-糖苷酶抑制剂...）
  - A10X  其他

ATC code 层级:
  3位 A10 (顶类) / 4位 A10A (大类) / 5位 A10BA (药物类别) / 7位 A10BA02 (具体成分)

输出 (data/processed/ontology/):
  - diabetes_drugs.csv: rxcui, ingredient, atc_code, atc_class, category, brand_names

用法:
    python src/ontology/extract_rxnorm_drugs.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

RRF = Path("data/raw/rxnorm/rrf")
OUT_DIR = Path("data/processed/ontology")

# RXNCONSO 列索引
RXCUI, LAT, TS, LUI, STT, SUI, ISPREF, RXAUI, SAUI, SCUI, SDUI, SAB, TTY, CODE, STR, SRL, SUPPRESS, CVF = range(18)


def parse_conso():
    atc_name: dict[str, str] = {}          # ATC code -> 类别名 (4/5位)
    anti_diabetic: dict[str, str] = {}      # ingredient rxcui -> atc_code (7位, A10*)
    rxnorm_in: dict[str, str] = {}          # rxcui -> RxNorm 成分首选名 (IN/PIN)
    brand_name: dict[str, str] = {}         # rxcui -> 商品名 (BN)

    with (RRF / "RXNCONSO.RRF").open(encoding="utf-8") as f:
        for line in f:
            c = line.split("|")
            sab, tty, code, name = c[SAB], c[TTY], c[CODE], c[STR]
            if sab == "ATC" and code.startswith("A10"):
                if len(code) in (4, 5) and code not in atc_name and "in ATC" not in name:
                    atc_name[code] = name
                if len(code) == 7:  # 具体成分
                    anti_diabetic[c[RXCUI]] = code
            elif sab == "RXNORM":
                if tty in ("IN", "PIN") and c[RXCUI] not in rxnorm_in:
                    rxnorm_in[c[RXCUI]] = name
                elif tty == "BN":
                    brand_name[c[RXCUI]] = name
    return atc_name, anti_diabetic, rxnorm_in, brand_name


def parse_tradenames(ingredient_cuis: set[str]) -> dict[str, set[str]]:
    """ingredient rxcui -> {brand rxcui}，基于 has_tradename/tradename_of。"""
    ing2brand: dict[str, set[str]] = defaultdict(set)
    # RXNREL: 0RXCUI1 1RXAUI1 2STYPE1 3REL 4RXCUI2 5RXAUI2 6STYPE2 7RELA 8RUI...
    with (RRF / "RXNREL.RRF").open(encoding="utf-8") as f:
        for line in f:
            c = line.split("|")
            rela, cui1, cui2 = c[7], c[0], c[4]
            if rela in ("has_tradename", "tradename_of"):
                # 取关系中"非成分"的一侧作为商品名概念
                if cui1 in ingredient_cuis:
                    ing2brand[cui1].add(cui2)
                if cui2 in ingredient_cuis:
                    ing2brand[cui2].add(cui1)
    return ing2brand


def category(atc_code: str) -> str:
    if atc_code.startswith("A10A"):
        return "insulin"
    if atc_code.startswith("A10B"):
        return "oral/non-insulin"
    return "other"


def main() -> None:
    print("[1/3] 解析 RXNCONSO ...")
    atc_name, anti_diabetic, rxnorm_in, brand_name = parse_conso()
    print(f"  降糖药成分(ATC A10 7位): {len(anti_diabetic)}")
    print(f"  ATC 类别名: {len(atc_name)}  RxNorm成分名: {len(rxnorm_in)}  商品名: {len(brand_name)}")

    print("[2/3] 解析商品名关系 (RXNREL) ...")
    ing2brand = parse_tradenames(set(anti_diabetic.keys()))

    print("[3/3] 写出 diabetes_drugs.csv ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rxcui, atc_code in sorted(anti_diabetic.items(), key=lambda x: x[1]):
        name = rxnorm_in.get(rxcui, "")
        cls = atc_name.get(atc_code[:5], atc_name.get(atc_code[:4], ""))
        brands = sorted({brand_name[b] for b in ing2brand.get(rxcui, set()) if b in brand_name})
        rows.append({
            "rxcui": rxcui,
            "ingredient": name,
            "atc_code": atc_code,
            "atc_class": cls,
            "category": category(atc_code),
            "brand_names": "; ".join(brands[:10]),
        })

    with (OUT_DIR / "diabetes_drugs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rxcui", "ingredient", "atc_code", "atc_class", "category", "brand_names"])
        w.writeheader()
        w.writerows(rows)

    # 统计
    from collections import Counter
    cls_cnt = Counter(r["atc_class"] for r in rows if r["atc_class"])
    cat_cnt = Counter(r["category"] for r in rows)
    print(f"\n[完成] {len(rows)} 个降糖药成分 -> {OUT_DIR/'diabetes_drugs.csv'}")
    print("  大类分布:", dict(cat_cnt))
    print("  药物类别(前12):")
    for cls, n in cls_cnt.most_common(12):
        print(f"    {cls:<55} {n}")


if __name__ == "__main__":
    main()
