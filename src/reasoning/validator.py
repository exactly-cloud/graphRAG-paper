"""
模块4 step 2：本体禁忌校验器（OWL + HermiT DL 推理）

加载 diabetes.owl，用 HermiT 分类（计算 is_a 闭包），然后对"给某病情开某药"的
论断做禁忌校验：判断 病情类 ⊑ CI_<drug>（药物禁忌条件类）是否成立。
由于禁忌病种 ⊑ CI_<drug> 且具体病种 ⊑ 禁忌病种（经桥接/层级），HermiT 能推出
更具体的病种（如"4期糖尿病肾病"）也违反禁忌 → 这是 DL 层级推理的价值。

每个违规给出 provenance：违反的公理 + 触发的禁忌病种 + 来源(MED-RT)。

用法（库）:
    from src.reasoning.validator import OntologyValidator
    v = OntologyValidator()                # 首次会跑 HermiT 分类
    v.check_pair(drug_id, condition_id)
    v.check_texts("glyburide", "gestational diabetes")
用法（命令行自测）:
    python src/reasoning/validator.py
"""
from __future__ import annotations

import json
from pathlib import Path

from owlready2 import get_ontology, sync_reasoner

from src.ontology.entity_linker import EntityLinker

OUT = Path("data/processed/reasoning")


class OntologyValidator:
    def __init__(self, run_reasoner: bool = True):
        self.index = json.loads((OUT / "owl_index.json").read_text(encoding="utf-8"))
        self.onto = get_ontology(str(OUT / "diabetes.owl")).load()
        self.consistent = True
        if run_reasoner:
            try:
                with self.onto:
                    sync_reasoner(self.onto, debug=0)
            except Exception as e:
                self.consistent = False
                print(f"[reasoner] 推理告警: {e}")
        self.cls_of = self.index["cls_of"]
        self.name_of = self.index["name_of"]
        self.ci_class_of = self.index["ci_class_of"]
        self.drug_ci = self.index["drug_ci"]
        self._el = None

    @property
    def el(self):
        if self._el is None:
            self._el = EntityLinker()
        return self._el

    def _cls(self, name):
        return self.onto[name] if name else None

    def check_pair(self, drug_id: str, condition_id: str) -> dict:
        """核心：判断给 condition_id 病人开 drug_id 是否违反禁忌（含层级推理）。"""
        res = {"drug_id": drug_id, "condition_id": condition_id, "violation": False}
        ci_name = self.ci_class_of.get(drug_id)
        cond_cls = self._cls(self.cls_of.get(condition_id))
        if not ci_name or cond_cls is None:
            return res
        ci_cls = self._cls(ci_name)
        if ci_cls is None:
            return res
        # DL 推理后的祖先闭包里是否含 CI_<drug>
        if ci_cls in cond_cls.ancestors():
            res["violation"] = True
            # 找触发的禁忌病种（病情的祖先中属于该药直接禁忌集的那个）
            anc_ids = {cid for cid, cn in self.cls_of.items()
                       if self._cls(cn) in cond_cls.ancestors()}
            trig = [c for c in self.drug_ci.get(drug_id, []) if c in anc_ids]
            trig_id = trig[0] if trig else (self.drug_ci.get(drug_id, [None])[0])
            res.update({
                "drug": self.name_of.get(drug_id, drug_id),
                "condition": self.name_of.get(condition_id, condition_id),
                "trigger": self.name_of.get(trig_id, trig_id),
                "trigger_id": trig_id,
                "axiom": f"contraindicated_with({self.name_of.get(drug_id)}, "
                         f"{self.name_of.get(trig_id)})",
                "via_hierarchy": trig_id != condition_id,
                "source": "MED-RT",
            })
        return res

    def _link_id(self, text: str, want_drug: bool):
        for e in self.el.link(text):
            is_drug = e["node_type"] == "Drug"
            if e["concept_id"] in self.cls_of and is_drug == want_drug:
                return e["concept_id"]
        return None

    def check_texts(self, drug_text: str, condition_text: str) -> dict:
        d = self._link_id(drug_text, True)
        c = self._link_id(condition_text, False)
        if not d or not c:
            return {"violation": False, "note": f"实体未链接(drug={d}, cond={c})"}
        return self.check_pair(d, c)


def _demo():
    v = OntologyValidator()
    print(f"本体一致性: {'一致' if v.consistent else '不一致'}\n")
    cases = [
        ("glyburide", "gestational diabetes mellitus"),
        ("chlorpropamide", "pregnancy"),
        ("canagliflozin", "chronic kidney disease stage 4 due to type 2 diabetes mellitus"),
        ("metformin", "ketoacidosis due to type 2 diabetes mellitus"),
        ("metformin", "type 2 diabetes mellitus"),  # 安全对照
    ]
    for drug, cond in cases:
        r = v.check_texts(drug, cond)
        if r.get("violation"):
            tag = "经层级推理" if r.get("via_hierarchy") else "直接匹配"
            print(f"[违规-{tag}] {drug} + {cond}")
            print(f"     公理: {r['axiom']}  (触发禁忌病种: {r['trigger']}, 来源 {r['source']})")
        else:
            print(f"[安全/未命中] {drug} + {cond}  {r.get('note','')}")


if __name__ == "__main__":
    _demo()
