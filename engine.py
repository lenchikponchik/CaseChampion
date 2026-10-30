"""Прямой вывод по правилам с коэффициентами уверенности (как в MYCIN).

Рабочая память — словарь: имя факта -> (значение, CF).
Предметных знаний здесь нет: болезни, пороги, д.в. и правила лежат в kb/*.json.
"""
import json
from pathlib import Path

import config

OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}


def load_kb(path):
    path = Path(path)
    diseases = json.loads((path / "diseases.json").read_text(encoding="utf-8"))
    products = json.loads((path / "products.json").read_text(encoding="utf-8"))
    rules = json.loads((path / "rules.json").read_text(encoding="utf-8"))
    return {**diseases, **rules, "products": products}


def compare(a, op, b):
    try:
        return OPS[op](a, b)
    except (TypeError, KeyError):
        return False


def check(cond, memory):
    # CF если условие выполнено, иначе None
    if "any" in cond:  # ИЛИ - берем максимальный CF
        cfs = [cf for cf in (check(c, memory) for c in cond["any"]) if cf is not None]
        return max(cfs) if cfs else None

    fact, op = cond["fact"], cond["op"]
    if op == "missing":
        return 1.0 if fact not in memory else None
    if fact not in memory:
        return None
    if op == "exists":
        return 1.0

    value, cf = memory[fact]
    if "ref" in cond:
        if cond["ref"] not in memory:
            return None
        target, ref_cf = memory[cond["ref"]]
        cf = min(cf, ref_cf)
    else:
        target = cond["value"]
    return cf if compare(value, op, target) else None


def set_fact(memory, name, value, cf):
    # повторный вывод усиливает CF
    if name in memory and memory[name][0] == value:
        old_cf = memory[name][1]
        cf = old_cf + cf * (1 - old_cf)
    memory[name] = (value, cf)


def fill(template, values):
    class SafeDict(dict):
        def __missing__(self, key):
            return "?"

    pretty = {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in values.items()}
    return template.format_map(SafeDict(pretty))


def fact_values(memory):
    return {name: value for name, (value, _) in memory.items()}


def apply(action, cf, memory, state, kb):
    if "set" in action:
        set_fact(memory, action["set"], action["value"], cf)
        return f"{action['set']} = {action['value']}"

    if "message" in action:
        text = fill(action["message"], fact_values(memory))
        state["messages"].append(text)
        return f"сообщение: {text}"

    if "load_products" in action:
        disease_id = memory.get("disease", (None, 0))[0]
        disease = next((d for d in kb["diseases"] if d["id"] == disease_id), None)
        ids = disease["products"] if disease else []
        state["products"] = [p for p in kb["products"] if p["id"] in ids]
        memory["products_count"] = (len(state["products"]), 1.0)
        return f"кандидаты д.в.: {len(state['products'])}"

    if "exclude_products" in action:
        rule = action["exclude_products"]
        if "ref" in rule:
            target = memory.get(rule["ref"], (None, 0))[0]
        else:
            target = rule["value"]
        kept = []
        for product in state["products"]:
            if compare(product.get(rule["field"]), rule["op"], target):
                reason = fill(rule["reason"], {**fact_values(memory), **product})
                state["excluded"].append({"product": product, "reason": reason})
            else:
                kept.append(product)
        removed = len(state["products"]) - len(kept)
        state["products"] = kept
        memory["products_count"] = (len(kept), 1.0)
        return f"исключено д.в.: {removed}"

    return "неизвестное действие"


def run(facts, kb):
    memory = dict(facts)
    state = {"products": [], "excluded": [], "messages": []}
    trace = []
    fired = set()

    changed = True
    while changed:
        changed = False
        for rule in kb["rules"]:
            if rule["id"] in fired:
                continue
            cfs = [check(cond, memory) for cond in rule["if"]]
            if any(cf is None for cf in cfs):
                continue
            cf = rule["cf"] * min(cfs, default=1.0)
            results = [apply(action, cf, memory, state, kb) for action in rule["then"]]
            fired.add(rule["id"])
            changed = True
            trace.append({
                "id": rule["id"],
                "group": rule["group"],
                "text": rule["text"],
                "cf": round(cf, 2),
                "result": "; ".join(results),
            })

    def value(name):
        return memory[name][0] if name in memory else None

    disease = next((d for d in kb["diseases"] if d["id"] == value("disease")), None)
    action = value("action")
    return {
        "status": value("status"),
        "status_cf": memory["status"][1] if "status" in memory else None,
        "disease": disease,
        "risk": value("risk"),
        "action": action,
        "action_label": kb["action_labels"].get(action, ""),
        "decision_cf": memory["action"][1] if "action" in memory else None,
        "products": state["products"],
        "excluded": state["excluded"],
        "messages": state["messages"],
        "trace": trace,
        "facts": memory,
    }


# начальные факты для вывода (нужны и в app.py, и в evaluate.py)

def find_disease(kb, disease_id):
    return next((d for d in kb["diseases"] if d["id"] == disease_id), None)


def label_ru(pred, kb):
    entry = pred["entry"]
    if entry is None:
        return f"{pred['label']} (нет в базе знаний)"
    if entry["kind"] == "invalid":
        return kb["kind_labels"]["invalid"].capitalize()
    crop = kb["crops"].get(entry["crop"], entry["crop"])
    if entry["kind"] == "healthy":
        return f"{crop} — {kb['kind_labels']['healthy']}"
    disease = find_disease(kb, entry.get("disease"))
    name = disease["name_ru"] if disease else entry.get("disease", "?")
    return f"{crop} — {name[0].lower() + name[1:]}"


def build_facts(field, preds, kb):
    facts = {name: (value, 1.0) for name, value in field.items()}
    facts.update({name: (value, 1.0) for name, value in config.CONFIDENCE.items()})
    facts["crop_ru"] = (kb["crops"][field["crop"]], 1.0)

    top = preds[0]
    facts["nn_label"] = (top["label"], 1.0)
    facts["nn_prob"] = (top["prob"], 1.0)
    if len(preds) > 1:
        second = preds[1]
        facts["nn_gap"] = (top["prob"] - second["prob"], 1.0)
        facts["nn_second_ru"] = (label_ru(second, kb), 1.0)
        facts["nn_second_prob"] = (second["prob"], 1.0)

    entry = top["entry"]
    if entry is None:
        return facts
    # CF этих фактов = вероятность нейросети
    facts["nn_kind"] = (entry["kind"], top["prob"])
    if "crop" in entry:
        facts["nn_crop"] = (entry["crop"], top["prob"])
        facts["nn_crop_ru"] = (kb["crops"].get(entry["crop"], entry["crop"]), 1.0)

    disease = find_disease(kb, entry.get("disease"))
    if entry["kind"] == "disease" and disease:
        facts["disease"] = (disease["id"], 1.0)
        facts["disease_ru"] = (disease["name_ru"], 1.0)
        for name in ["symptoms", "threshold_min", "threshold_max", "t_min", "t_max", "rh_min"]:
            facts[name] = (disease[name], 1.0)
    return facts
