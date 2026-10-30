"""Оценка нейросети и гибрида на тестовых снимках.

python evaluate.py [модель ...]  - по умолчанию модель из config.py
список снимков в results/testset.csv, сами файлы в data/
результат: results/<модель>/predictions.csv, confusion.csv, metrics.json
"""
import json
import sys
import time

import pandas as pd
from PIL import Image

import classifier
import config
import engine

# ответы "перепроверь/пересними" вместо обработки
CHECK_ACTIONS = {"recheck", "reshoot"}


def class_key(entry):
    if entry is None:
        return "нет в БЗ"
    if entry["kind"] == "disease":
        return entry["disease"]
    return f"{entry['crop']}_healthy" if entry["kind"] == "healthy" else "invalid"


def run_model(model_id, test, kb):
    classifier.load_model(model_id)  # прогрев, чтобы загрузка не попала в замер
    rows = []
    for item in test.itertuples():
        image = Image.open(config.DATA_DIR / item.path).convert("RGB")
        start = time.perf_counter()
        preds = classifier.predict(image, kb["labels"], model_id)
        elapsed = time.perf_counter() - start
        true_entry = kb["labels"][classifier.normalize_label(item.true_class)]
        # условия сценария 1, культура - настоящая (для не-листа берем пшеницу)
        field = {**config.FIELD_DEFAULTS, "crop": true_entry.get("crop", "wheat")}
        result = engine.run(engine.build_facts(field, preds, kb), kb)
        rows.append({
            "set": item.set, "path": item.path, "true": class_key(true_entry),
            "pred": class_key(preds[0]["entry"]), "pred_label": preds[0]["label"],
            "prob": round(preds[0]["prob"], 4), "time_s": round(elapsed, 4),
            "status": result["status"], "action": result["action"],
        })
    return pd.DataFrame(rows)


def hybrid_stats(df):
    # ошибки "не та болезнь" и сколько из них поймали правила
    wrong = df[(df.pred != df.true) & ~df.pred.str.endswith("healthy") & ~df.pred.isin(["invalid", "нет в БЗ"])]
    caught = wrong.action.isin(CHECK_ACTIONS)
    correct = df[df.pred == df.true]
    return {
        "n": len(df),
        "nn_wrong_disease": len(wrong),
        "nn_wrong_disease_share": round(len(wrong) / len(df), 4),
        "hybrid_caught": int(caught.sum()),
        "hybrid_caught_share_of_errors": round(caught.mean(), 4) if len(wrong) else None,
        "hybrid_passed": int((~caught).sum()),
        "hybrid_wrong_treatment_share": round((~caught).sum() / len(df), 4),
        # лишние проверки: нейросеть права, а система все равно просит перепроверить
        "false_alarms": int(correct.action.isin(CHECK_ACTIONS).sum()),
        "false_alarm_share_of_correct": round(correct.action.isin(CHECK_ACTIONS).mean(), 4) if len(correct) else None,
    }


def macro_f1(true, pred):
    scores = []
    for label in true.unique():
        tp = ((pred == label) & (true == label)).sum()
        precision = tp / (pred == label).sum() if (pred == label).any() else 0.0
        recall = tp / (true == label).sum()
        scores.append(2 * precision * recall / (precision + recall) if tp else 0.0)
    return sum(scores) / len(scores)


def summarize(df):
    metrics = {"mean_time_s": round(df.time_s.mean(), 4)}
    for name, part in df[df.set != "not_leaf"].groupby("set"):
        metrics[name] = {
            "n": len(part),
            "accuracy": round((part.pred == part.true).mean(), 4),
            "macro_f1": round(macro_f1(part.true, part.pred), 4),
            "per_class_accuracy": (part.pred == part.true).groupby(part.true).mean().round(4).to_dict(),
            "hybrid": hybrid_stats(part),
        }
    leaves = df[df.set == "not_leaf"]
    if len(leaves):
        metrics["not_leaf"] = {
            "n": len(leaves),
            "filtered": int(leaves.action.isin(CHECK_ACTIONS).sum()),
            "by_status": leaves.status.fillna("—").value_counts().to_dict(),
            "nn_said_invalid": int((leaves.pred == "invalid").sum()),
        }
    return metrics


if __name__ == "__main__":
    kb = engine.load_kb(config.KB_DIR)
    test = pd.read_csv(config.RESULTS_DIR / "testset.csv")
    for model_id in sys.argv[1:] or [config.MODEL_ID]:
        df = run_model(model_id, test, kb)
        out = config.RESULTS_DIR / model_id.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "predictions.csv", index=False)
        main = df[df.set == "test"]
        pd.crosstab(main.true, main.pred).to_csv(out / "confusion.csv")
        metrics = {"model": model_id, **summarize(df)}
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=1))
