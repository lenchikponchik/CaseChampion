"""Нейросеть: загрузка модели и top-3 по фото листа.

python classifier.py [фото ...] - печатает id2label, метки без записи в БЗ
и top-3 для фото (по умолчанию все из examples/)
"""
import re
import sys
import time

import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config


def normalize_label(label):
    label = label.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", label).strip()


@st.cache_resource(show_spinner="Загрузка нейросети…")
def load_model(model_id=config.MODEL_ID):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    print(f"Модель {model_id}, id2label:")
    for idx, label in sorted(model.config.id2label.items()):
        print(f"  {idx}: {label}")
    return processor, model


def predict(image, labels, model_id=config.MODEL_ID):
    processor, model = load_model(model_id)
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, k=min(config.TOP_K, probs.shape[0]))

    results = []
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        label = model.config.id2label[idx]
        results.append({
            "label": label,
            "prob": prob,
            "entry": labels.get(normalize_label(label)),
        })
    return results


if __name__ == "__main__":
    import engine

    kb = engine.load_kb(config.KB_DIR)
    _, model = load_model()

    missing = [l for l in model.config.id2label.values() if normalize_label(l) not in kb["labels"]]
    if missing:
        print("\nНет записи в kb/diseases.json -> labels для меток:", ", ".join(missing))
    else:
        print("\nВсе метки модели есть в базе знаний.")

    paths = sys.argv[1:] or sorted(
        p for p in config.EXAMPLES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for path in paths:
        start = time.perf_counter()
        preds = predict(Image.open(path), kb["labels"])
        elapsed = time.perf_counter() - start
        top = ", ".join(f"{p['label']} {p['prob']:.2f}" for p in preds)
        print(f"{path}: {top}  [{elapsed:.2f} с]")
