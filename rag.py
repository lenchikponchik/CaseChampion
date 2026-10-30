"""Поиск по документам (RAG без генерации): находит фрагменты официальных материалов к решению.

Документы лежат в kb/docs/*.txt: первая строка - заголовок, вторая - ссылка, третья - дата, дальше абзацы.
Абзацы режутся на фрагменты, фрагменты переводятся в векторы моделью rubert-tiny2,
поиск - по косинусной близости с запросом плюс небольшая добавка за совпавшие слова
(у маленькой модели «ржавчина кукурузы» и «ржавчина пшеницы» иначе слишком похожи). Языковая модель не используется: показываем сами цитаты.
"""
import re

import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

import config


def load_chunks():
    chunks = []
    for path in sorted(config.DOCS_DIR.glob("*.txt")):
        title, url, date, *body = path.read_text(encoding="utf-8").split("\n")
        text = ""
        for para in filter(None, (p.strip() for p in body)):
            # короткие абзацы (подзаголовки) приклеиваем к следующим
            if len(text) > config.RAG_CHUNK_CHARS // 3 and len(text) + len(para) > config.RAG_CHUNK_CHARS:
                chunks.append({"title": title, "url": url, "date": date, "text": text})
                text = ""
            text = f"{text} {para}".strip()
        if text:
            chunks.append({"title": title, "url": url, "date": date, "text": text})
    return chunks


def embed(texts, tokenizer, model):
    vectors = []
    for start in range(0, len(texts), 32):
        batch = tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            cls = model(**batch).last_hidden_state[:, 0]
        vectors.append(torch.nn.functional.normalize(cls, dim=-1))
    return torch.cat(vectors)


@st.cache_resource(show_spinner="Индексация документов…")
def load_index():
    tokenizer = AutoTokenizer.from_pretrained(config.RAG_MODEL_ID)
    model = AutoModel.from_pretrained(config.RAG_MODEL_ID).eval()
    chunks = load_chunks()
    return chunks, embed([f"{c['title']}. {c['text']}" for c in chunks], tokenizer, model), tokenizer, model


def stems(text):
    # грубая основа слова: первые 5 букв, этого хватает для падежей
    return {w[:5] for w in re.findall(r"[а-яёa-z]{4,}", text.lower())}


def search(query, k=config.RAG_TOP_K):
    chunks, matrix, tokenizer, model = load_index()
    words = stems(query)
    overlap = torch.tensor([len(words & stems(c["title"] + " " + c["text"])) / max(len(words), 1) for c in chunks])
    scores = matrix @ embed([query], tokenizer, model)[0] + config.RAG_WORD_WEIGHT * overlap
    top = torch.topk(scores, k=min(k, len(chunks)))
    return [{**chunks[i], "score": s} for s, i in zip(top.values.tolist(), top.indices.tolist())]


if __name__ == "__main__":
    import sys

    for hit in search(" ".join(sys.argv[1:]) or "бурая ржавчина пшеницы порог вредоносности"):
        print(f"{hit['score']:.2f}  {hit['title']}\n      {hit['text'][:200]}…\n")
