"""Интерфейс на Streamlit. Запуск: streamlit run app.py"""
import csv
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

import classifier
import config
import engine
import rag
import weather

st.set_page_config(page_title="Защита посевов: гибридная СППР", page_icon="🌾", layout="wide")

ACTION_STYLE = {
    "treat": st.error, "treat_postponed": st.error, "consult": st.error,
    "monitor_3d": st.warning, "recheck": st.warning, "reshoot": st.warning,
    "none": st.success, "monitor_7d": st.success,
}
FRAC_OPTIONS = ["нет", "3", "11", "4", "27", "M01", "M03"]
HISTORY_COLUMNS = ["время", "культура", "метка модели", "вероятность", "решение", "CF"]


def save_history(field, preds, result):
    is_new = not config.HISTORY_PATH.exists()
    with open(config.HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(HISTORY_COLUMNS)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            field["crop"],
            preds[0]["label"],
            round(preds[0]["prob"], 3),
            result["action"] or "",
            round(result["decision_cf"], 2) if result["decision_cf"] is not None else "",
        ])


def fill_weather():
    try:
        data = weather.get_weather(st.session_state.lat, st.session_state.lon)
    except Exception as error:
        st.session_state.weather_error = str(error)
        return
    data["temp"] = float(data["temp"])
    data["wind"] = float(data["wind"])
    st.session_state.update(data)
    st.session_state.weather_info = data


def sidebar(kb):
    for name, value in config.FIELD_DEFAULTS.items():
        st.session_state.setdefault(name, value)

    with st.sidebar:
        st.header("Условия на поле")
        crops = kb["crops"]
        crop = st.selectbox("Культура", list(crops), format_func=crops.get, key="crop")
        st.slider("Поражённых растений, %", 0, 100, key="pct")
        st.number_input("Температура воздуха, °C", -30.0, 50.0, step=0.5, format="%.1f", key="temp")
        st.slider("Относительная влажность, %", 0, 100, key="humidity")
        st.checkbox("Осадки за последние 3 суток", key="rain_3d")
        st.checkbox("Дождь в ближайшие 6 ч", key="rain_6h")
        st.number_input("Ветер, м/с", 0.0, 40.0, step=0.5, format="%.1f", key="wind")
        st.number_input("Дней до уборки", 0, 365, key="days_to_harvest")
        st.selectbox("Последняя обработка, FRAC-группа", FRAC_OPTIONS, key="last_frac")

        with st.expander("Погода по координатам"):
            st.number_input("Широта", -90.0, 90.0, format="%.2f", key="lat")
            st.number_input("Долгота", -180.0, 180.0, format="%.2f", key="lon")
            st.button("Подставить погоду", on_click=fill_weather)
            if "weather_error" in st.session_state:
                st.warning(f"Погода недоступна, введите вручную. {st.session_state.pop('weather_error')}")
            if "weather_info" in st.session_state:
                st.caption("Погода подставлена из Open-Meteo")

    names = ["pct", "temp", "humidity", "rain_3d", "rain_6h", "wind", "days_to_harvest", "last_frac"]
    return {"crop": crop, **{name: st.session_state[name] for name in names}}


def pick_image():
    source = st.radio("Источник фото", ["Пример из examples/", "Загрузить своё фото"], horizontal=True)
    if source == "Загрузить своё фото":
        file = st.file_uploader("Фото листа", type=["jpg", "jpeg", "png"])
        return Image.open(file) if file else None

    examples = sorted(
        p for p in config.EXAMPLES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not examples:
        st.info("В папке examples/ пока нет фото — список нужных файлов в examples/README.txt")
        return None
    path = st.selectbox("Пример", examples, format_func=lambda p: p.name)
    return Image.open(path)


def show_network(preds, kb, elapsed):
    st.subheader("Нейросеть")
    for pred in preds:
        st.progress(pred["prob"], text=f"{engine.label_ru(pred, kb)}: {pred['prob'] * 100:.1f} %")
    for pred in preds:
        if pred["entry"] is None:
            st.warning(f"Метки модели «{pred['label']}» нет в kb/diseases.json → labels")
    st.caption(f"Модель: {config.MODEL_ID} · распознавание {elapsed:.2f} с на CPU")


def show_decision(result, kb):
    st.subheader("Решение")
    if result["action"]:
        show = ACTION_STYLE.get(result["action"], st.info)
        show(f"**{result['action_label']}**  \nуверенность решения {result['decision_cf']:.2f}")
    else:
        st.info("Решение не сформировано: ни одно правило не определило действие")

    if result["status"]:
        status = kb["status_labels"].get(result["status"], result["status"])
        st.markdown(f"**Статус:** {status} (CF {result['status_cf']:.2f})")
    disease = result["disease"]
    if disease and result["status"] in ("confirmed", "probable"):
        st.markdown(f"**Диагноз:** {disease['name_ru']} — *{disease['pathogen']}* ({disease['group']})")
        st.markdown(f"**Признаки:** {disease['symptoms']}")
        if result["risk"]:
            risk_cf = result["facts"]["risk"][1]
            st.markdown(f"**Риск:** {kb['risk_labels'].get(result['risk'], result['risk'])} (CF {risk_cf:.2f})")
        demo = " · *условные значения*" if disease["demo"] else ""
        low, high = disease["threshold_min"], disease["threshold_max"]
        threshold = f"{low} %" if low == high else f"{low}–{high} %"
        st.markdown(f"**ЭПВ:** {threshold} поражённых растений — {disease['threshold_note']}{demo}")
        if disease["note"]:
            st.markdown(f"**Примечание:** {disease['note']}")


def show_products(result):
    if result["products"]:
        st.subheader("Рекомендованные действующие вещества")
        table = pd.DataFrame([{
            "Д.в.": p["name_ru"], "Класс": p["chem_class"], "FRAC": p["frac"], "Действие": p["mode"],
            "Срок ожидания, дн.": p["waiting_days"], "Норма": p["rate"],
        } for p in result["products"]])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Сверить с Государственным каталогом пестицидов и агрохимикатов")
    if result["excluded"]:
        st.markdown("**Исключённые д.в.:**")
        for item in result["excluded"]:
            st.markdown(f":gray[~~{item['product']['name_ru']}~~ — {item['reason']}]")

    disease = result["disease"]
    if disease and result["status"] in ("confirmed", "probable"):
        st.markdown("**Агротехнические меры:** " + ", ".join(disease["measures"]))
    for message in result["messages"]:
        st.info(message)


def show_explanation(result):
    with st.expander("Почему такое решение", expanded=True):
        if not result["trace"]:
            st.write("Ни одно правило не сработало.")
        for step in result["trace"]:
            st.markdown(
                f"**{step['id']}** · {step['group']} · CF **{step['cf']:.2f}**  \n"
                f"{step['text']}  \n:gray[→ {step['result']}]"
            )
    with st.expander("Рабочая память"):
        memory = pd.DataFrame(
            [{"Факт": name, "Значение": str(value), "CF": round(cf, 2)}
             for name, (value, cf) in result["facts"].items()]
        )
        st.dataframe(memory, hide_index=True, width="stretch")


def show_sources(hits):
    for hit in hits:
        date = f" · {hit['date']}" if hit["date"] else ""
        st.markdown(f"**[{hit['title']}]({hit['url'].replace(' ', '%20')})**{date} · близость {hit['score']:.2f}  \n"
                    f":gray[{hit['text']}]")


def show_documents(result, field, kb):
    # RAG: к решению прикладываем фрагменты официальных документов о найденной болезни
    disease = result["disease"]
    if not disease or result["status"] not in ("confirmed", "probable"):
        return
    query = f"{disease['name_ru']} {kb['crops'][field['crop']]} меры защиты порог вредоносности"
    with st.expander("Что пишут официальные источники (поиск по документам)"):
        show_sources(rag.search(query))


def tab_diagnostics(kb, field):
    image = pick_image()
    clicked = st.button("Получить решение", type="primary", disabled=image is None)
    photo, left, right = st.columns([2, 3, 3])
    if image is not None:
        photo.image(image, caption="Фото листа", width="stretch")
    if not clicked:
        return

    try:
        classifier.load_model()  # чтобы загрузка не попала в замер времени
        start = time.perf_counter()
        preds = classifier.predict(image, kb["labels"])
        elapsed = time.perf_counter() - start
    except Exception as error:
        st.error(f"Не удалось загрузить или запустить модель {config.MODEL_ID}: {error}")
        return
    result = engine.run(engine.build_facts(field, preds, kb), kb)
    save_history(field, preds, result)

    with left:
        show_network(preds, kb, elapsed)
    with right:
        show_decision(result, kb)
    show_products(result)
    show_explanation(result)
    show_documents(result, field, kb)


def tab_knowledge(kb):
    def demo_mark(item):
        return "условные значения" if item.get("demo") else ""

    st.subheader("Болезни")
    st.dataframe(pd.DataFrame([{
        "id": d["id"], "Культура": kb["crops"].get(d["crop"], d["crop"]), "Болезнь": d["name_ru"],
        "Возбудитель": d["pathogen"], "ЭПВ, %": f"{d['threshold_min']}–{d['threshold_max']}",
        "t, °C": f"{d['t_min']}–{d['t_max']}", "RH ≥, %": d["rh_min"],
        "Д.в.": ", ".join(d["products"]), "Пометка": demo_mark(d),
    } for d in kb["diseases"]]), hide_index=True, width="stretch")

    st.subheader("Действующие вещества")
    st.dataframe(pd.DataFrame([{
        "id": p["id"], "Д.в.": p["name_ru"], "Класс": p["chem_class"], "FRAC": p["frac"],
        "Действие": p["mode"], "Срок ожидания, дн.": p["waiting_days"], "Норма": p["rate"],
        "Пометка": demo_mark(p),
    } for p in kb["products"]]), hide_index=True, width="stretch")
    st.caption("Сверить с Государственным каталогом пестицидов и агрохимикатов")

    st.subheader("Правила")
    st.dataframe(pd.DataFrame([{
        "ID": r["id"], "Группа": r["group"], "Правило": r["text"], "CF": r["cf"],
    } for r in kb["rules"]]), hide_index=True, width="stretch")


def tab_documents():
    st.caption(f"{len(list(config.DOCS_DIR.glob('*.txt')))} документов Россельхозцентра и Минсельхозпрода РТ, "
               f"поиск по смыслу моделью {config.RAG_MODEL_ID}")
    query = st.text_input("Вопрос", placeholder="например: когда опрыскивать картофель от фитофтороза")
    if query:
        show_sources(rag.search(query, k=5))


def tab_history():
    if not config.HISTORY_PATH.exists():
        st.info("Журнал пуст: решения появятся после нажатия «Получить решение».")
        return
    history = pd.read_csv(config.HISTORY_PATH, dtype=str)
    st.dataframe(history.iloc[::-1], hide_index=True, width="stretch")


# БЗ читаем каждый раз, чтобы правки JSON подхватывались без перезапуска
kb = engine.load_kb(config.KB_DIR)
st.title("Защита посевов: гибридная СППР")
field = sidebar(kb)

diagnostics, knowledge, documents, history = st.tabs(["Диагностика", "База знаний", "Документы", "Журнал"])
with diagnostics:
    tab_diagnostics(kb, field)
with knowledge:
    tab_knowledge(kb)
with documents:
    tab_documents()
with history:
    tab_history()

st.caption(
    "Учебный прототип. Перед применением пестицидов сверяйтесь с Государственным каталогом "
    "пестицидов и агрохимикатов, разрешённых к применению на территории РФ."
)
