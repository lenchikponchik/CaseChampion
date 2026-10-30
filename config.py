# настройки: модель, пороги, пути
from pathlib import Path

# ViT-Tiny: пшеница, кукуруза, картофель, рис
MODEL_ID = "wambugu71/crop_leaf_diseases_vit"
# запасная (PlantVillage, нет пшеницы и риса), но с transformers 5 не грузится
# MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

TOP_K = 3

# пороги уверенности, правила R3/R5/R6/R7 берут их через ref
CONFIDENCE = {
    "prob_low": 0.60,   # R3
    "prob_high": 0.85,  # R5
    "gap_min": 0.15,    # R7
}

# по умолчанию = сценарий 1, их же использует evaluate.py
FIELD_DEFAULTS = {
    "pct": 6, "temp": 20.0, "humidity": 85, "rain_3d": True, "rain_6h": False,
    "wind": 3.0, "days_to_harvest": 40, "last_frac": "нет", "lat": 55.75, "lon": 37.62,
}

# осадки, мм
RAIN_3D_MM = 1.0
RAIN_6H_MM = 0.5

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "kb"
EXAMPLES_DIR = BASE_DIR / "examples"
HISTORY_PATH = BASE_DIR / "history.csv"
DATA_DIR = BASE_DIR / "data"  # тестовые фото, в git не кладём
RESULTS_DIR = BASE_DIR / "results"
