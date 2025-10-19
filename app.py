# app.py
import os
from flask import Flask, request, jsonify
from functools import wraps
import joblib
import numpy as np
import unicodedata

app = Flask(__name__)

# ===== Seguridad básica por API KEY =====
API_KEY = os.environ.get("API_KEY", "")
def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if not API_KEY or key != API_KEY:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# ===== Cargar modelo =====
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_rf_contratacion.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("ERROR cargando modelo:", e, flush=True)

# ===== Utilidades de normalización =====
def norm(s: str) -> str:
    """
    Mayúsculas sin acentos (NFD) y recorte de espacios.
    """
    s = (s or "").strip().upper()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s

def normalize_education(s: str) -> str:
    """
    Uniforma variantes de educación al catálogo esperado.
    """
    v = norm(s)
    if v == "MAESTRIA":          # sin acento -> con acento
        v = "MAESTRÍA"
    return v

def normalize_strategy(s: str) -> str:
    """
    Uniforma variantes de estrategia de reclutamiento al catálogo esperado.
    """
    v = norm(s)
    if v in ("PORTAL DE EMPLEO", "PORTALES", "PORTALES EMPLEO"):
        return "PORTALES DE EMPLEO"
    if v == "RECOMENDADA":
        return "RECOMENDADO"
    return v

# ===== Codificadores (1-BASED, como en entrenamiento) =====
MAP_EDUCATION = {
    "PREPARATORIA": 1,
    "LICENCIATURA": 2,
    "MAESTRÍA": 3,   # se llega aquí tras normalize_education()
    "DOCTORADO": 4,
}
MAP_STRATEGY = {
    "RECOMENDADO": 1,
    "PORTALES DE EMPLEO": 2,
    "HEADHUNTING": 3,
}

def to_viability(score: int) -> str:
    if score >= 80: return "ALTA"
    if score >= 60: return "MEDIA"
    return "BAJA"

@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": model is not None})

@app.post("/predict")
@require_api_key
def predict():
    """
    Espera JSON con:
    {
      "EducationLevel": "Preparatoria|Licenciatura|Maestría|Doctorado",
      "ExperienceYears": 0-40,
      "RecruitmentStrategy": "Recomendado|Portales de empleo|Headhunting",
      "PersonalityScore": 0-100,
      "SkillScore": 0-100,
      "InterviewScore": 0-100
    }
    """
    if model is None:
        return jsonify({"ok": False, "error": "Modelo no cargado"}), 500

    data = request.get_json(force=True, silent=True) or {}
    try:
        # Texto -> normalizado al catálogo; luego mapeo 1..4 / 1..3
        edu_txt = normalize_education(data.get("EducationLevel", ""))
        strat_txt = normalize_strategy(data.get("RecruitmentStrategy", ""))

        edu   = MAP_EDUCATION.get(edu_txt, 1)   # 1..4
        strat = MAP_STRATEGY.get(strat_txt, 1)  # 1..3
        yrs   = int(data.get("ExperienceYears", 0))

        ps    = int(data.get("PersonalityScore", 0))
        ss    = int(data.get("SkillScore", 0))
        iscore= int(data.get("InterviewScore", 0))

        # ORDEN EXACTO según feature_names_in_:
        # ['PersonalityScore','SkillScore','InterviewScore','EducationLevel','ExperienceYears','RecruitmentStrategy']
        X = np.array([[ps, ss, iscore, edu, yrs, strat]], dtype=float)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            p = float(proba[0, 1])  # classes_ = [0,1], positiva en índice 1
            evaluation_score = int(round(max(0.0, min(1.0, p)) * 100))
        else:
            y = float(model.predict(X)[0])
            evaluation_score = int(max(0, min(100, round(y))))

        viability = to_viability(evaluation_score)

        return jsonify({
            "ok": True,
            "evaluation_score": evaluation_score,
            "viability": viability
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/rank")
@require_api_key
def rank():
    """
    Espera JSON con:
      { "items": [ {"id": 123, "evaluation_score": 88}, ... ] }
    Devuelve items ordenados + rank 1..N
    """
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("items", [])
    try:
        items_sorted = sorted(items, key=lambda z: z.get("evaluation_score", 0), reverse=True)
        for i, it in enumerate(items_sorted, start=1):
            it["rank"] = i
        return jsonify({"ok": True, "items": items_sorted})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ===== Endpoint opcional de depuración para verificar el vector X
@app.post("/debug/echo")
@require_api_key
def debug_echo():
    data = request.get_json(force=True, silent=True) or {}

    edu_txt = normalize_education(data.get("EducationLevel", ""))
    strat_txt = normalize_strategy(data.get("RecruitmentStrategy", ""))

    edu   = MAP_EDUCATION.get(edu_txt, 1)
    strat = MAP_STRATEGY.get(strat_txt, 1)
    yrs   = int(data.get("ExperienceYears", 0))

    ps    = int(data.get("PersonalityScore", 0))
    ss    = int(data.get("SkillScore", 0))
    iscore= int(data.get("InterviewScore", 0))

    X = [ps, ss, iscore, edu, yrs, strat]
    return jsonify({"ok": True, "X": X, "edu_txt": edu_txt, "strat_txt": strat_txt})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
