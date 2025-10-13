# app.py
import os, json
from flask import Flask, request, jsonify
from flask import Response
from functools import wraps
import joblib
import numpy as np

app = Flask(__name__)

# ===== Seguridad básica por API KEY =====
API_KEY = os.environ.get("API_KEY", "")  # config en Railway
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

# ===== Codificadores simples (si el modelo espera numéricos) =====
# Si tu modelo ya incluye Pipeline con OneHot/Ordinal, puedes NO mapear aquí.
MAP_EDUCATION = {
    "PREPARATORIA": 1, "LICENCIATURA": 2, "MAESTRÍA": 3, "MAESTRIA": 3, "DOCTORADO": 4
}
MAP_STRATEGY = {
    "RECOMENDADO": 1, "PORTALES DE EMPLEO": 2, "HEADHUNTING": 3
}

def norm(s):
    # Sube a mayúscula SIN acentos (eliminación simple)
    import unicodedata
    s = (s or "").strip().upper()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

def to_viability(score):
    if score >= 80: return "ALTA"
    if score >= 60: return "MEDIA"
    return "BAJA"

@app.route("/health", methods=["GET"])
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
        edu = MAP_EDUCATION.get(norm(data.get("EducationLevel","")), 1)
        yrs = int(data.get("ExperienceYears", 0))
        strat = MAP_STRATEGY.get(norm(data.get("RecruitmentStrategy","")), 1)
        ps = int(data.get("PersonalityScore", 0))
        ss = int(data.get("SkillScore", 0))
        iscore = int(data.get("InterviewScore", 0))

        X = np.array([[edu, yrs, strat, ps, ss, iscore]], dtype=float)

        # Si tu modelo predice probabilidad:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Suponiendo clase positiva = 1
            p = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            evaluation_score = int(round(p * 100))
        else:
            # Si es regresión o score directo 0-100
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
