"""
app.py
======
Flask backend untuk Language Identification Web App.
Jalankan setelah train_and_save.py selesai.

Usage:
    python app.py
Lalu buka: http://localhost:5000
"""

import json
import os
import pickle
import re
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
from flask import Flask, jsonify, render_template, request

MODELS_DIR = "models"
app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# LAZY-LOAD MODEL CACHE (load saat pertama kali dipakai)
# ─────────────────────────────────────────────────────────
_loaded_models = {}


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 1 else ""


def check_model_exists(model_key: str) -> bool:
    """Cek apakah file model tersedia."""
    paths = {
        "baseline_svm":         os.path.join(MODELS_DIR, "baseline_svm.joblib"),
        "naive_bayes":          os.path.join(MODELS_DIR, "naive_bayes.joblib"),
        "logistic_regression":  os.path.join(MODELS_DIR, "logistic_regression.joblib"),
        "svm":                  os.path.join(MODELS_DIR, "svm.joblib"),
        "char_cnn":             os.path.join(MODELS_DIR, "char_cnn.keras"),
        "xlmr":                 os.path.join(MODELS_DIR, "xlmr"),
    }
    path = paths.get(model_key)
    if path is None:
        return False
    return os.path.exists(path)


def load_model(model_key: str):
    """Load dan cache model berdasarkan key."""
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    print(f"  [load] Loading model: {model_key}...")

    if model_key in ("baseline_svm", "naive_bayes", "logistic_regression", "svm"):
        path = os.path.join(MODELS_DIR, f"{model_key}.joblib")
        pipe = joblib.load(path)
        _loaded_models[model_key] = ("sklearn", pipe)

    elif model_key == "char_cnn":
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        model_path = os.path.join(MODELS_DIR, "char_cnn.keras")
        tok_path   = os.path.join(MODELS_DIR, "char_cnn_tokenizer.json")
        le_path    = os.path.join(MODELS_DIR, "char_cnn_label_encoder.pkl")
        meta_path  = os.path.join(MODELS_DIR, "char_cnn_meta.json")

        keras_model = tf.keras.models.load_model(model_path)
        with open(tok_path) as f:
            char_tok = tokenizer_from_json(json.load(f))
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        _loaded_models[model_key] = ("char_cnn", keras_model, char_tok, le, meta)

    elif model_key == "xlmr":
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)

        xlmr_path = os.path.join(MODELS_DIR, "xlmr")
        le_path   = os.path.join(MODELS_DIR, "xlmr_label_encoder.pkl")

        tok   = AutoTokenizer.from_pretrained(xlmr_path)
        model = AutoModelForSequenceClassification.from_pretrained(xlmr_path)
        model.eval()
        with open(le_path, "rb") as f:
            le = pickle.load(f)

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        model.to(device)
        _loaded_models[model_key] = ("xlmr", model, tok, le, device)

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    print(f"  [load] ✅ {model_key} ready")
    return _loaded_models[model_key]


# ─────────────────────────────────────────────────────────
# PREDICT HELPERS
# ─────────────────────────────────────────────────────────
def predict_sklearn(pipe, text: str):
    clean = preprocess(text)
    if not clean:
        return None, {}

    clf = pipe.named_steps["clf"]
    vec = pipe.named_steps["vec"]
    X   = vec.transform([clean])

    if hasattr(clf, "predict_proba"):
        proba   = clf.predict_proba(X)[0]
        classes = clf.classes_
    else:
        # LinearSVC → proper softmax dari decision_function
        import numpy as np
        dec     = clf.decision_function(X)[0]
        dec_arr = np.array(dec)
        exp_dec = np.exp(dec_arr - dec_arr.max())  # stable softmax
        proba   = exp_dec / exp_dec.sum()
        classes = clf.classes_

    top5 = sorted(zip(classes, proba), key=lambda x: -x[1])[:5]
    pred = top5[0][0]
    conf = {lang: round(float(p) * 100, 2) for lang, p in top5}

    return pred, conf


def predict_char_cnn(keras_model, char_tok, le, meta, text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    clean = preprocess(text)
    if not clean:
        return None, {}

    seq   = char_tok.texts_to_sequences([clean])
    seq   = pad_sequences(seq, maxlen=meta["max_len"], padding="post")
    proba = keras_model.predict(seq, verbose=0)[0]

    top5_idx  = proba.argsort()[-5:][::-1]
    pred      = le.inverse_transform([top5_idx[0]])[0]
    conf      = {le.inverse_transform([i])[0]: round(float(proba[i]) * 100, 2)
                 for i in top5_idx}
    return pred, conf


def predict_xlmr(model, tok, le, device, text: str):
    import torch

    clean = preprocess(text) or text.strip()
    if not clean:
        return None, {}

    inputs = tok(clean, return_tensors="pt", padding=True,
                 truncation=True, max_length=64).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    proba    = torch.softmax(logits, dim=-1).cpu().numpy()
    top5_idx = proba.argsort()[-5:][::-1]
    pred     = le.inverse_transform([top5_idx[0]])[0]
    conf     = {le.inverse_transform([i])[0]: round(float(proba[i]) * 100, 2)
                for i in top5_idx}
    return pred, conf


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Baca metrics kalau ada
    metrics = {}
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Cek ketersediaan model
    model_keys = ["baseline_svm", "naive_bayes", "logistic_regression",
                  "svm", "char_cnn", "xlmr"]
    availability = {k: check_model_exists(k) for k in model_keys}

    return render_template("index.html", metrics=metrics, availability=availability)


@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json()
    text       = data.get("text", "").strip()
    model_key  = data.get("model", "svm")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong."}), 400

    if not check_model_exists(model_key):
        return jsonify({"error": f"Model '{model_key}' belum dilatih. Jalankan train_and_save.py terlebih dahulu."}), 400

    try:
        t0      = time.perf_counter()
        loaded  = load_model(model_key)
        kind    = loaded[0]

        if kind == "sklearn":
            pipe = loaded[1]
            pred, conf = predict_sklearn(pipe, text)
        elif kind == "char_cnn":
            _, keras_model, char_tok, le, meta = loaded
            pred, conf = predict_char_cnn(keras_model, char_tok, le, meta, text)
        elif kind == "xlmr":
            _, model, tok, le, device = loaded
            pred, conf = predict_xlmr(model, tok, le, device, text)
        else:
            return jsonify({"error": "Model type tidak dikenal."}), 500

        infer_ms = round((time.perf_counter() - t0) * 1000, 2)

        if pred is None:
            return jsonify({"error": "Teks terlalu pendek atau tidak valid setelah preprocessing."}), 400

        return jsonify({
            "prediction": pred,
            "confidence": conf,
            "infer_ms":   infer_ms,
            "model":      model_key,
            "low_confidence": conf.get(pred, 100) < 40.0,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models/status")
def models_status():
    keys = ["baseline_svm", "naive_bayes", "logistic_regression",
            "svm", "char_cnn", "xlmr"]
    return jsonify({k: check_model_exists(k) for k in keys})


if __name__ == "__main__":
    print("\n🌐 Language Identification Web App")
    print("   Buka browser di: http://localhost:5001\n")

    # Warmup char_cnn saat startup supaya tidak lambat di request pertama
    if check_model_exists("char_cnn"):
        print("  [warmup] Warming up Char-CNN...")
        loaded = load_model("char_cnn")
        _, keras_model, char_tok, le, meta = loaded
        predict_char_cnn(keras_model, char_tok, le, meta, "warmup text")
        print("  [warmup] ✅ Char-CNN ready\n")
    
    if check_model_exists("xlmr"):
        print("  [warmup] Warming up XLM-RoBERTa...")
        loaded = load_model("xlmr")
        _, model, tok, le, device = loaded
        predict_xlmr(model, tok, le, device, "warmup text")
        print("  [warmup] ✅ XLM-RoBERTa ready\n")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False
    )