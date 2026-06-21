import argparse
import json
import os
import pickle
import re
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)



# PREPROCESSING
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 1 else ""


def augment_with_short_texts(df, n_short=200):
    import random
    random.seed(42)

    char_level_langs = {"Chinese", "Japanese", "Thai"}

    rows = []
    for _, row in df.iterrows():
        text = str(row['text'])
        lang = row['language']

        if lang in char_level_langs:
            # Potong 20–60 karakter pertama
            if len(text) > 60:
                cut = random.randint(20, 60)
                short_text = text[:cut]
            else:
                continue
        else:
            words = text.split()
            if len(words) > 15:
                cut = random.randint(5, 15)
                short_text = ' '.join(words[:cut])
            else:
                continue

        rows.append({'language': lang, 'text': short_text})

    extra = pd.DataFrame(rows)
    extra = extra.groupby('language').head(n_short)
    return pd.concat([df, extra], ignore_index=True)

# LOAD & SPLIT DATA
def load_data(csv_path: str):
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", sep=",", on_bad_lines="skip", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["language", "text"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV harus punya kolom: {required}. Ditemukan: {df.columns.tolist()}")

    df.dropna(subset=["language"], inplace=True)
    df = augment_with_short_texts(df, n_short=200)
    df["clean_text"] = df["text"].apply(preprocess)

    counts = df["language"].value_counts()
    low = counts[counts < 50].index
    df = df[~df["language"].isin(low)]
    df = df[df["clean_text"].str.len() > 0]

    print(f"  Shape        : {df.shape}")
    print(f"  Languages    : {df['language'].nunique()} → {sorted(df['language'].unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["language"],
        test_size=0.20, random_state=42, stratify=df["language"]
    )
    print(f"  Train / Test : {len(X_train):,} / {len(X_test):,} samples\n")
    return X_train, X_test, y_train, y_test, df

# CLASSICAL MODELS (sklearn Pipeline = vectorizer + clf)
def train_classical(X_train, X_test, y_train, y_test):
    configs = {
        "baseline_svm": {
            "label": "Baseline SVM (n=1)",
            "vec": TfidfVectorizer(analyzer="char", ngram_range=(1, 1),
                                   sublinear_tf=True, max_features=50_000),
            "clf": CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42), cv=3),
        },
        "naive_bayes": {
            "label": "Naive Bayes (n=2-3)",
            "vec": TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                   sublinear_tf=True, max_features=50_000),
            "clf": MultinomialNB(alpha=0.1),
        },
        "logistic_regression": {
            "label": "Logistic Regression (n=2-3)",
            "vec": TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                   sublinear_tf=True, max_features=50_000),
            "clf": LogisticRegression(max_iter=1000, C=5, solver="lbfgs",
                                      random_state=42),
        },
        "svm": {
            "label": "SVM (n=2-3)",
            "vec": TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                   sublinear_tf=True, max_features=50_000),
            "clf": CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42), cv=3),
        },
    }

    metrics_all = {}
    for key, cfg in configs.items():
        print(f"{'='*55}")
        print(f"  🔧 Training: {cfg['label']}")
        print(f"{'='*55}")

        pipe = Pipeline([("vec", cfg["vec"]), ("clf", cfg["clf"])])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        infer_time = (time.perf_counter() - t1) / len(X_test) * 1000

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        print(f"  Accuracy   : {acc*100:.2f}%")
        print(f"  Macro F1   : {f1*100:.2f}%")
        print(f"  Train time : {train_time:.2f}s")
        print(f"  Infer time : {infer_time:.4f} ms/sample\n")

        # Simpan pipeline
        save_path = os.path.join(MODELS_DIR, f"{key}.joblib")
        joblib.dump(pipe, save_path)
        print(f"  ✅ Saved → {save_path}\n")

        metrics_all[key] = {
            "label": cfg["label"],
            "accuracy": round(acc * 100, 2),
            "macro_f1": round(f1 * 100, 2),
            "train_time_s": round(train_time, 2),
            "infer_ms": round(infer_time, 4),
        }

    return metrics_all

# CHAR-CNN (TensorFlow/Keras)
def train_char_cnn(X_train, X_test, y_train, y_test, df):
    print(f"{'='*55}")
    print(f"  🔧 Training: Char-CNN")
    print(f"{'='*55}")

    try:
        import tensorflow as tf
        from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Embedding,
                                              GlobalMaxPooling1D, Input)
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
    except ImportError:
        print("  ⚠️  TensorFlow tidak ditemukan. Skip Char-CNN.")
        print("     Install dengan: pip install tensorflow\n")
        return None

    MAX_LEN = 256
    EPOCHS  = 10
    BATCH   = 64
    EMB_DIM = 100

    # Tokenizer karakter
    char_tok = Tokenizer(char_level=True, lower=True, filters=None)
    char_tok.fit_on_texts(df["text"])
    vocab_size = len(char_tok.word_index) + 1

    # Label encoder
    le = LabelEncoder()
    le.fit(df["language"])
    NUM_CLASSES = len(le.classes_)

    X_tr = pad_sequences(char_tok.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post")
    X_te = pad_sequences(char_tok.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding="post")
    y_tr = tf.keras.utils.to_categorical(le.transform(y_train), num_classes=NUM_CLASSES)
    y_te = tf.keras.utils.to_categorical(le.transform(y_test),  num_classes=NUM_CLASSES)

    # Build model
    inp = Input(shape=(MAX_LEN,))
    emb = Embedding(vocab_size, EMB_DIM, input_length=MAX_LEN)(inp)
    conv_outs = []
    for fs in [3, 4, 5]:
        c = Conv1D(128, fs, activation="relu")(emb)
        p = GlobalMaxPooling1D()(c)
        conv_outs.append(p)
    x = tf.keras.layers.concatenate(conv_outs)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
            validation_data=(X_te, y_te), verbose=1)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()                                   
    y_pred_proba = model.predict(X_te)
    y_pred_idx   = y_pred_proba.argmax(axis=1)
    y_pred       = le.inverse_transform(y_pred_idx)
    y_true_str   = le.inverse_transform(le.transform(y_test))

    acc = accuracy_score(y_true_str, y_pred)
    f1  = f1_score(y_true_str, y_pred, average="macro")
    infer_ms = (time.perf_counter() - t1) / len(X_te) * 1000  # ← pakai t1
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Macro F1   : {f1*100:.2f}%\n")

    # Simpan
    model_path = os.path.join(MODELS_DIR, "char_cnn.keras")
    model.save(model_path)

    tok_cfg = char_tok.to_json()
    with open(os.path.join(MODELS_DIR, "char_cnn_tokenizer.json"), "w") as f:
        json.dump(tok_cfg, f)

    with open(os.path.join(MODELS_DIR, "char_cnn_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    meta = {"max_len": MAX_LEN}
    with open(os.path.join(MODELS_DIR, "char_cnn_meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"  ✅ Saved → {model_path}/  (+ tokenizer + label encoder)\n")

    return {
        "label": "Char-CNN",
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(f1 * 100, 2),
        "train_time_s": round(train_time, 2),
        "infer_ms": round(infer_ms, 4),
    }

# XLM-RoBERTa (HuggingFace Transformers)
def train_xlmr(X_train, X_test, y_train, y_test,all_labels):
    print(f"{'='*55}")
    print(f"  🔧 Training: XLM-RoBERTa")
    print(f"{'='*55}")

    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, Trainer, TrainingArguments)
    except ImportError:
        print("  ⚠️  transformers / datasets tidak ditemukan. Skip XLM-R.")
        print("     Install dengan: pip install transformers datasets accelerate torch\n")
        return None

    MODEL_NAME = "xlm-roberta-base"
    XLMR_DIR   = os.path.join(MODELS_DIR, "xlmr")

    le = LabelEncoder()
    le.fit(all_labels)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tok(example["text"], padding="max_length", truncation=True, max_length=64)

    train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": le.transform(y_train)})
    test_ds  = Dataset.from_dict({"text": X_test.tolist(),  "label": le.transform(y_test)})
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "xlmr_checkpoints"),
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=8,
        per_device_eval_batch_size=8, num_train_epochs=3,
        weight_decay=0.01, load_best_model_at_end=True,
        logging_dir="./logs", report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    eval_res = trainer.evaluate()
    acc = eval_res["eval_accuracy"]
    f1  = eval_res["eval_macro_f1"]
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Macro F1   : {f1*100:.2f}%\n")

    # Simpan
    trainer.save_model(XLMR_DIR)
    tok.save_pretrained(XLMR_DIR)
    with open(os.path.join(MODELS_DIR, "xlmr_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"  ✅ Saved → {XLMR_DIR}/\n")

    return {
        "label": "XLM-RoBERTa",
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(f1 * 100, 2),
        "train_time_s": round(train_time, 2),
        "infer_ms": 0,
    }


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.csv", help="Path ke CSV dataset")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip training Char-CNN")
    parser.add_argument("--skip-xlmr", action="store_true", help="Skip training XLM-R")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, df = load_data(args.data)

    # Classical models
    all_metrics = train_classical(X_train, X_test, y_train, y_test)

    # Char-CNN
    if not args.skip_cnn:
        cnn_metrics = train_char_cnn(X_train, X_test, y_train, y_test, df)
        if cnn_metrics:
            all_metrics["char_cnn"] = cnn_metrics
    else:
        print("  ⏭  Skipping Char-CNN (--skip-cnn)\n")

    # XLM-R
    if not args.skip_xlmr:
        xlmr_metrics = train_xlmr(X_train, X_test, y_train, y_test, df["language"])
        if xlmr_metrics:
            all_metrics["xlmr"] = xlmr_metrics
    else:
        print("  ⏭  Skipping XLM-RoBERTa (--skip-xlmr)\n")

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n📊 Metrics saved → {metrics_path}")
    print("\n✅ Semua model berhasil disimpan! Sekarang jalankan: python app.py\n")


if __name__ == "__main__":
    main()
