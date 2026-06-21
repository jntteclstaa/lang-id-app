# Language Identification — Web App

NLP Final Project · Group 3

## Struktur Folder

```
lang-id-app/
├── train_and_save.py     ← Jalankan SEKALI untuk training
├── app.py                ← Backend Flask (jalankan setiap mau pakai)
├── templates/
│   └── index.html        ← Frontend web
├── models/               ← Otomatis dibuat saat training
│   ├── baseline_svm.joblib
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── char_cnn/         ← Keras SavedModel
│   ├── char_cnn_tokenizer.json
│   ├── char_cnn_label_encoder.pkl
│   ├── char_cnn_meta.json
│   ├── xlmr/             ← HuggingFace SavedModel
│   ├── xlmr_label_encoder.pkl
│   └── metrics.json      ← Tampil di web sebagai tabel perbandingan
├── dataset.csv           ← ⚠ Taruh file dataset di sini
└── requirements.txt
```

---
# Dikarenakan models nya memiliki ukuran yang sangat besar, jadi sebelum di compile & run,
# install terlebih dahulu untuk data models nya melalui google drive :
# https://drive.google.com/file/d/1ZOT2tzNhvjoSe2GdVtdbxRBtKeZq_c3M/view?usp=sharing
---

## Setup di VS Code

### 1. Install dependensi

```bash
# Buat virtual environment (disarankan)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install packages
pip install -r requirements.txt
```

> Kalau tidak pakai Char-CNN, hapus baris `tensorflow` dari requirements.txt  
> Kalau tidak pakai XLM-R, hapus baris `torch`, `transformers`, `datasets`, `accelerate`

---

### 2. Taruh dataset

Copy file `dataset.csv` ke folder `lang-id-app/`.  
Dataset harus punya kolom `text` dan `language`.

---

### 3. Training (sekali saja!)

```bash
# Train semua model (termasuk Char-CNN dan XLM-R)
python train_and_save.py --data dataset.csv

# Skip Char-CNN (lebih cepat, tidak butuh TensorFlow)
python train_and_save.py --data dataset.csv --skip-cnn

# Skip XLM-R (lebih cepat, tidak butuh GPU)
python train_and_save.py --data dataset.csv --skip-xlmr

# Skip keduanya (hanya 4 model classical)
python train_and_save.py --data dataset.csv --skip-cnn --skip-xlmr
```

Model akan tersimpan di folder `models/`. **Tidak perlu training ulang** setiap restart.

---

### 4. Jalankan web app

```bash
python app.py
```

Buka browser di: **http://localhost:5000**

---

## Cara Pakai Web

1. **Pilih model** — klik salah satu dari 6 card model di bagian atas
2. **Masukkan teks** — bisa pakai demo pills atau ketik sendiri
3. **Klik "Identifikasi Bahasa"** — atau tekan Ctrl+Enter
4. Lihat hasil prediksi + top-5 confidence bars

---

## Tips

- Model classical (SVM, NB, LR) sangat cepat (<1ms per prediksi)
- Char-CNN butuh TensorFlow; load pertama kali agak lama (~5 detik)
- XLM-R butuh RAM besar (~2GB); GPU opsional tapi disarankan
- Model di-cache dalam memori setelah pertama kali diload — request berikutnya langsung cepat
- Card model yang **abu-abu** berarti belum ditraining — jalankan `train_and_save.py` dulu

---

## Model yang Tersedia

| Model | Type | N-gram |
|-------|------|--------|
| Baseline SVM | Classical | n=1 (unigram) |
| Naive Bayes | Classical | n=2–3 |
| Logistic Regression | Classical | n=2–3 |
| SVM (n=2–3) | Classical | n=2–3 |
| Char-CNN | Deep Learning | — |
| XLM-RoBERTa | Transformer | — |
