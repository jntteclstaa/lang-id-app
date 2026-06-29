Ini yang siap paste:
# Language Identification — Web App

NLP Final Project · Group 3

## Struktur Folder

```text
lang-id-app/
├── download_models.py    ← Jalankan dulu untuk install/download model
├── train_and_save.py     ← Jalankan hanya jika ingin training ulang
├── app.py                ← Backend Flask
├── templates/
│   └── index.html        ← Frontend web
├── models/               ← Berisi model hasil download/training
│   ├── baseline_svm.joblib
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── char_cnn.keras
│   ├── char_cnn_tokenizer.json
│   ├── char_cnn_label_encoder.pkl
│   ├── char_cnn_meta.json
│   ├── xlmr/
│   ├── xlmr_label_encoder.pkl
│   └── metrics.json
├── dataset.csv           ← Dataset training
└── requirements.txt
```

---

## Setup di VS Code

### 1. Install dependensi

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> Disarankan memakai Python 3.11 atau 3.12 agar TensorFlow/Char-CNN bisa berjalan.

---

### 2. Install/download models terlebih dahulu

Karena ukuran model besar, file model tidak disimpan langsung di repository.

Sebelum menjalankan aplikasi atau melakukan training ulang, download model terlebih dahulu:

```bash
python download_models.py
```

Script ini akan otomatis:

1. Mengecek apakah folder `models/` sudah lengkap
2. Download model dari Google Drive jika belum ada
3. Extract model ke folder `models/`
4. Validasi file model yang dibutuhkan

Link sumber model:

```text
https://drive.google.com/file/d/1ZOT2tzNhvjoSe2GdVtdbxRBtKeZq_c3M/view?usp=sharing
```

---

### 3. Taruh dataset

Copy file `dataset.csv` ke folder `lang-id-app/`.

Dataset harus punya kolom `text` dan `language`.

---

### 4. Training ulang jika diperlukan

Training ulang hanya diperlukan jika ingin membuat model baru dari dataset.

Pastikan models sudah diinstall/download terlebih dahulu:

```bash
python download_models.py
```

Setelah itu jalankan training:

```bash
# Train semua model: classical + Char-CNN + XLM-R
python train_and_save.py --data dataset.csv

# Skip Char-CNN: lebih ringan karena tidak memakai TensorFlow
python train_and_save.py --data dataset.csv --skip-cnn

# Skip XLM-R: lebih cepat dan lebih hemat RAM karena tidak train transformer
python train_and_save.py --data dataset.csv --skip-xlmr

# Skip Char-CNN dan XLM-R: hanya train model classical
python train_and_save.py --data dataset.csv --skip-cnn --skip-xlmr
```

Model baru akan tersimpan di folder `models/`.

Penjelasan opsi training:

- `--skip-cnn` berarti tidak melatih model Char-CNN. Gunakan ini jika TensorFlow bermasalah atau ingin training lebih ringan.
- `--skip-xlmr` berarti tidak melatih model XLM-RoBERTa. Gunakan ini jika tidak ada GPU, RAM terbatas, atau ingin training lebih cepat.
- Tanpa skip berarti semua model dilatih, tetapi prosesnya paling lama dan paling berat.

Rekomendasi optimal:

- Untuk laptop biasa atau demo cepat: gunakan `--skip-cnn --skip-xlmr`.
- Untuk hasil yang tetap kuat tetapi tidak terlalu berat: gunakan `--skip-xlmr`.
- Untuk eksperimen lengkap dan mesin cukup kuat: jalankan tanpa skip.
- Untuk free deployment: model classical paling aman karena lebih kecil, cepat, dan hemat RAM.

---

### 5. Jalankan web app

```bash
python app.py
```

Buka browser di:

```text
http://localhost:5001
```

> Saat `python app.py` dijalankan, aplikasi juga akan otomatis mengecek dan mendownload model jika belum tersedia.

---

## Cara Pakai Web

1. **Pilih model** — klik salah satu card model di bagian atas
2. **Masukkan teks** — bisa pakai demo pills atau ketik sendiri
3. **Klik "Identifikasi Bahasa"** — atau tekan Ctrl+Enter
4. Lihat hasil prediksi dan top-5 confidence bars

---

## Tips

- Jalankan `python download_models.py` sebelum training atau menjalankan aplikasi
- Model classical (SVM, NB, LR) sangat cepat
- Char-CNN membutuhkan TensorFlow dan disarankan memakai Python 3.11/3.12
- XLM-R membutuhkan RAM besar dan load pertama kali lebih lama
- Model di-cache dalam memori setelah pertama kali diload
- Folder `models/` tidak perlu di-push ke Git karena ukurannya besar

---

## Model yang Tersedia

| Model | Type | N-gram |
|-------|------|--------|
| Baseline SVM | Classical | n=1 |
| Naive Bayes | Classical | n=2–3 |
| Logistic Regression | Classical | n=2–3 |
| SVM | Classical | n=2–3 |
| Char-CNN | Deep Learning | — |
| XLM-RoBERTa | Transformer | — |
