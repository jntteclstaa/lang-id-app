import os
import shutil
import zipfile
from pathlib import Path

import gdown


MODELS_DIR = Path("models")
MODEL_DRIVE_ID = os.getenv("MODEL_DRIVE_ID", "1ZOT2tzNhvjoSe2GdVtdbxRBtKeZq_c3M")
MODEL_ARCHIVE_PATH = Path(os.getenv("MODEL_ARCHIVE_PATH", "/tmp/lang-id-models.zip"))

REQUIRED_MODEL_PATHS = [
    MODELS_DIR / "baseline_svm.joblib",
    MODELS_DIR / "naive_bayes.joblib",
    MODELS_DIR / "logistic_regression.joblib",
    MODELS_DIR / "svm.joblib",
    MODELS_DIR / "char_cnn.keras",
    MODELS_DIR / "char_cnn_tokenizer.json",
    MODELS_DIR / "char_cnn_label_encoder.pkl",
    MODELS_DIR / "char_cnn_meta.json",
    MODELS_DIR / "xlmr",
    MODELS_DIR / "xlmr_label_encoder.pkl",
    MODELS_DIR / "metrics.json",
]


def models_ready() -> bool:
    return all(path.exists() for path in REQUIRED_MODEL_PATHS)


def _flatten_single_nested_models_dir() -> None:
    nested = MODELS_DIR / "models"
    if not nested.is_dir():
        return

    for item in nested.iterdir():
        target = MODELS_DIR / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))

    try:
        nested.rmdir()
    except OSError:
        pass


def download_models_if_needed() -> bool:
    MODELS_DIR.mkdir(exist_ok=True)

    if models_ready():
        print("[models] Model files already exist. Skipping download.")
        return False

    print("[models] Model files not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
    gdown.download(url, str(MODEL_ARCHIVE_PATH), quiet=False)

    if not MODEL_ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Download failed: {MODEL_ARCHIVE_PATH}")

    print("[models] Extracting model archive...")
    with zipfile.ZipFile(MODEL_ARCHIVE_PATH, "r") as archive:
        archive.extractall(MODELS_DIR)

    _flatten_single_nested_models_dir()

    if not models_ready():
        found = sorted(str(path) for path in MODELS_DIR.rglob("*") if path.is_file())
        raise RuntimeError(
            "Model download completed, but required model files were not found. "
            f"Found files: {found[:30]}"
        )

    print("[models] Models are ready.")
    return True


if __name__ == "__main__":
    download_models_if_needed()
