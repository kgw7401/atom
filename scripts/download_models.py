"""Download MediaPipe model files for Track B.

Run once before using the pose estimator:
    python scripts/download_models.py
"""

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
}


def download(name: str, url: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"  [skip] {name} already exists ({dest.stat().st_size // 1024} KB)")
        return
    print(f"  [download] {name} ...", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f" done ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Downloading models to {MODELS_DIR}/")
    for name, url in MODELS.items():
        download(name, url)
    print("Done.")


if __name__ == "__main__":
    main()
