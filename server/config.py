"""Server configuration loaded from environment variables.

All settings use ATOM_ prefix. Example: ATOM_DATABASE_URL=...
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database (SQLite for local dev, PostgreSQL for production)
    database_url: str = "sqlite+aiosqlite:///atom_dev.db"

    # Model paths
    lstm_checkpoint: Path = Path("models/lstm_best.pt")
    pose_model: Path = Path("models/pose_landmarker.task")
    config_path: Path = Path("configs/boxing.yaml")

    # Analysis
    confidence_threshold: float = 0.7

    model_config = {"env_prefix": "ATOM_"}


settings = Settings()
