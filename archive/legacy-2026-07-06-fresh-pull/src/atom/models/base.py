"""SQLAlchemy async engine and session setup.

Uses SQLite (aiosqlite) for local development.
Switch to PostgreSQL by changing ATOM_DATABASE_URL env var.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Default DB location: <project-root>/data/atom.db
DEFAULT_DB_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "atom.db"
DEFAULT_DB_URL = f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}"

DATABASE_URL = os.getenv("ATOM_DATABASE_URL", DEFAULT_DB_URL)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db_path() -> Path:
    """Return the resolved database file path."""
    url = DATABASE_URL
    if url.startswith("sqlite"):
        # Extract path from sqlite URL
        path = url.split("///", 1)[1]
        return Path(path)
    return DEFAULT_DB_PATH


async def init_db() -> None:
    """Create all tables (dev only — use Alembic for production)."""
    get_db_path().parent.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
