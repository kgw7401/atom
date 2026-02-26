"""FastAPI application for Project Atom."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.models.db import init_db
from server.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + ensure uploads dir exists."""
    await init_db()
    Path("uploads").mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="Project Atom",
    description="State-vector-based adaptive boxing training system",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from server.routers import sessions, training, users  # noqa: E402

app.include_router(users.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")


@app.get("/", response_model=HealthResponse)
async def health():
    return HealthResponse()
