"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routers import scripts, sessions, users
from server.services.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and init DB on startup."""
    from server.models.db import init_db
    await init_db()
    app.state.inference = InferenceService()
    yield


app = FastAPI(
    title="Atom API",
    version="0.1.0",
    description="Boxing coaching MVP — session scripts, analysis, and digital twin",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scripts.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
