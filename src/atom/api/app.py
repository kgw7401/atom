"""FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from atom.api.routers.sessions import router as sessions_router
from atom.api.routers.history import router as history_router
from atom.api.routers.today import router as today_router
from atom.models.base import async_session, init_db
from atom.seed import seed_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with async_session() as session:
        counts = await seed_all(session)
    print(f"[atom] DB ready. Seeded: {counts}")
    yield


app = FastAPI(
    title="Atom",
    version="0.3.0",
    description="AI Boxing Coach API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(history_router)
app.include_router(today_router)

# Serve audio chunk files (self-recorded)
AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


@app.get("/")
async def root():
    return {"name": "Atom", "version": "0.3.0", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
