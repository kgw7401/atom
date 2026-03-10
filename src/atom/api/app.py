"""FastAPI application — optional server for external API access."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atom.api.routers.combos import router as combos_router
from atom.api.routers.sessions import router as sessions_router
from atom.api.routers.history import router as history_router
from atom.api.routers.ws_session import router as ws_router

app = FastAPI(title="Atom", version="0.1.0", description="AI Boxing Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(combos_router)
app.include_router(sessions_router)
app.include_router(history_router)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
