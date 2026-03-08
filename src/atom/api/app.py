"""FastAPI application — optional server for external API access."""

from fastapi import FastAPI

from atom.api.routers.combos import router as combos_router
from atom.api.routers.sessions import router as sessions_router
from atom.api.routers.history import router as history_router

app = FastAPI(title="Atom", version="0.1.0", description="AI Boxing Coach API")

app.include_router(combos_router)
app.include_router(sessions_router)
app.include_router(history_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
