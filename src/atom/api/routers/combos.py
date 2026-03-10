"""Combo CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from atom.api.schemas import ComboCreate, ComboResponse, ComboUpdate
from atom.models.base import async_session
from atom.services.combo_service import (
    ComboImmutableError,
    ComboNotFoundError,
    ComboService,
    ComboValidationError,
)

router = APIRouter(prefix="/api/combos", tags=["combos"])


@router.get("", response_model=list[ComboResponse])
async def list_combos(complexity: int | None = Query(default=None)):
    async with async_session() as session:
        svc = ComboService(session)
        return await svc.list(complexity=complexity)


@router.get("/{combo_id}", response_model=ComboResponse)
async def get_combo(combo_id: str):
    async with async_session() as session:
        svc = ComboService(session)
        try:
            return await svc.get(combo_id)
        except ComboNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("", response_model=ComboResponse, status_code=201)
async def create_combo(body: ComboCreate):
    async with async_session() as session:
        svc = ComboService(session)
        try:
            return await svc.create(body.display_name, body.actions)
        except ComboValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))


@router.put("/{combo_id}", response_model=ComboResponse)
async def update_combo(combo_id: str, body: ComboUpdate):
    kwargs = body.model_dump(exclude_none=True)
    if not kwargs:
        raise HTTPException(status_code=400, detail="No fields to update")

    async with async_session() as session:
        svc = ComboService(session)
        try:
            return await svc.update(combo_id, **kwargs)
        except ComboNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ComboImmutableError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ComboValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{combo_id}", status_code=204)
async def delete_combo(combo_id: str):
    async with async_session() as session:
        svc = ComboService(session)
        try:
            await svc.delete(combo_id)
        except ComboNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ComboImmutableError as e:
            raise HTTPException(status_code=403, detail=str(e))
