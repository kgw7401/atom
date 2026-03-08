"""ComboService — CRUD operations for combinations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import Action, Combination


class ComboError(Exception):
    """Base error for combo operations."""
    pass


class ComboNotFoundError(ComboError):
    pass


class ComboValidationError(ComboError):
    pass


class ComboImmutableError(ComboError):
    pass


class ComboService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list(
        self,
        complexity: int | None = None,
    ) -> list[Combination]:
        """List all combinations, optionally filtered by complexity."""
        stmt = select(Combination).order_by(Combination.complexity, Combination.display_name)
        if complexity is not None:
            stmt = stmt.where(Combination.complexity == complexity)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get(self, id_or_name: str) -> Combination:
        """Get a combination by ID or display_name."""
        # Try by ID first
        stmt = select(Combination).where(Combination.id == id_or_name)
        result = await self.session.execute(stmt)
        combo = result.scalar_one_or_none()
        if combo is not None:
            return combo

        # Try by display_name
        stmt = select(Combination).where(Combination.display_name == id_or_name)
        result = await self.session.execute(stmt)
        combo = result.scalar_one_or_none()
        if combo is not None:
            return combo

        raise ComboNotFoundError(f"Combo not found: '{id_or_name}'")

    async def create(self, display_name: str, actions: list[str]) -> Combination:
        """Create a new user combo. Validates actions exist and display_name is unique."""
        # Validate display_name uniqueness
        stmt = select(Combination).where(Combination.display_name == display_name)
        result = await self.session.execute(stmt)
        if result.scalar_one_or_none() is not None:
            raise ComboValidationError(f"Combo '{display_name}' already exists")

        # Validate all actions exist
        await self._validate_actions(actions)

        combo = Combination(
            display_name=display_name,
            actions=actions,
            complexity=len(actions),
            is_system=False,
        )
        self.session.add(combo)
        await self.session.commit()
        await self.session.refresh(combo)
        return combo

    async def update(self, id_or_name: str, **kwargs) -> Combination:
        """Update a combo. System combos are immutable."""
        combo = await self.get(id_or_name)

        if combo.is_system:
            raise ComboImmutableError("System combos cannot be edited")

        if "display_name" in kwargs:
            # Check uniqueness of new name
            new_name = kwargs["display_name"]
            stmt = select(Combination).where(
                Combination.display_name == new_name,
                Combination.id != combo.id,
            )
            result = await self.session.execute(stmt)
            if result.scalar_one_or_none() is not None:
                raise ComboValidationError(f"Combo '{new_name}' already exists")
            combo.display_name = new_name

        if "actions" in kwargs:
            actions = kwargs["actions"]
            await self._validate_actions(actions)
            combo.actions = actions
            combo.complexity = len(actions)

        await self.session.commit()
        await self.session.refresh(combo)
        return combo

    async def delete(self, id_or_name: str) -> None:
        """Delete a combo. System combos cannot be deleted."""
        combo = await self.get(id_or_name)

        if combo.is_system:
            raise ComboImmutableError("System combos cannot be deleted")

        await self.session.delete(combo)
        await self.session.commit()

    async def _validate_actions(self, actions: list[str]) -> None:
        """Validate all action names exist in the Actions table."""
        if not actions:
            raise ComboValidationError("Combo must have at least one action")

        result = await self.session.execute(select(Action.name))
        valid_names = {row[0] for row in result.all()}

        invalid = [a for a in actions if a not in valid_names]
        if invalid:
            raise ComboValidationError(
                f"Action(s) not found: {', '.join(repr(a) for a in invalid)}"
            )
