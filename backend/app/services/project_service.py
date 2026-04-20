"""Project service — CRUD operations for projects and their wiki membership.

Projects group wikis together and support personal/shared visibility.
Uses async SQLAlchemy sessions passed in from the caller (no internal session factory).
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import ProjectRecord, ProjectWikiRecord, WikiRecord

logger = logging.getLogger(__name__)


class ProjectService:
    """Handles project creation, retrieval, update, deletion, and wiki membership."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_project(
        self,
        owner_id: str,
        name: str,
        description: str | None,
        visibility: str,
    ) -> ProjectRecord:
        """Create a new project owned by *owner_id*.

        Returns the newly created ProjectRecord (not yet committed when called
        outside a transaction — callers should manage their own commit).
        """
        project = ProjectRecord(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            name=name,
            description=description,
            visibility=visibility,
        )
        self.session.add(project)
        await self.session.flush()
        await self.session.refresh(project)
        logger.info("Created project %s (owner=%s)", project.id, owner_id)
        return project

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_project(self, project_id: str, user_id: str) -> ProjectRecord | None:
        """Return a project visible to *user_id*, or None if inaccessible.

        Access rules:
        - Owner always has access.
        - Any user can access projects with visibility == "shared".
        - "personal" projects are only visible to the owner.
        """
        result = await self.session.execute(
            select(ProjectRecord).where(ProjectRecord.id == project_id)
        )
        project = result.scalar_one_or_none()
        if project is None:
            return None
        if project.owner_id == user_id:
            return project
        if project.visibility == "shared":
            return project
        return None

    async def list_projects(self, user_id: str) -> list[ProjectRecord]:
        """Return all projects accessible to *user_id*: owned + shared."""
        result = await self.session.execute(
            select(ProjectRecord).where(
                (ProjectRecord.owner_id == user_id) | (ProjectRecord.visibility == "shared")
            )
        )
        return list(result.scalars().all())

    async def get_wiki_count(self, project_id: str) -> int:
        """Return the number of wikis in a project."""
        result = await self.session.execute(
            select(func.count()).where(ProjectWikiRecord.project_id == project_id).select_from(ProjectWikiRecord)
        )
        return result.scalar_one()

    async def batch_get_wiki_counts(self, project_ids: list[str]) -> dict[str, int]:
        """Return a mapping of project_id -> wiki count for all given project IDs.

        Uses a single GROUP BY query instead of one query per project.
        """
        if not project_ids:
            return {}
        result = await self.session.execute(
            select(ProjectWikiRecord.project_id, func.count().label("cnt"))
            .where(ProjectWikiRecord.project_id.in_(project_ids))
            .group_by(ProjectWikiRecord.project_id)
        )
        counts = {row.project_id: row.cnt for row in result}
        # Fill in zero for projects with no wikis
        return {pid: counts.get(pid, 0) for pid in project_ids}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_project(
        self,
        project_id: str,
        owner_id: str,
        **fields: object,
    ) -> ProjectRecord | None:
        """Update project fields.  Only the owner may update.

        Returns the updated ProjectRecord, or None if not found / not owner.
        """
        result = await self.session.execute(
            select(ProjectRecord).where(ProjectRecord.id == project_id)
        )
        project = result.scalar_one_or_none()
        if project is None or project.owner_id != owner_id:
            return None
        for key, value in fields.items():
            if value is not None and hasattr(project, key):
                setattr(project, key, value)
        await self.session.flush()
        await self.session.refresh(project)
        logger.info("Updated project %s", project_id)
        return project

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_project(self, project_id: str, owner_id: str) -> bool:
        """Delete a project and all its ProjectWikiRecord rows.

        Returns False if the project doesn't exist or the caller is not the owner.
        Note: wikis themselves are NOT deleted — only the membership records.
        """
        result = await self.session.execute(
            select(ProjectRecord).where(ProjectRecord.id == project_id)
        )
        project = result.scalar_one_or_none()
        if project is None or project.owner_id != owner_id:
            return False
        # Remove all wiki memberships first
        await self.session.execute(
            delete(ProjectWikiRecord).where(ProjectWikiRecord.project_id == project_id)
        )
        await self.session.delete(project)
        await self.session.flush()
        logger.info("Deleted project %s", project_id)
        return True

    # ------------------------------------------------------------------
    # Wiki membership
    # ------------------------------------------------------------------

    async def add_wiki(
        self,
        project_id: str,
        wiki_id: str,
        owner_id: str,
        added_by: str | None,
    ) -> ProjectWikiRecord | None:
        """Add a wiki to a project.

        Returns the ProjectWikiRecord (new or existing — idempotent).
        Returns None if the caller is not the project owner.
        """
        result = await self.session.execute(
            select(ProjectRecord).where(ProjectRecord.id == project_id)
        )
        project = result.scalar_one_or_none()
        if project is None or project.owner_id != owner_id:
            return None

        # Idempotency check
        existing = await self.session.execute(
            select(ProjectWikiRecord).where(
                ProjectWikiRecord.project_id == project_id,
                ProjectWikiRecord.wiki_id == wiki_id,
            )
        )
        existing_record = existing.scalar_one_or_none()
        if existing_record is not None:
            return existing_record  # already present — idempotent return

        membership = ProjectWikiRecord(
            project_id=project_id,
            wiki_id=wiki_id,
            added_by=added_by,
        )
        self.session.add(membership)
        await self.session.flush()
        await self.session.refresh(membership)
        logger.info("Added wiki %s to project %s", wiki_id, project_id)
        return membership

    async def remove_wiki(self, project_id: str, wiki_id: str, owner_id: str) -> bool:
        """Remove a wiki from a project.

        Returns False if the project doesn't exist or the caller is not the owner.
        """
        result = await self.session.execute(
            select(ProjectRecord).where(ProjectRecord.id == project_id)
        )
        project = result.scalar_one_or_none()
        if project is None or project.owner_id != owner_id:
            return False
        await self.session.execute(
            delete(ProjectWikiRecord).where(
                ProjectWikiRecord.project_id == project_id,
                ProjectWikiRecord.wiki_id == wiki_id,
            )
        )
        await self.session.flush()
        logger.info("Removed wiki %s from project %s", wiki_id, project_id)
        return True

    async def list_project_wikis(self, project_id: str, user_id: str) -> list[WikiRecord] | None:
        """Return all WikiRecords in a project, or None if not accessible to *user_id*."""
        project = await self.get_project(project_id, user_id)
        if project is None:
            return None

        result = await self.session.execute(
            select(WikiRecord)
            .join(ProjectWikiRecord, WikiRecord.id == ProjectWikiRecord.wiki_id)
            .where(ProjectWikiRecord.project_id == project_id)
        )
        return list(result.scalars().all())
