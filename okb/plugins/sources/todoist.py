"""Todoist API source for syncing tasks into OKB."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from okb.ingest import Document
    from okb.plugins.base import SyncState


class TodoistSource:
    """API source for Todoist tasks.

    Syncs active and optionally completed tasks for semantic search and actionable item queries.

    Config example:
        plugins:
          sources:
            todoist:
              enabled: true
              token: ${TODOIST_TOKEN}
              include_completed: false   # Include recently completed tasks
              completed_days: 30         # Days of completed tasks to sync
              include_comments: false    # Include task comments (expensive)
              project_filter: []         # Optional: list of project IDs to sync

    Usage:
        okb sync run todoist
        okb sync run todoist --full  # Full resync
    """

    name = "todoist"
    source_type = "todoist-task"

    def __init__(self) -> None:
        self._client = None
        self._include_completed = False
        self._completed_days = 30
        self._include_comments = False
        self._project_filter: list[str] | None = None
        self._projects: dict[str, str] = {}  # id -> name

    def configure(self, config: dict) -> None:
        """Initialize Todoist client with API token.

        Args:
            config: Source configuration containing 'token' and optional settings
        """
        from todoist_api_python.api import TodoistAPI

        token = config.get("token")
        if not token:
            raise ValueError("todoist source requires 'token' in config")

        self._client = TodoistAPI(token)
        self._include_completed = config.get("include_completed", False)
        self._completed_days = config.get("completed_days", 30)
        self._include_comments = config.get("include_comments", False)
        self._project_filter = config.get("project_filter")

    def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
        """Fetch tasks from Todoist.

        Active tasks are always fully synced (API has no "modified since" filter).
        Completed tasks use state.last_sync for incremental fetching.

        Args:
            state: Previous sync state for incremental updates, or None for full sync

        Returns:
            Tuple of (list of documents, new sync state)
        """
        from okb.plugins.base import SyncState as SyncStateClass

        if self._client is None:
            raise RuntimeError("Source not configured. Call configure() first.")

        documents: list[Document] = []

        print("Fetching Todoist tasks...", file=sys.stderr)

        # Load projects for name lookup
        self._load_projects()

        # Fetch active tasks
        active_docs = self._fetch_active_tasks()
        documents.extend(active_docs)
        print(f"  Synced {len(active_docs)} active tasks", file=sys.stderr)

        # Fetch completed tasks if enabled
        if self._include_completed:
            since = state.last_sync if state and state.last_sync else None
            completed_docs = self._fetch_completed_tasks(since)
            documents.extend(completed_docs)
            print(f"  Synced {len(completed_docs)} completed tasks", file=sys.stderr)

        new_state = SyncStateClass(last_sync=datetime.now(UTC))
        return documents, new_state

    def _load_projects(self) -> None:
        """Load projects for name lookup."""
        try:
            self._projects = {}
            for project_batch in self._client.get_projects():
                for p in project_batch:
                    self._projects[p.id] = p.name
        except Exception as e:
            print(f"  Warning: Could not load projects: {e}", file=sys.stderr)
            self._projects = {}

    def _fetch_active_tasks(self) -> list[Document]:
        """Fetch all active tasks."""
        documents = []

        for task_batch in self._client.get_tasks():
            for task in task_batch:
                # Apply project filter if configured
                if self._project_filter and task.project_id not in self._project_filter:
                    continue

                doc = self._task_to_document(task, is_completed=False)
                if doc:
                    documents.append(doc)

        return documents

    def _fetch_completed_tasks(self, since: datetime | None) -> list[Document]:
        """Fetch completed tasks within the configured window."""
        documents = []

        # Determine date range
        until = datetime.now(UTC)
        if since:
            start = since
        else:
            start = until - timedelta(days=self._completed_days)

        try:
            for task_batch in self._client.get_completed_tasks_by_completion_date(
                since=start,
                until=until,
            ):
                for task in task_batch:
                    # Apply project filter if configured
                    if self._project_filter and task.project_id not in self._project_filter:
                        continue

                    doc = self._task_to_document(task, is_completed=True)
                    if doc:
                        documents.append(doc)
        except Exception as e:
            print(f"  Warning: Could not fetch completed tasks: {e}", file=sys.stderr)

        return documents

    def _task_to_document(self, task, is_completed: bool) -> Document | None:
        """Convert a Todoist task to a Document."""
        from okb.ingest import Document, DocumentMetadata

        # Build content from task content + description + optional comments
        content_parts = [task.content]
        if task.description:
            content_parts.append(task.description)

        if self._include_comments:
            comments = self._fetch_task_comments(task.id)
            if comments:
                content_parts.append("\n## Comments\n" + "\n".join(comments))

        content = "\n\n".join(content_parts)

        # Parse due date
        due_date = None
        if task.due:
            due_date = self._parse_due(task.due)

        # Map priority: Todoist uses 1-4 (4=urgent), OKB uses 1-5 (1=highest)
        priority = 5 - task.priority if task.priority else None

        # Get project name
        project_name = self._projects.get(task.project_id)

        # Build metadata
        metadata = DocumentMetadata(
            tags=task.labels or [],
            project=project_name,
            extra={
                "todoist_id": task.id,
                "project_id": task.project_id,
            },
        )

        # Determine status
        status = "completed" if is_completed or task.is_completed else "pending"

        return Document(
            source_path=f"todoist://task/{task.id}",
            source_type=self.source_type,
            title=task.content,
            content=content,
            metadata=metadata,
            due_date=due_date,
            status=status,
            priority=priority,
        )

    def _parse_due(self, due) -> datetime | None:
        """Parse Todoist Due object to datetime."""
        if due is None:
            return None

        try:
            # Due has 'datetime' (full datetime) or 'date' (date only)
            if hasattr(due, "datetime") and due.datetime:
                return datetime.fromisoformat(due.datetime.replace("Z", "+00:00"))
            elif hasattr(due, "date") and due.date:
                # Date-only: treat as end of day in UTC
                if isinstance(due.date, str):
                    d = datetime.strptime(due.date, "%Y-%m-%d").date()
                else:
                    d = due.date
                return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=UTC)
        except Exception:
            pass
        return None

    def _fetch_task_comments(self, task_id: str) -> list[str]:
        """Fetch comments for a task."""
        comments = []
        try:
            for comment_batch in self._client.get_comments(task_id=task_id):
                for comment in comment_batch:
                    comments.append(f"- {comment.content}")
        except Exception:
            pass
        return comments

    def list_projects(self) -> list[tuple[str, str]]:
        """List all projects with their IDs.

        Returns:
            List of (project_id, project_name) tuples
        """
        if self._client is None:
            raise RuntimeError("Source not configured. Call configure() first.")

        projects = []
        for project_batch in self._client.get_projects():
            for p in project_batch:
                projects.append((p.id, p.name))
        return projects
