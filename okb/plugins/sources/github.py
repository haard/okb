"""GitHub API source for syncing repository content."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from github import Github
    from github.Repository import Repository

    from okb.ingest import Document
    from okb.plugins.base import SyncState

# Extensions that can be ingested (matches config.py defaults)
INGESTABLE_EXTENSIONS = frozenset(
    [
        # Documents
        ".md",
        ".txt",
        ".markdown",
        ".org",
        # Code
        ".py",
        ".rb",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".sql",
        ".sh",
        ".bash",
        ".fish",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".html",
        ".css",
        ".scss",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".cpp",
        ".h",
    ]
)

# Priority label mapping (label -> priority 1-5, 1=highest)
PRIORITY_LABELS = {
    # Explicit priority labels
    "priority:critical": 1,
    "priority:high": 2,
    "priority:medium": 3,
    "priority:low": 4,
    "p0": 1,
    "p1": 2,
    "p2": 3,
    "p3": 4,
    "p4": 5,
    # Severity/type labels
    "critical": 1,
    "urgent": 1,
    "bug": 2,
    "security": 2,
    "enhancement": 4,
    "feature": 4,
    "documentation": 5,
    "question": 5,
}


def _get_priority_from_labels(labels: list) -> int | None:
    """Extract priority from GitHub labels."""
    for label in labels:
        label_name = label.name.lower()
        if label_name in PRIORITY_LABELS:
            return PRIORITY_LABELS[label_name]
    return None


class GitHubSource:
    """API source for GitHub repository content.

    Syncs repository files, issues, PRs, and wiki pages.

    Config example:
        plugins:
          sources:
            github:
              enabled: true
              token: ${GITHUB_TOKEN}

    Usage:
        lkb sync run github --repo owner/repo              # README + docs/ (default)
        lkb sync run github --repo owner/repo --source     # All source files
        lkb sync run github --repo owner/repo --issues     # Include issues
        lkb sync run github --repo owner/repo --prs        # Include PRs
        lkb sync run github --repo owner/repo --wiki       # Include wiki
    """

    name = "github"
    source_type = "github-source"

    def __init__(self) -> None:
        self._client: Github | None = None
        self._token: str | None = None
        self._repos: list[str] = []
        self._include_source: bool = False
        self._include_issues: bool = False
        self._include_prs: bool = False
        self._include_wiki: bool = False

    def configure(self, config: dict) -> None:
        """Initialize GitHub client with token.

        Args:
            config: Source configuration containing 'token' and CLI options
        """
        from github import Github

        token = config.get("token")
        if not token:
            raise ValueError("github source requires 'token' in config")

        repos = config.get("repos", [])
        if not repos:
            raise ValueError("github source requires --repo flag")

        self._client = Github(token)
        self._token = token
        self._repos = repos
        self._include_source = config.get("include_source", False)
        self._include_issues = config.get("include_issues", False)
        self._include_prs = config.get("include_prs", False)
        self._include_wiki = config.get("include_wiki", False)

    def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
        """Fetch content from GitHub repositories.

        Args:
            state: Previous sync state for incremental updates

        Returns:
            Tuple of (list of documents, new sync state)
        """
        from okb.plugins.base import SyncState as SyncStateClass

        if self._client is None:
            raise RuntimeError("Source not configured. Call configure() first.")

        documents: list[Document] = []
        extra = state.extra if state else {}
        last_sync = state.last_sync if state else None

        for repo_name in self._repos:
            print(f"Syncing {repo_name}...", file=sys.stderr)
            try:
                repo = self._client.get_repo(repo_name)
                repo_extra = extra.get(repo_name, {})

                # Sync source files (default: README + docs/, or all with --source)
                source_docs, new_sha = self._sync_source_files(repo, repo_extra.get("commit_sha"))
                documents.extend(source_docs)
                extra[repo_name] = {"commit_sha": new_sha}

                # Sync issues if requested
                if self._include_issues:
                    issue_docs = self._sync_issues(repo, last_sync)
                    documents.extend(issue_docs)

                # Sync PRs if requested
                if self._include_prs:
                    pr_docs = self._sync_prs(repo, last_sync)
                    documents.extend(pr_docs)

                # Sync wiki if requested
                if self._include_wiki:
                    wiki_docs = self._sync_wiki(repo)
                    documents.extend(wiki_docs)

            except Exception as e:
                print(f"  Error syncing {repo_name}: {e}", file=sys.stderr)

        new_state = SyncStateClass(
            last_sync=datetime.now(UTC),
            extra=extra,
        )

        return documents, new_state

    def _sync_source_files(
        self, repo: Repository, last_sha: str | None
    ) -> tuple[list[Document], str]:
        """Sync source files from the repository.

        Args:
            repo: GitHub repository object
            last_sha: Last synced commit SHA

        Returns:
            Tuple of (documents, current commit SHA)
        """
        from okb.ingest import Document, DocumentMetadata

        documents: list[Document] = []

        # Get current HEAD commit
        default_branch = repo.default_branch
        current_sha = repo.get_branch(default_branch).commit.sha

        # Skip if no changes
        if last_sha == current_sha:
            print(f"  Source files unchanged (SHA: {current_sha[:8]})", file=sys.stderr)
            return documents, current_sha

        print(f"  Fetching source files (SHA: {current_sha[:8]})...", file=sys.stderr)

        # Get repository tree
        tree = repo.get_git_tree(current_sha, recursive=True)

        for item in tree.tree:
            if item.type != "blob":
                continue

            path = item.path

            # Check if file should be included
            if not self._should_include_file(path):
                continue

            # Check extension
            ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
            if ext.lower() not in INGESTABLE_EXTENSIONS:
                continue

            try:
                content = self._get_file_content(repo, item.sha)
                if content is None:
                    continue

                title = path.split("/")[-1]
                doc = Document(
                    source_path=f"github://{repo.full_name}/blob/{default_branch}/{path}",
                    source_type="github-source",
                    title=title,
                    content=content,
                    metadata=DocumentMetadata(
                        project=repo.name,
                        extra={
                            "repo": repo.full_name,
                            "path": path,
                            "sha": item.sha,
                        },
                    ),
                )
                documents.append(doc)
                print(f"    Synced: {path}", file=sys.stderr)
            except Exception as e:
                print(f"    Error fetching {path}: {e}", file=sys.stderr)

        return documents, current_sha

    def _should_include_file(self, path: str) -> bool:
        """Check if a file should be included based on sync options."""
        if self._include_source:
            # Include all files
            return True

        # Default: README* at root + docs/**/*
        path_lower = path.lower()

        # README files at root
        if "/" not in path and path_lower.startswith("readme"):
            return True

        # Files in docs/ directory
        if path_lower.startswith("docs/"):
            return True

        return False

    def _get_file_content(self, repo: Repository, sha: str) -> str | None:
        """Get file content by blob SHA."""
        import base64

        blob = repo.get_git_blob(sha)
        if blob.encoding == "base64":
            try:
                return base64.b64decode(blob.content).decode("utf-8")
            except UnicodeDecodeError:
                return None  # Binary file
        return blob.content

    def _sync_issues(self, repo: Repository, since: datetime | None) -> list[Document]:
        """Sync GitHub issues."""
        from okb.ingest import Document, DocumentMetadata

        documents: list[Document] = []
        print("  Fetching issues...", file=sys.stderr)

        # Fetch issues updated since last sync
        kwargs = {"state": "all", "sort": "updated", "direction": "desc"}
        if since:
            kwargs["since"] = since

        issues = repo.get_issues(**kwargs)
        count = 0

        for issue in issues:
            # Skip pull requests (they show up in issues API)
            if issue.pull_request is not None:
                continue

            # Build content (title + body, no comments per user request)
            content = f"# {issue.title}\n\n{issue.body or ''}"

            # Extract labels
            labels = [label.name for label in issue.labels]

            doc = Document(
                source_path=f"github://{repo.full_name}/issues/{issue.number}",
                source_type="github-issue",
                title=f"#{issue.number}: {issue.title}",
                content=content,
                status="open" if issue.state == "open" else "closed",
                priority=_get_priority_from_labels(issue.labels),
                metadata=DocumentMetadata(
                    project=repo.name,
                    tags=labels,
                    extra={
                        "repo": repo.full_name,
                        "number": issue.number,
                        "author": issue.user.login if issue.user else None,
                        "created_at": issue.created_at.isoformat() if issue.created_at else None,
                        "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                        "url": issue.html_url,
                    },
                ),
            )
            documents.append(doc)
            count += 1

        print(f"    Synced {count} issues", file=sys.stderr)
        return documents

    def _sync_prs(self, repo: Repository, since: datetime | None) -> list[Document]:
        """Sync GitHub pull requests."""
        from okb.ingest import Document, DocumentMetadata

        documents: list[Document] = []
        print("  Fetching pull requests...", file=sys.stderr)

        # Fetch PRs - unfortunately the PR API doesn't have a 'since' parameter
        # We'll filter by updated_at manually
        pulls = repo.get_pulls(state="all", sort="updated", direction="desc")
        count = 0

        for pr in pulls:
            # Skip if older than last sync
            if since and pr.updated_at and pr.updated_at < since:
                break  # PRs are sorted by updated, so we can stop here

            # Build content (title + body, no comments per user request)
            content = f"# {pr.title}\n\n{pr.body or ''}"

            # Extract labels
            labels = [label.name for label in pr.labels]

            # Determine status
            if pr.merged:
                status = "merged"
            elif pr.state == "open":
                status = "open"
            else:
                status = "closed"

            doc = Document(
                source_path=f"github://{repo.full_name}/pull/{pr.number}",
                source_type="github-pr",
                title=f"PR #{pr.number}: {pr.title}",
                content=content,
                status=status,
                priority=_get_priority_from_labels(pr.labels),
                metadata=DocumentMetadata(
                    project=repo.name,
                    tags=labels,
                    extra={
                        "repo": repo.full_name,
                        "number": pr.number,
                        "author": pr.user.login if pr.user else None,
                        "created_at": pr.created_at.isoformat() if pr.created_at else None,
                        "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
                        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                        "url": pr.html_url,
                        "base": pr.base.ref,
                        "head": pr.head.ref,
                    },
                ),
            )
            documents.append(doc)
            count += 1

        print(f"    Synced {count} pull requests", file=sys.stderr)
        return documents

    def _sync_wiki(self, repo: Repository) -> list[Document]:
        """Sync GitHub wiki pages.

        Note: GitHub doesn't have an API for wiki content.
        We clone the wiki repo and read files directly.
        """
        from okb.ingest import Document, DocumentMetadata

        documents: list[Document] = []

        # Check if wiki exists
        if not repo.has_wiki:
            print("  Wiki not enabled for this repo", file=sys.stderr)
            return documents

        print("  Fetching wiki pages...", file=sys.stderr)

        # Wiki is a separate git repo at {repo}.wiki.git
        # We need to use git to clone it
        import tempfile
        from pathlib import Path
        from subprocess import CalledProcessError, run

        # Use token in URL for authentication
        wiki_url = f"https://{self._token}@github.com/{repo.full_name}.wiki.git"

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Clone wiki repo (shallow)
                result = run(
                    ["git", "clone", "--depth", "1", wiki_url, tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    # Wiki might not exist even if has_wiki is True
                    print("  Wiki repository not accessible", file=sys.stderr)
                    return documents

                # Find all markdown files
                wiki_path = Path(tmpdir)
                count = 0

                for md_file in wiki_path.glob("*.md"):
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        title = md_file.stem.replace("-", " ")

                        doc = Document(
                            source_path=f"github://{repo.full_name}/wiki/{md_file.stem}",
                            source_type="github-wiki",
                            title=title,
                            content=content,
                            metadata=DocumentMetadata(
                                project=repo.name,
                                extra={
                                    "repo": repo.full_name,
                                    "page": md_file.stem,
                                },
                            ),
                        )
                        documents.append(doc)
                        count += 1
                    except Exception as e:
                        print(f"    Error reading {md_file.name}: {e}", file=sys.stderr)

                print(f"    Synced {count} wiki pages", file=sys.stderr)

            except CalledProcessError as e:
                print(f"    Error cloning wiki: {e}", file=sys.stderr)
            except Exception as e:
                print(f"    Error syncing wiki: {e}", file=sys.stderr)

        return documents
