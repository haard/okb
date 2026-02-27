"""Slack API source for syncing channel messages into OKB."""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from okb.ingest import Document
    from okb.plugins.base import SyncState


class SlackSource:
    """API source for Slack workspace messages.

    Syncs messages from whitelisted channels/DMs. Threads become one document each;
    non-threaded messages are grouped into daily digests per channel.

    Config example:
        plugins:
          sources:
            slack:
              enabled: true
              token: ${SLACK_USER_TOKEN}
              channels:
                - C12345
                - D67890
              include_threads: true
              lookback_days: 90
              workspace_name: slack

    Usage:
        okb sync run slack
        okb sync run slack --channel C12345
        okb sync run slack --full
    """

    name = "slack"
    source_type = "slack-message"

    def __init__(self) -> None:
        self._token: str | None = None
        self._channels: list[str] = []
        self._include_threads: bool = True
        self._lookback_days: int = 90
        self._workspace_name: str = "slack"
        self._team_id: str | None = None
        self._headers: dict[str, str] = {}
        self._users: dict[str, str] = {}  # user_id -> display_name
        self._channel_names: dict[str, str] = {}  # channel_id -> name

    def configure(self, config: dict) -> None:
        """Initialize with Slack user token and channel list.

        Args:
            config: Source configuration containing 'token' and 'channels'
        """
        token = config.get("token")
        if not token:
            raise ValueError("slack source requires 'token' in config")

        channels = config.get("channels", [])
        if not channels:
            raise ValueError(
                "slack source requires 'channels' in config (list of channel/DM IDs)"
            )

        self._token = token
        self._channels = list(channels)
        self._include_threads = config.get("include_threads", True)
        self._lookback_days = config.get("lookback_days", 90)
        self._workspace_name = config.get("workspace_name", "slack")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
        """Fetch messages from configured Slack channels.

        Args:
            state: Previous sync state for incremental updates

        Returns:
            Tuple of (list of documents, new sync state)
        """
        from okb.plugins.base import SyncState as SyncStateClass

        if self._token is None:
            raise RuntimeError("Source not configured. Call configure() first.")

        documents: list[Document] = []
        extra = dict(state.extra) if state else {}

        # Load users for display name resolution
        self._load_users()

        for channel_id in self._channels:
            print(f"Syncing channel {channel_id}...", file=sys.stderr)

            # Resolve channel name
            self._resolve_channel_name(channel_id)
            channel_name = self._channel_names.get(channel_id, channel_id)

            # Determine oldest timestamp to fetch from
            channel_state = extra.get(channel_id, {})
            if channel_state and channel_state.get("latest_ts"):
                oldest_ts = channel_state["latest_ts"]
            else:
                oldest_dt = datetime.now(UTC) - timedelta(days=self._lookback_days)
                oldest_ts = str(oldest_dt.timestamp())

            try:
                channel_docs, highest_ts = self._sync_channel(
                    channel_id, channel_name, oldest_ts
                )
                documents.extend(channel_docs)

                # Update state with highest ts seen
                if highest_ts:
                    extra[channel_id] = {"latest_ts": highest_ts}
                elif channel_state:
                    extra[channel_id] = channel_state  # preserve existing

            except Exception as e:
                print(f"  Error syncing {channel_id}: {e}", file=sys.stderr)

        new_state = SyncStateClass(
            last_sync=datetime.now(UTC),
            extra=extra,
        )
        return documents, new_state

    def _sync_channel(
        self, channel_id: str, channel_name: str, oldest_ts: str
    ) -> tuple[list[Document], str | None]:
        """Sync a single channel, returning documents and highest ts seen."""
        documents: list[Document] = []
        highest_ts: str | None = None

        # Fetch message history
        messages = self._fetch_history(channel_id, oldest_ts)

        if not messages:
            print(f"  No new messages in #{channel_name}", file=sys.stderr)
            return documents, highest_ts

        # Track highest ts
        for msg in messages:
            ts = msg.get("ts", "")
            if highest_ts is None or ts > highest_ts:
                highest_ts = ts

        # Separate thread roots from standalone messages
        thread_roots = []
        standalone = []

        for msg in messages:
            if msg.get("reply_count", 0) > 0 or msg.get("thread_ts") == msg.get("ts"):
                # This is a thread root (has replies)
                if msg.get("reply_count", 0) > 0:
                    thread_roots.append(msg)
            elif msg.get("thread_ts"):
                # This is a reply in a thread - skip, we'll get it via replies API
                continue
            else:
                standalone.append(msg)

        # Process threads
        if self._include_threads and thread_roots:
            thread_count = 0
            for root in thread_roots:
                replies = self._fetch_replies(channel_id, root["ts"])
                doc = self._thread_to_document(channel_id, channel_name, root, replies)
                if doc:
                    documents.append(doc)
                    thread_count += 1

                    # Track highest ts from replies too
                    for reply in replies:
                        rts = reply.get("ts", "")
                        if highest_ts is None or rts > highest_ts:
                            highest_ts = rts

            if thread_count:
                print(f"  {thread_count} threads in #{channel_name}", file=sys.stderr)

        # Group standalone messages by date -> daily digests
        if standalone:
            daily_groups: dict[str, list[dict]] = {}
            for msg in standalone:
                ts = float(msg.get("ts", 0))
                date_str = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
                daily_groups.setdefault(date_str, []).append(msg)

            digest_count = 0
            for date_str, day_msgs in sorted(daily_groups.items()):
                doc = self._daily_to_document(channel_id, channel_name, date_str, day_msgs)
                if doc:
                    documents.append(doc)
                    digest_count += 1

            if digest_count:
                print(
                    f"  {digest_count} daily digests ({len(standalone)} messages) "
                    f"in #{channel_name}",
                    file=sys.stderr,
                )

        return documents, highest_ts

    def _load_users(self) -> None:
        """Fetch users.list and cache id->display_name mapping."""
        if self._users:
            return  # Already loaded

        try:
            data = self._api_call("users.list")
            for member in data.get("members", []):
                uid = member["id"]
                profile = member.get("profile", {})
                name = (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or member.get("real_name")
                    or member.get("name")
                    or uid
                )
                self._users[uid] = name

                # Capture team_id from first user if not set
                if self._team_id is None and member.get("team_id"):
                    self._team_id = member["team_id"]

        except Exception as e:
            print(f"  Warning: Could not load users: {e}", file=sys.stderr)

    def _resolve_channel_name(self, channel_id: str) -> None:
        """Try to resolve a channel ID to a name via conversations.info."""
        if channel_id in self._channel_names:
            return

        try:
            data = self._api_call("conversations.info", {"channel": channel_id})
            channel = data.get("channel", {})
            name = channel.get("name")
            if name:
                self._channel_names[channel_id] = name
            else:
                # DMs don't have names - use the user's name if possible
                user_id = channel.get("user")
                if user_id and user_id in self._users:
                    self._channel_names[channel_id] = f"dm-{self._users[user_id]}"
                else:
                    self._channel_names[channel_id] = channel_id
        except Exception:
            self._channel_names[channel_id] = channel_id

    def _fetch_history(self, channel_id: str, oldest: str) -> list[dict]:
        """Fetch paginated conversations.history for a channel."""
        messages = []
        cursor = None

        while True:
            params: dict = {
                "channel": channel_id,
                "oldest": oldest,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._api_call("conversations.history", params)
            batch = data.get("messages", [])
            messages.extend(batch)

            # Check for pagination
            meta = data.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        # API returns newest first, reverse to chronological
        messages.reverse()
        return messages

    def _fetch_replies(self, channel_id: str, thread_ts: str) -> list[dict]:
        """Fetch paginated conversations.replies for a thread."""
        messages = []
        cursor = None

        while True:
            params: dict = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._api_call("conversations.replies", params)
            batch = data.get("messages", [])
            messages.extend(batch)

            meta = data.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        return messages

    def _api_call(self, method: str, params: dict | None = None) -> dict:
        """Make a Slack API call with rate-limit handling.

        Args:
            method: Slack API method name (e.g., 'conversations.history')
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If API returns an error
        """
        url = f"https://slack.com/api/{method}"

        # Conservative delay between calls
        time.sleep(1)

        for attempt in range(3):
            resp = requests.get(url, headers=self._headers, params=params or {}, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(
                    f"  Rate limited, waiting {retry_after}s...",
                    file=sys.stderr,
                )
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok"):
                error = data.get("error", "unknown_error")
                raise RuntimeError(f"Slack API error ({method}): {error}")

            return data

        raise RuntimeError(f"Slack API failed after retries ({method})")

    def _resolve_user(self, user_id: str) -> str:
        """Resolve a user ID to a display name."""
        return self._users.get(user_id, user_id)

    def _format_message(self, msg: dict, include_date: bool = True) -> str:
        """Render a message as "[timestamp] Author: text".

        Args:
            msg: Slack message dict
            include_date: Whether to include full date or just time
        """
        ts = float(msg.get("ts", 0))
        dt = datetime.fromtimestamp(ts, tz=UTC)

        if include_date:
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = dt.strftime("%H:%M")

        user_id = msg.get("user", "")
        author = self._resolve_user(user_id)
        text = msg.get("text", "")

        # Resolve user mentions in text: <@U12345> -> @DisplayName
        import re

        def replace_mention(match):
            uid = match.group(1)
            return f"@{self._resolve_user(uid)}"

        text = re.sub(r"<@(U[A-Z0-9]+)>", replace_mention, text)

        # Clean up Slack link formatting: <url|label> -> label, <url> -> url
        text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2", text)
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)

        return f"[{time_str}] {author}: {text}"

    def _thread_to_document(
        self, channel_id: str, channel_name: str, root: dict, replies: list[dict]
    ) -> Document | None:
        """Convert a thread (root + replies) into a Document."""
        from okb.ingest import Document, DocumentMetadata

        if not replies:
            replies = [root]

        # Build content from all messages in the thread
        lines = []
        participants = set()
        for msg in replies:
            lines.append(self._format_message(msg, include_date=True))
            user_id = msg.get("user", "")
            if user_id:
                participants.add(self._resolve_user(user_id))

        content = "\n".join(lines)
        if not content.strip():
            return None

        thread_ts = root.get("ts", "")
        root_user = self._resolve_user(root.get("user", ""))
        reply_count = len(replies) - 1  # Exclude root message
        team_id = self._team_id or "T0"

        title = f"#{channel_name}: Thread by {root_user} ({reply_count} replies)"

        return Document(
            source_path=f"slack://{team_id}/{channel_id}/thread/{thread_ts}",
            source_type=self.source_type,
            title=title,
            content=content,
            metadata=DocumentMetadata(
                tags=["slack", channel_name],
                project=self._workspace_name,
                extra={
                    "channel_id": channel_id,
                    "thread_ts": thread_ts,
                    "message_count": len(replies),
                    "participants": sorted(participants),
                },
            ),
        )

    def _daily_to_document(
        self, channel_id: str, channel_name: str, date_str: str, messages: list[dict]
    ) -> Document | None:
        """Convert a day's standalone messages into a daily digest Document."""
        from okb.ingest import Document, DocumentMetadata

        # Build content - use time-only since date is in the title
        lines = []
        for msg in messages:
            lines.append(self._format_message(msg, include_date=False))

        content = "\n".join(lines)
        if not content.strip():
            return None

        team_id = self._team_id or "T0"

        # Format date for title
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_display = dt.strftime("%b %d, %Y")
        except ValueError:
            date_display = date_str

        title = f"#{channel_name}: Messages on {date_display} ({len(messages)} messages)"

        return Document(
            source_path=f"slack://{team_id}/{channel_id}/daily/{date_str}",
            source_type=self.source_type,
            title=title,
            content=content,
            metadata=DocumentMetadata(
                tags=["slack", channel_name],
                project=self._workspace_name,
                extra={
                    "channel_id": channel_id,
                    "date": date_str,
                    "message_count": len(messages),
                },
            ),
        )
