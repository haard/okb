"""TODO extraction from document content using LLM."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from .base import ExtractedTodo

TODO_SYSTEM_PROMPT = """\
You are an expert at identifying action items and tasks in text.
Extract TODO items from the given document content.

Look for:
- Explicit markers: TODO, FIXME, HACK, XXX, ACTION
- Action phrases: "need to", "should", "must", "have to", "action item"
- Deadlines and commitments: "by Friday", "before the meeting", "this week"
- Questions implying needed work: "What about X?", "How do we handle Y?"
- Incomplete items marked for follow-up

For each TODO found, extract:
- title: A concise description of the task (imperative form: "Fix the bug", not "The bug needs fixing")
- content: Additional context or details (optional)
- due_date: If a deadline is mentioned, in ISO format YYYY-MM-DD (optional)
- priority: 1=urgent, 2=high, 3=normal, 4=low, 5=someday (optional)
- assignee: Person responsible if mentioned (optional)
- source_context: The exact text snippet where this TODO was found

Return JSON array of extracted TODOs. Return empty array [] if none found.
Be conservative - only extract clear action items, not vague mentions.
"""

TODO_USER_PROMPT = """\
Document title: {title}
Source type: {source_type}

Content:
{content}

Extract all TODO items from this content as JSON array.
"""


def extract_todos(
    content: str,
    title: str,
    source_type: str,
    min_confidence: float = 0.7,
) -> list[ExtractedTodo]:
    """Extract TODO items from document content using LLM.

    Args:
        content: Document content to analyze
        title: Document title for context
        source_type: Type of document (markdown, code, org, etc.)
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        List of extracted TODO items
    """
    from .. import complete

    # Truncate content if too long (keep first ~20k chars for context)
    if len(content) > 20000:
        content = content[:20000] + "\n\n[... content truncated ...]"

    prompt = TODO_USER_PROMPT.format(
        title=title,
        source_type=source_type,
        content=content,
    )

    response = complete(
        prompt=prompt,
        system=TODO_SYSTEM_PROMPT,
        max_tokens=2048,
        use_cache=True,
    )

    if response is None:
        return []

    return _parse_todo_response(response.content, min_confidence)


def _parse_todo_response(response_text: str, min_confidence: float) -> list[ExtractedTodo]:
    """Parse LLM response into ExtractedTodo objects."""
    # Try to extract JSON from response
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not json_match:
        return []

    try:
        todos_data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(todos_data, list):
        return []

    todos = []
    for item in todos_data:
        if not isinstance(item, dict):
            continue

        title = item.get("title")
        if not title or not isinstance(title, str):
            continue

        # Parse due_date if present
        due_date = None
        if due_str := item.get("due_date"):
            try:
                due_date = datetime.fromisoformat(due_str).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                pass

        # Parse priority
        priority = None
        if p := item.get("priority"):
            try:
                priority = int(p)
                if priority < 1 or priority > 5:
                    priority = None
            except (ValueError, TypeError):
                pass

        # Get confidence (default to 0.8 if not specified)
        confidence = item.get("confidence", 0.8)
        if not isinstance(confidence, (int, float)):
            confidence = 0.8

        if confidence < min_confidence:
            continue

        todos.append(
            ExtractedTodo(
                title=title.strip(),
                content=item.get("content"),
                due_date=due_date,
                priority=priority,
                assignee=item.get("assignee"),
                confidence=float(confidence),
                source_context=item.get("source_context"),
            )
        )

    return todos
