"""LLM-based document filtering for pre-ingest classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ingest import Document


class FilterAction(Enum):
    """Actions the filter can take on a document."""

    INGEST = "ingest"  # Process normally
    SKIP = "skip"  # Don't ingest
    REVIEW = "review"  # Flag for manual review (still ingest)


@dataclass
class FilterResult:
    """Result of LLM filtering on a document."""

    action: FilterAction
    reason: str
    confidence: float | None = None  # Optional confidence score 0-1

    @property
    def should_ingest(self) -> bool:
        """Whether the document should be ingested."""
        return self.action in (FilterAction.INGEST, FilterAction.REVIEW)


DEFAULT_SYSTEM_PROMPT = """\
You are a document classifier. Analyze the document and decide whether it should
be ingested into a knowledge base.

Respond with a JSON object containing:
- "action": one of "ingest", "skip", or "review"
- "reason": brief explanation (1 sentence)

Use these guidelines:
- "ingest": valuable content worth indexing (notes, docs, important emails, etc.)
- "skip": low-value content (spam, marketing, automated notifications, duplicates)
- "review": uncertain cases that need human review

Respond ONLY with the JSON object, no other text."""


def _build_filter_prompt(document: Document, custom_prompt: str | None = None) -> str:
    """Build the prompt for document filtering.

    Args:
        document: Document to classify
        custom_prompt: Optional custom instructions to append

    Returns:
        Formatted prompt string
    """
    parts = [
        f"Title: {document.title}",
        f"Source: {document.source_type}",
    ]

    if document.metadata and document.metadata.tags:
        parts.append(f"Tags: {', '.join(document.metadata.tags)}")

    # Include content preview (truncate if too long)
    content_preview = document.content[:2000]
    if len(document.content) > 2000:
        content_preview += "\n[... truncated ...]"

    parts.append(f"\nContent:\n{content_preview}")

    if custom_prompt:
        parts.append(f"\nAdditional instructions: {custom_prompt}")

    return "\n".join(parts)


def _parse_filter_response(response: str) -> FilterResult:
    """Parse the LLM response into a FilterResult.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed FilterResult

    Raises:
        ValueError: If response cannot be parsed
    """
    # Try to extract JSON from response
    # Handle cases where LLM wraps in markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback: assume entire response is JSON
            json_str = response.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse filter response as JSON: {e}")

    action_str = data.get("action", "").lower()
    try:
        action = FilterAction(action_str)
    except ValueError:
        # Default to ingest if action is invalid
        action = FilterAction.INGEST

    reason = data.get("reason", "No reason provided")
    confidence = data.get("confidence")

    return FilterResult(action=action, reason=reason, confidence=confidence)


def filter_document(
    document: Document,
    custom_prompt: str | None = None,
    use_cache: bool = True,
) -> FilterResult:
    """Filter a single document using the configured LLM.

    Args:
        document: Document to filter
        custom_prompt: Optional custom classification instructions
        use_cache: Whether to use cached responses

    Returns:
        FilterResult with action and reason

    Raises:
        RuntimeError: If LLM is not configured
    """
    from . import complete

    prompt = _build_filter_prompt(document, custom_prompt)
    response = complete(prompt, system=DEFAULT_SYSTEM_PROMPT, use_cache=use_cache)

    if response is None:
        # LLM not configured - default to ingest
        return FilterResult(
            action=FilterAction.INGEST,
            reason="LLM not configured, defaulting to ingest",
        )

    try:
        return _parse_filter_response(response.content)
    except ValueError as e:
        # Parse error - default to ingest with warning
        return FilterResult(
            action=FilterAction.INGEST,
            reason=f"Failed to parse LLM response: {e}",
        )


def filter_documents(
    documents: list[Document],
    custom_prompt: str | None = None,
    use_cache: bool = True,
) -> list[tuple[Document, FilterResult]]:
    """Filter multiple documents.

    Args:
        documents: List of documents to filter
        custom_prompt: Optional custom classification instructions
        use_cache: Whether to use cached responses

    Returns:
        List of (document, filter_result) tuples
    """
    results = []
    for doc in documents:
        result = filter_document(doc, custom_prompt=custom_prompt, use_cache=use_cache)
        results.append((doc, result))
    return results
