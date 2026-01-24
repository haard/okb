"""LLM integration for document classification and enrichment.

This package provides a provider-agnostic interface for LLM operations,
with support for Claude API, AWS Bedrock, and response caching.

Usage:
    from okb.llm import get_llm, complete

    # Get configured provider (returns None if disabled)
    llm = get_llm()
    if llm:
        response = llm.complete("Summarize this document", system="Be concise")

    # Or use convenience function with caching
    response = complete("Classify this email")
"""

from .base import LLMProvider, LLMResponse
from .filter import FilterAction, FilterResult, filter_document, filter_documents

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "FilterAction",
    "FilterResult",
    "filter_document",
    "filter_documents",
    "get_llm",
    "complete",
]


def get_llm() -> LLMProvider | None:
    """Get the configured LLM provider, or None if disabled.

    Reads configuration from the global config object.
    Lazily initializes the provider on first call.

    Returns:
        Configured LLMProvider instance, or None if llm.provider is not set
    """
    from .providers import get_provider

    return get_provider()


def complete(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 1024,
    use_cache: bool = True,
) -> LLMResponse | None:
    """Generate a completion using the configured LLM provider.

    Convenience function that handles caching and provider initialization.

    Args:
        prompt: The user prompt to complete
        system: Optional system prompt for context/instructions
        max_tokens: Maximum tokens in the response
        use_cache: Whether to use cached responses (default True)

    Returns:
        LLMResponse with the generated content, or None if LLM is disabled
    """
    from .cache import cache_response, get_cached
    from .providers import get_provider

    provider = get_provider()
    if provider is None:
        return None

    # Check cache first
    if use_cache:
        cached = get_cached(prompt, system, provider.name)
        if cached is not None:
            return cached

    # Generate new response
    response = provider.complete(prompt, system=system, max_tokens=max_tokens)

    # Cache the response
    if use_cache:
        cache_response(prompt, system, provider.name, response)

    return response
