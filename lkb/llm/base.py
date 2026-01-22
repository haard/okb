"""Protocol definitions and types for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Standard response from LLM providers."""

    content: str  # Raw text response
    model: str  # Model that generated it
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        """Total tokens used (input + output)."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM backend providers.

    Plugins implement this to add support for different LLM services.

    Example:
        class MyProvider:
            name = 'my-llm'

            def configure(self, config: dict) -> None:
                self._api_key = config.get('api_key')

            def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
                # Call the LLM API
                ...

            def is_available(self) -> bool:
                return self._api_key is not None
    """

    name: str  # Provider identifier, e.g., "claude", "bedrock", "ollama"

    def configure(self, config: dict) -> None:
        """Configure the provider with settings from config.

        Config values may include resolved environment variables.

        Args:
            config: Provider-specific configuration dict
        """
        ...

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt to complete
            system: Optional system prompt for context/instructions
            max_tokens: Maximum tokens in the response

        Returns:
            LLMResponse with the generated content and metadata
        """
        ...

    def is_available(self) -> bool:
        """Check if the provider is configured and reachable.

        Returns:
            True if the provider can accept requests
        """
        ...
