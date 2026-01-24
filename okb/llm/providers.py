"""LLM provider implementations."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from .base import LLMResponse

if TYPE_CHECKING:
    from .base import LLMProvider


class ClaudeProvider:
    """Claude API provider using the Anthropic SDK.

    Supports both direct API access and AWS Bedrock.

    Config:
        model: Model name (default: claude-haiku-4-5-20251001)
        api_key: API key (default: reads ANTHROPIC_API_KEY)
        timeout: Request timeout in seconds (default: 30)

    For Bedrock:
        use_bedrock: true
        aws_region: AWS region (default: us-west-2)
    """

    name = "claude"

    def __init__(self) -> None:
        self._client = None
        self._model: str = "claude-haiku-4-5-20251001"
        self._timeout: int = 30
        self._use_bedrock: bool = False
        self._aws_region: str = "us-west-2"

    def configure(self, config: dict) -> None:
        """Configure the Claude provider.

        Args:
            config: Configuration dict with optional keys:
                - model: Model name
                - api_key: API key (or uses ANTHROPIC_API_KEY env var)
                - timeout: Request timeout
                - use_bedrock: Use AWS Bedrock instead of direct API
                - aws_region: AWS region for Bedrock
        """
        self._model = config.get("model", self._model)
        self._timeout = config.get("timeout", self._timeout)
        self._use_bedrock = config.get("use_bedrock", False)
        self._aws_region = config.get("aws_region", self._aws_region)

        if self._use_bedrock:
            self._init_bedrock_client()
        else:
            self._init_anthropic_client(config.get("api_key"))

    def _init_anthropic_client(self, api_key: str | None = None) -> None:
        """Initialize the standard Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install 'local-kb[llm]'"
            )

        # Use provided key, or fall back to env var (SDK default behavior)
        kwargs = {"timeout": self._timeout}
        if api_key:
            kwargs["api_key"] = api_key

        self._client = Anthropic(**kwargs)

    def _init_bedrock_client(self) -> None:
        """Initialize the Bedrock client."""
        try:
            from anthropic import AnthropicBedrock
        except ImportError:
            raise ImportError(
                "anthropic[bedrock] package not installed. "
                "Install with: pip install 'anthropic[bedrock]'"
            )

        self._client = AnthropicBedrock(
            aws_region=self._aws_region,
            timeout=self._timeout,
        )

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a completion using Claude.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with generated content
        """
        if self._client is None:
            raise RuntimeError("Provider not configured. Call configure() first.")

        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        # Extract text content from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        if self._client is None:
            return False

        # For Bedrock, assume available if client is configured
        # (AWS credentials are validated on first request)
        if self._use_bedrock:
            return True

        # For direct API, check if API key is set
        return bool(os.environ.get("ANTHROPIC_API_KEY") or hasattr(self._client, "_api_key"))

    def list_models(self) -> list[str]:
        """List available Claude models.

        Returns a static list of commonly used models.
        """
        if self._use_bedrock:
            return [
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "anthropic.claude-sonnet-4-20250514-v1:0",
                "anthropic.claude-opus-4-20250514-v1:0",
            ]
        return [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ]


class ModalProvider:
    """Modal-based LLM provider using open models (Llama, Mistral, etc.).

    Runs on Modal GPU infrastructure - no API key needed, pay per compute.
    Requires deploying the Modal app first: `modal deploy lkb/modal_llm.py`

    Config:
        model: Model name (default: meta-llama/Llama-3.2-3B-Instruct)
        timeout: Request timeout in seconds (default: 60)
    """

    name = "modal"

    def __init__(self) -> None:
        self._llm = None
        self._model: str = "meta-llama/Llama-3.2-3B-Instruct"
        self._timeout: int = 60

    def configure(self, config: dict) -> None:
        """Configure the Modal provider.

        Args:
            config: Configuration dict with optional keys:
                - model: HuggingFace model ID
                - timeout: Request timeout in seconds
        """
        self._model = config.get("model", self._model)
        self._timeout = config.get("timeout", self._timeout)

        try:
            import modal
        except ImportError:
            raise ImportError("modal package not installed. Install with: pip install modal")

        try:
            self._llm = modal.Cls.from_name("knowledge-llm", "LLM")()
        except modal.exception.NotFoundError:
            raise RuntimeError(
                "Modal LLM app not deployed. Deploy with: modal deploy lkb/modal_llm.py"
            )

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 256,
    ) -> LLMResponse:
        """Generate a completion using Modal LLM.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with generated content
        """
        if self._llm is None:
            raise RuntimeError("Provider not configured. Call configure() first.")

        response = self._llm.complete.remote(
            prompt,
            system=system,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response["content"],
            model=response["model"],
            input_tokens=response.get("input_tokens"),
            output_tokens=response.get("output_tokens"),
        )

    def is_available(self) -> bool:
        """Check if Modal LLM is available."""
        return self._llm is not None

    def list_models(self) -> list[str]:
        """List recommended models for Modal."""
        return [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]


# Registry of available providers
PROVIDERS: dict[str, type] = {
    "claude": ClaudeProvider,
    "modal": ModalProvider,
}

# Cached provider instance
_provider_instance: LLMProvider | None = None
_provider_initialized: bool = False


def get_provider() -> LLMProvider | None:
    """Get the configured LLM provider instance.

    Returns None if LLM is disabled (no provider configured).
    Caches the provider instance for reuse.
    """
    global _provider_instance, _provider_initialized

    if _provider_initialized:
        return _provider_instance

    from ..config import config

    # Check if LLM is configured
    provider_name = config.llm_provider
    if not provider_name:
        _provider_initialized = True
        return None

    if provider_name not in PROVIDERS:
        print(
            f"Warning: Unknown LLM provider '{provider_name}'. Available: {list(PROVIDERS.keys())}",
            file=sys.stderr,
        )
        _provider_initialized = True
        return None

    # Create and configure provider
    provider_class = PROVIDERS[provider_name]
    provider = provider_class()

    # Build config dict from Config object
    provider_config = {
        "model": config.llm_model,
        "timeout": config.llm_timeout,
        "use_bedrock": config.llm_use_bedrock,
        "aws_region": config.llm_aws_region,
    }

    try:
        provider.configure(provider_config)
    except ImportError as e:
        print(f"Warning: Could not initialize LLM provider: {e}", file=sys.stderr)
        _provider_initialized = True
        return None

    _provider_instance = provider
    _provider_initialized = True
    return provider


def reset_provider() -> None:
    """Reset the cached provider instance.

    Useful for testing or after config changes.
    """
    global _provider_instance, _provider_initialized
    _provider_instance = None
    _provider_initialized = False
