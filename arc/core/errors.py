"""
Arc exception hierarchy.

Every error in the system inherits from ArcError.
Each subsystem has its own error class for targeted catching.

Usage:
    try:
        await provider.generate(...)
    except LLMError as e:
        # Handle LLM-specific failures
    except ArcError as e:
        # Handle any Arc error
"""


class ArcError(Exception):
    """Base exception for all Arc errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


# ━━━ Layer 0: Core Errors ━━━


class ConfigError(ArcError):
    """Configuration is invalid, missing, or malformed."""

    pass


class RegistryError(ArcError):
    """Provider not found or registration conflict."""

    pass


class ProviderNotFoundError(RegistryError):
    """Requested provider does not exist in registry."""

    pass


# ━━━ Layer 1: Provider Errors ━━━


class LLMError(ArcError):
    """LLM provider failure — API errors, rate limits, etc."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        retryable: bool = False,
        details: dict | None = None,
    ):
        self.provider = provider
        self.model = model
        self.retryable = retryable
        super().__init__(message, details)


class StorageError(ArcError):
    """Storage backend failure — database errors, corruption, etc."""

    pass


class ShellError(ArcError):
    """Shell session failure — process errors, timeout, etc."""

    pass


class EmbeddingError(ArcError):
    """Embedding computation failure."""

    pass


# ━━━ Layer 2: System Errors ━━━


class SkillError(ArcError):
    """Skill activation, execution, or lifecycle failure."""

    def __init__(
        self,
        message: str,
        skill_name: str = "",
        tool_name: str = "",
        details: dict | None = None,
    ):
        self.skill_name = skill_name
        self.tool_name = tool_name
        super().__init__(message, details)


class SecurityError(ArcError):
    """Permission denied or policy violation."""

    def __init__(
        self,
        message: str,
        capability: str = "",
        action: str = "",
        details: dict | None = None,
    ):
        self.capability = capability
        self.action = action
        super().__init__(message, details)


class MemoryError_(ArcError):
    """Memory system failure. Named with underscore to avoid shadowing builtin."""

    pass


# ━━━ Layer 3+: Higher Layer Errors ━━━


class RecipeError(ArcError):
    """Invalid recipe, parsing failure, or validation error."""

    pass


class PipelineError(ArcError):
    """Pipeline step failure or type mismatch."""

    def __init__(
        self,
        message: str,
        step: int = -1,
        recipe: str = "",
        details: dict | None = None,
    ):
        self.step = step
        self.recipe = recipe
        super().__init__(message, details)


class TeachError(ArcError):
    """Teaching recording or analysis failure."""

    pass