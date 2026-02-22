"""
Arc Registry — typed service locator for dependency injection.

Components register themselves by CATEGORY and NAME.
Other components retrieve them by category (and optionally name).

Categories: "llm", "skill", "platform", "storage", "shell",
            "embedding", "middleware", "memory", "security"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from arc.core.errors import ProviderNotFoundError, RegistryError

logger = logging.getLogger(__name__)


class Registry:
    """
    Simple typed service locator.

    Usage:
        registry = Registry()

        # Register
        registry.register("llm", "ollama", ollama_provider)
        registry.register("skill", "filesystem", fs_skill)

        # Retrieve
        llm = registry.get("llm")                # default
        llm = registry.get("llm", "ollama")       # specific
        all_skills = registry.get_all("skill")     # all in category

    Default selection:
        When name is not specified:
        1. If a default is set for the category, use it
        2. Otherwise, return the first registered provider
    """

    def __init__(self) -> None:
        self._providers: dict[str, dict[str, Any]] = defaultdict(dict)
        self._defaults: dict[str, str] = {}
        self._registration_order: dict[str, list[str]] = defaultdict(list)

    def register(self, category: str, name: str, provider: Any) -> None:
        """
        Register a provider.

        Args:
            category: Provider category ("llm", "skill", etc.)
            name: Provider name ("ollama", "filesystem", etc.)
            provider: The provider instance

        If a provider with the same category+name exists, it's replaced.
        """
        is_new = name not in self._providers[category]
        self._providers[category][name] = provider

        if is_new:
            self._registration_order[category].append(name)

        logger.debug(f"Registered {category}/{name}")

    def get(self, category: str, name: str | None = None) -> Any:
        """
        Get a provider by category and optional name.

        If name is None, returns the default provider for the category.
        Default is either explicitly set or the first registered.

        Raises:
            ProviderNotFoundError: If category is empty or name not found
        """
        providers = self._providers.get(category)
        if not providers:
            raise ProviderNotFoundError(
                f"No providers registered for category '{category}'"
            )

        if name is None:
            # Return default
            default_name = self._defaults.get(category)
            if default_name and default_name in providers:
                return providers[default_name]
            # Return first registered
            first_name = self._registration_order[category][0]
            return providers[first_name]

        if name not in providers:
            available = ", ".join(providers.keys())
            raise ProviderNotFoundError(
                f"Provider '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )

        return providers[name]

    def get_all(self, category: str) -> list[Any]:
        """Get all providers in a category, in registration order."""
        providers = self._providers.get(category, {})
        order = self._registration_order.get(category, [])
        return [providers[name] for name in order if name in providers]

    def has(self, category: str, name: str | None = None) -> bool:
        """Check if a provider exists."""
        if name is None:
            return bool(self._providers.get(category))
        return name in self._providers.get(category, {})

    def set_default(self, category: str, name: str) -> None:
        """Set the default provider for a category."""
        if not self.has(category, name):
            raise RegistryError(
                f"Cannot set default: '{name}' not found in category '{category}'"
            )
        self._defaults[category] = name

    def get_names(self, category: str) -> list[str]:
        """List all provider names in a category."""
        return list(self._registration_order.get(category, []))

    def remove(self, category: str, name: str) -> None:
        """Remove a provider from the registry."""
        if category in self._providers and name in self._providers[category]:
            del self._providers[category][name]
            self._registration_order[category] = [
                n for n in self._registration_order[category] if n != name
            ]
            # Clear default if it was the removed provider
            if self._defaults.get(category) == name:
                del self._defaults[category]
            logger.debug(f"Removed {category}/{name}")

    def clear(self) -> None:
        """Remove all providers. Used in testing."""
        self._providers.clear()
        self._defaults.clear()
        self._registration_order.clear()