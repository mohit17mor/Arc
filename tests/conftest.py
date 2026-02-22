"""Shared test fixtures for Arc."""

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.core.bus import EventBus
from arc.core.registry import Registry
from arc.llm.mock import MockLLMProvider


@pytest.fixture
def config():
    """Create a default config without loading from disk."""
    return ArcConfig()


@pytest.fixture
def bus():
    """Create a fresh event bus."""
    return EventBus()


@pytest.fixture
def registry():
    """Create a fresh registry."""
    return Registry()


@pytest.fixture
def kernel(config):
    """Create a kernel with default config."""
    return Kernel(config=config)


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return MockLLMProvider()