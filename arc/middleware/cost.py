"""
Cost Tracking Middleware — tracks token usage and costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from arc.core.events import Event, EventType
from arc.middleware.base import MiddlewareNext

logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    """
    Tracks token usage and costs across LLM calls.

    Usage:
        tracker = CostTracker()
        kernel.use(tracker.middleware)

        # After some LLM calls...
        print(f"Total cost: ${tracker.session_cost:.4f}")
    """

    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0

    @property
    def session_cost(self) -> float:
        """Total cost for this session in USD."""
        return (
            self.input_tokens * self.cost_per_input_token
            + self.output_tokens * self.cost_per_output_token
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    async def middleware(self, event: Event, next_handler: MiddlewareNext) -> Event:
        """Middleware that tracks LLM response costs."""
        result = await next_handler(event)

        if event.type == EventType.LLM_RESPONSE:
            self.request_count += 1
            self.input_tokens += event.data.get("input_tokens", 0)
            self.output_tokens += event.data.get("output_tokens", 0)

            logger.debug(
                f"LLM call #{self.request_count}: "
                f"+{event.data.get('input_tokens', 0)} in, "
                f"+{event.data.get('output_tokens', 0)} out, "
                f"total: {self.total_tokens} tokens"
            )

        return result

    def reset(self) -> None:
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    def summary(self) -> dict[str, Any]:
        """Get summary of usage."""
        return {
            "requests": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.session_cost,
        }