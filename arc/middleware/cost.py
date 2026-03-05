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

    Separates main-agent tokens from worker/sub-agent tokens so
    users can see the breakdown.  Also tracks per-turn tokens so
    platforms can display usage after each message.

    Usage:
        tracker = CostTracker()
        kernel.use(tracker.middleware)

        # After some LLM calls...
        print(f"Total cost: ${tracker.session_cost:.4f}")
    """

    # Session-wide totals (main agent only)
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

    # Worker/sub-agent totals (aggregated)
    worker_input_tokens: int = 0
    worker_output_tokens: int = 0
    worker_request_count: int = 0

    # Per-turn counters — reset before each user turn
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0
    turn_request_count: int = 0
    turn_peak_input: int = 0  # largest single-call input_tokens this turn

    # Model context window — set once at startup
    context_window: int = 0

    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0

    @property
    def session_cost(self) -> float:
        """Total cost for this session in USD (main + workers)."""
        total_in = self.input_tokens + self.worker_input_tokens
        total_out = self.output_tokens + self.worker_output_tokens
        return (
            total_in * self.cost_per_input_token
            + total_out * self.cost_per_output_token
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens used (main + workers)."""
        return (
            self.input_tokens + self.output_tokens
            + self.worker_input_tokens + self.worker_output_tokens
        )

    @property
    def main_total_tokens(self) -> int:
        """Total tokens used by main agent only."""
        return self.input_tokens + self.output_tokens

    @property
    def worker_total_tokens(self) -> int:
        """Total tokens used by all workers combined."""
        return self.worker_input_tokens + self.worker_output_tokens

    @property
    def turn_total_tokens(self) -> int:
        """Tokens used in the current turn (main agent only)."""
        return self.turn_input_tokens + self.turn_output_tokens

    def start_turn(self) -> None:
        """Reset per-turn counters. Call at the start of each user message."""
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0
        self.turn_request_count = 0
        self.turn_peak_input = 0

    async def middleware(self, event: Event, next_handler: MiddlewareNext) -> Event:
        """Middleware that tracks LLM response costs."""
        result = await next_handler(event)

        if event.type == EventType.LLM_RESPONSE:
            inp = event.data.get("input_tokens", 0)
            out = event.data.get("output_tokens", 0)
            source = event.source or ""

            is_worker = source.startswith("worker:") or source.startswith("scheduler")

            if is_worker:
                self.worker_request_count += 1
                self.worker_input_tokens += inp
                self.worker_output_tokens += out
            else:
                self.request_count += 1
                self.input_tokens += inp
                self.output_tokens += out
                # Per-turn (main agent only)
                self.turn_request_count += 1
                self.turn_input_tokens += inp
                self.turn_output_tokens += out
                if inp > self.turn_peak_input:
                    self.turn_peak_input = inp

            logger.debug(
                f"LLM call ({'worker' if is_worker else 'main'}): "
                f"+{inp} in, +{out} out, "
                f"total: {self.total_tokens} tokens"
            )

        return result

    def reset(self) -> None:
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0
        self.worker_input_tokens = 0
        self.worker_output_tokens = 0
        self.worker_request_count = 0
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0
        self.turn_request_count = 0
        self.turn_peak_input = 0

    def summary(self) -> dict[str, Any]:
        """Get summary of usage."""
        return {
            "requests": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.main_total_tokens,
            "cost_usd": self.session_cost,
            # Worker breakdown
            "worker_requests": self.worker_request_count,
            "worker_input_tokens": self.worker_input_tokens,
            "worker_output_tokens": self.worker_output_tokens,
            "worker_total_tokens": self.worker_total_tokens,
            # Grand total
            "grand_total_tokens": self.total_tokens,
            # Per-turn (most recent)
            "turn_input_tokens": self.turn_input_tokens,
            "turn_output_tokens": self.turn_output_tokens,
            "turn_total_tokens": self.turn_total_tokens,
            "turn_requests": self.turn_request_count,
            "turn_peak_input": self.turn_peak_input,
            # Model info
            "context_window": self.context_window,
        }