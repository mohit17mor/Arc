"""
Scheduler Job — the core data model.

A Job describes what to run, when to run it, and its current state.
Triggers are stored as plain dicts so they serialize cleanly to SQLite JSON.

Trigger dict shapes:
    {"type": "cron",     "expression": "0 9 * * 1-5"}
    {"type": "interval", "seconds": 1800}
    {"type": "oneshot",  "at": 1740481200}
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    """A scheduled task."""

    name: str            # human-readable label, unique per user
    prompt: str          # what to ask the LLM when this fires
    trigger: dict        # serialised trigger (see module docstring)

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    next_run: int = 0    # unix timestamp — 0 means "compute on first check"
    last_run: int = 0    # unix timestamp — 0 means never run
    active: bool = True
    use_tools: bool = False  # False = plain LLM text; True = full sub-agent with tools
    created_at: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "trigger": self.trigger,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "active": self.active,
            "use_tools": self.use_tools,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            id=d["id"],
            name=d["name"],
            prompt=d["prompt"],
            trigger=d["trigger"],
            next_run=d["next_run"],
            last_run=d["last_run"],
            active=bool(d["active"]),
            use_tools=bool(d.get("use_tools", False)),  # default False for old rows
            created_at=d["created_at"],
        )
