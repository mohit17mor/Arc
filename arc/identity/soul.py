"""
Soul Manager — handles identity.md file.

The identity.md file stores:
- Agent name
- User name
- Personality
- Learned facts about the user
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arc.identity.personality import get_personality

logger = logging.getLogger(__name__)

IDENTITY_TEMPLATE = '''# {agent_name}'s Soul

## Identity
name: {agent_name}
created: {created_date}
personality: {personality_id}

## My Human
user_name: {user_name}

## How I Behave
{personality_description}

## Things I've Learned About {user_name}
(This section grows as we interact)

---
*Edit this file to customize my personality. Changes take effect immediately.*
'''


class SoulManager:
    """
    Manages the identity.md file.

    Usage:
        soul = SoulManager(Path("~/.arc/identity.md"))
        soul.create("Friday", "Alex", "sarcastic")

        identity = soul.load()
        print(identity["agent_name"])  # "Friday"
    """

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def exists(self) -> bool:
        """Check if identity file exists."""
        return self._path.exists()

    def create(
        self,
        agent_name: str,
        user_name: str,
        personality_id: str,
        custom_system_prompt: str | None = None,
    ) -> None:
        """Create a new identity file."""
        from datetime import datetime

        personality = get_personality(personality_id)
        personality_description = (
            custom_system_prompt.strip()
            if personality_id == "custom" and custom_system_prompt and custom_system_prompt.strip()
            else personality.system_prompt
        )

        content = IDENTITY_TEMPLATE.format(
            agent_name=agent_name,
            user_name=user_name,
            personality_id=personality_id,
            personality_description=personality_description,
            created_date=datetime.now().strftime("%Y-%m-%d"),
        )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(content, encoding="utf-8")
        logger.info(f"Created identity at {self._path}")

    def load(self) -> dict[str, Any]:
        """Load identity from file."""
        if not self.exists():
            return {
                "agent_name": "Arc",
                "user_name": "User",
                "personality_id": "helpful",
                "system_prompt": get_personality("helpful").system_prompt,
            }

        content = self._path.read_text(encoding="utf-8")
        return self._parse_identity(content)

    def _parse_identity(self, content: str) -> dict[str, Any]:
        """Parse identity.md content."""
        result: dict[str, Any] = {
            "agent_name": "Arc",
            "user_name": "User",
            "personality_id": "helpful",
            "custom_system_prompt": "",
            "raw_content": content,
        }

        lines = content.split("\n")
        current_section = ""
        how_i_behave_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Track sections
            if stripped.startswith("## "):
                current_section = stripped[3:].lower()
                continue

            if current_section == "how i behave":
                how_i_behave_lines.append(line)

            # Parse key-value pairs
            if ":" in stripped and not stripped.startswith("#"):
                key, value = stripped.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key == "name" and current_section == "identity":
                    result["agent_name"] = value
                elif key == "user_name":
                    result["user_name"] = value
                elif key == "personality":
                    result["personality_id"] = value

        result["custom_system_prompt"] = "\n".join(how_i_behave_lines).strip()

        # Build system prompt
        personality = get_personality(result["personality_id"])
        result["system_prompt"] = self._build_system_prompt(result, personality)

        return result

    def _build_system_prompt(
        self,
        identity: dict[str, Any],
        personality: Any,
    ) -> str:
        """Build the full system prompt from identity."""
        parts = [
            f"Your name is {identity['agent_name']}.",
            f"You are talking to {identity['user_name']}.",
            "",
            (
                identity.get("custom_system_prompt", "").strip()
                if identity.get("personality_id") == "custom"
                else personality.system_prompt
            ) or personality.system_prompt,
        ]
        return "\n".join(parts)

    def get_system_prompt(self) -> str:
        """Get the system prompt from identity."""
        identity = self.load()
        return identity.get("system_prompt", get_personality("helpful").system_prompt)
