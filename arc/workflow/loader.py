"""
Workflow YAML loader — reads workflow files from ~/.arc/workflows/.

Supports two step formats:

    Simple (just a string):
        steps:
          - search the web for NVIDIA news
          - write a summary of the results

    Extended (dict with options):
        steps:
          - do: search the web for NVIDIA news
            retry: 2
            on_fail: continue
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from arc.workflow.types import OnFail, Workflow, WorkflowStep

logger = logging.getLogger(__name__)

# Default directory for workflow YAML files
_WORKFLOWS_DIR = Path.home() / ".arc" / "workflows"


def load_workflows(directory: Path | None = None) -> list[Workflow]:
    """
    Discover and load all workflow YAML files from a directory.

    Looks for ``*.yaml`` and ``*.yml`` files.  Silently skips files
    that fail to parse (logs a warning).
    """
    workflows_dir = directory or _WORKFLOWS_DIR
    if not workflows_dir.exists():
        return []

    workflows = []
    for path in sorted(workflows_dir.glob("*.y*ml")):
        if path.suffix not in (".yaml", ".yml"):
            continue
        try:
            wf = load_workflow_file(path)
            if wf:
                workflows.append(wf)
                logger.info(f"Loaded workflow: {wf.name} ({len(wf.steps)} steps)")
        except Exception as e:
            logger.warning(f"Failed to load workflow {path.name}: {e}")

    return workflows


def load_workflow_file(path: Path) -> Workflow | None:
    """Load a single workflow YAML file."""
    try:
        import yaml
    except ImportError:
        # PyYAML not installed — try tomllib-style fallback
        logger.warning("PyYAML not installed. Install with: pip install pyyaml")
        return None

    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    if not data or not isinstance(data, dict):
        return None

    return _parse_workflow(data, source_path=str(path))


def parse_workflow_from_dict(data: dict[str, Any]) -> Workflow:
    """Parse a workflow from a dict (for testing)."""
    return _parse_workflow(data, source_path="<dict>")


def _parse_workflow(data: dict[str, Any], source_path: str = "") -> Workflow:
    """Parse a workflow dict into a Workflow object."""
    name = data.get("name", Path(source_path).stem if source_path else "unnamed")
    description = data.get("description", "")

    # Parse trigger patterns
    trigger = data.get("trigger", "")
    if isinstance(trigger, str):
        trigger_patterns = [trigger] if trigger else []
    elif isinstance(trigger, list):
        trigger_patterns = trigger
    else:
        trigger_patterns = []

    # Parse steps
    raw_steps = data.get("steps", [])
    if not raw_steps:
        raise ValueError(f"Workflow '{name}' has no steps")

    steps = []
    for i, raw in enumerate(raw_steps):
        step = _parse_step(raw, index=i)
        steps.append(step)

    return Workflow(
        name=name,
        steps=steps,
        trigger_patterns=trigger_patterns,
        description=description,
        source_path=source_path,
    )


def _parse_step(raw: Any, index: int) -> WorkflowStep:
    """Parse a single step — handles both simple string and extended dict."""
    if isinstance(raw, str):
        # Simple form: just a plain-English instruction
        return WorkflowStep(instruction=raw.strip(), index=index)

    if isinstance(raw, dict):
        # Extended form
        instruction = raw.get("do", raw.get("instruction", ""))
        if not instruction:
            raise ValueError(f"Step {index} has no 'do' instruction")

        on_fail_str = raw.get("on_fail", "stop").lower()
        on_fail = OnFail.CONTINUE if on_fail_str == "continue" else OnFail.STOP

        return WorkflowStep(
            instruction=instruction.strip(),
            index=index,
            retry=int(raw.get("retry", 0)),
            on_fail=on_fail,
            ask_if_unclear=raw.get("ask_if_unclear", True),
            tool=raw.get("tool"),
            args=raw.get("args"),
            shell=raw.get("shell"),
        )

    raise ValueError(f"Step {index}: expected string or dict, got {type(raw).__name__}")


def match_workflow(user_input: str, workflows: list[Workflow]) -> Workflow | None:
    """
    Find a workflow whose trigger pattern matches the user input.

    Returns the first match, or None.
    """
    lower = user_input.lower()
    for wf in workflows:
        for pattern in wf.trigger_patterns:
            try:
                if re.search(pattern, lower):
                    return wf
            except re.error:
                # Treat as a plain substring match if regex fails
                if pattern.lower() in lower:
                    return wf
    return None
