"""
Skill auto-discovery — loads skills from built-in and user directories.

Scans two locations:
  - arc/skills/builtin/*.py   built-in skills (always loaded, no code change needed)
  - ~/.arc/skills/*.py        user custom skills (drop a .py file, restart Arc)

Also loads "soft skills":
  - ~/.arc/skills/*.md        plain-text instruction files injected into the
                              system prompt (persona, domain knowledge, style rules)
                              — no Python required.

Adding a new built-in skill:
    1. Create arc/skills/builtin/my_skill.py with a concrete Skill subclass.
    2. Restart Arc — it's automatically discovered and registered.
    No changes to __init__.py or cli/main.py needed.

Adding a user skill (no code deployment):
    1. Drop a .py file in ~/.arc/skills/ with a concrete Skill subclass.
    2. Restart Arc.

Adding a soft skill:
    1. Drop a .md file in ~/.arc/skills/.
    2. Restart Arc — its content is injected into every system prompt.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path

from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# Package name for built-in skills — used for clean importlib.import_module calls
# so that relative imports inside each skill file resolve correctly.
_BUILTIN_PACKAGE = "arc.skills.builtin"
_BUILTIN_DIR = Path(__file__).parent / "builtin"

# Default user skills directory
_USER_SKILLS_DIR = Path.home() / ".arc" / "skills"


# ━━━ Internal helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _collect_skill_classes(module: object) -> list[type[Skill]]:
    """Return all concrete Skill subclasses found in *module*."""
    found: list[type[Skill]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, Skill)
            and obj is not Skill
            and not inspect.isabstract(obj)
        ):
            found.append(obj)
    return found


def _scan_builtin_dir() -> list[type[Skill]]:
    """
    Scan arc/skills/builtin/*.py via the package import system.

    Using importlib.import_module (not spec_from_file_location) ensures that
    absolute imports inside each skill file (e.g. `from arc.core.types import …`)
    resolve correctly against the installed package.
    """
    seen: set[type] = set()
    classes: list[type[Skill]] = []

    for py_file in sorted(_BUILTIN_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue  # skip __init__.py, __pycache__, etc.

        module_name = f"{_BUILTIN_PACKAGE}.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            for cls in _collect_skill_classes(module):
                if cls not in seen:
                    seen.add(cls)
                    classes.append(cls)
            logger.debug(f"Loaded builtin skill module: {py_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load builtin skill '{py_file.name}': {e}")

    return classes


def _scan_user_dir(user_dir: Path) -> list[type[Skill]]:
    """
    Scan user_dir/*.py for custom skills.

    Each file is loaded as an isolated module so user skills can't
    accidentally shadow built-in names.
    """
    if not user_dir.exists():
        return []

    seen: set[type] = set()
    classes: list[type[Skill]] = []

    for py_file in sorted(user_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        # Use a namespaced module name to avoid collisions between user files
        module_name = f"arc_user_skills.{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Cannot create module spec for '{py_file.name}', skipping")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]

            for cls in _collect_skill_classes(module):
                if cls not in seen:
                    seen.add(cls)
                    classes.append(cls)
            logger.debug(f"Loaded user skill module: {py_file.name}")
        except Exception as e:
            # A broken user skill must never crash Arc startup — log and continue.
            logger.warning(f"Failed to load user skill '{py_file.name}': {e}")

    return classes


def _load_soft_skills(user_dir: Path) -> list[tuple[str, str]]:
    """
    Read *.md files from user_dir as soft skills.

    Returns a list of (name, content) pairs, sorted by filename.
    """
    if not user_dir.exists():
        return []

    results: list[tuple[str, str]] = []
    for md_file in sorted(user_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                results.append((md_file.stem, content))
                logger.debug(f"Loaded soft skill: {md_file.name}")
        except Exception as e:
            logger.warning(f"Failed to read soft skill '{md_file.name}': {e}")

    return results


# ━━━ Public API ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def discover_skills(user_dir: Path | None = None) -> list[Skill]:
    """
    Auto-discover and instantiate all available hard skills.

    Discovery order:
      1. arc/skills/builtin/*.py  (built-ins, always loaded)
      2. user_dir/*.py            (user custom skills, default ~/.arc/skills/)

    Only concrete Skill subclasses that can be instantiated with no arguments
    are returned. Broken files are skipped with a warning.
    """
    if user_dir is None:
        user_dir = _USER_SKILLS_DIR

    all_classes = _scan_builtin_dir() + _scan_user_dir(user_dir)

    instances: list[Skill] = []
    for cls in all_classes:
        try:
            instances.append(cls())
            logger.debug(f"Instantiated skill: {cls.__name__}")
        except Exception as e:
            logger.warning(f"Failed to instantiate skill '{cls.__name__}': {e}")

    skill_names = [s.manifest().name for s in instances]
    logger.info(f"Discovered {len(instances)} skill(s): {skill_names}")
    return instances


def discover_soft_skills(user_dir: Path | None = None) -> str:
    """
    Load all soft skills (*.md files) from user_dir.

    Returns a string ready to be appended to the system prompt,
    or an empty string if no soft skills exist.

    Example ~/.arc/skills/python_expert.md:
        You are an expert Python developer. Always prefer type hints,
        follow PEP 8, and suggest dataclasses over plain dicts.

    At runtime this becomes part of every system prompt automatically.
    """
    if user_dir is None:
        user_dir = _USER_SKILLS_DIR

    soft_skills = _load_soft_skills(user_dir)
    if not soft_skills:
        return ""

    parts = ["\n\n## Additional Instructions"]
    for name, content in soft_skills:
        title = name.replace("_", " ").replace("-", " ").title()
        parts.append(f"\n### {title}\n{content}")

    return "\n".join(parts)
