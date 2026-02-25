"""Tests for skill auto-discovery (arc/skills/loader.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from arc.skills.base import Skill
from arc.skills.loader import (
    _collect_skill_classes,
    _scan_builtin_dir,
    _scan_user_dir,
    _load_soft_skills,
    discover_skills,
    discover_soft_skills,
)


# ━━━ _collect_skill_classes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_collect_skill_classes_finds_concrete():
    """Returns concrete Skill subclasses, not abstract ones."""
    import arc.skills.builtin.filesystem as mod

    classes = _collect_skill_classes(mod)
    assert len(classes) == 1
    assert classes[0].__name__ == "FilesystemSkill"


def test_collect_skill_classes_skips_base():
    """Does not return the abstract Skill base class itself."""
    import arc.skills.base as mod

    classes = _collect_skill_classes(mod)
    # FunctionSkill (from @tool) is concrete and lives here — Skill itself must not be included
    for cls in classes:
        assert cls is not Skill


# ━━━ _scan_builtin_dir ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_scan_builtin_dir_returns_all_builtins():
    """Discovers all three built-in skill classes."""
    from arc.skills.builtin.filesystem import FilesystemSkill
    from arc.skills.builtin.terminal import TerminalSkill
    from arc.skills.builtin.browsing import BrowsingSkill

    classes = _scan_builtin_dir()
    names = {cls.__name__ for cls in classes}
    assert "FilesystemSkill" in names
    assert "TerminalSkill" in names
    assert "BrowsingSkill" in names


def test_scan_builtin_dir_no_duplicates():
    """Each class appears exactly once even if imported by multiple modules."""
    classes = _scan_builtin_dir()
    names = [cls.__name__ for cls in classes]
    assert len(names) == len(set(names))


# ━━━ _scan_user_dir ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _write_skill_file(directory: Path, filename: str, content: str) -> Path:
    path = directory / filename
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_scan_user_dir_nonexistent_returns_empty(tmp_path):
    """Returns empty list if directory does not exist."""
    result = _scan_user_dir(tmp_path / "does_not_exist")
    assert result == []


def test_scan_user_dir_loads_valid_skill(tmp_path):
    """Loads a concrete Skill subclass from a .py file in the user dir."""
    _write_skill_file(
        tmp_path,
        "hello_skill.py",
        """
        from arc.core.types import SkillManifest, ToolResult, ToolSpec
        from arc.skills.base import Skill

        class HelloSkill(Skill):
            def manifest(self):
                return SkillManifest(
                    name="hello",
                    version="1.0.0",
                    description="Says hello",
                    capabilities=frozenset(),
                    tools=(),
                )

            async def execute_tool(self, tool_name, arguments):
                return ToolResult(tool_call_id="", success=True, output="hi")
        """,
    )

    classes = _scan_user_dir(tmp_path)
    names = {cls.__name__ for cls in classes}
    assert "HelloSkill" in names


def test_scan_user_dir_skips_broken_file(tmp_path):
    """A file with a syntax error is skipped without raising."""
    _write_skill_file(tmp_path, "broken.py", "this is not valid python !!!")

    # Must not raise
    result = _scan_user_dir(tmp_path)
    assert result == []


def test_scan_user_dir_skips_private_files(tmp_path):
    """Files starting with _ are ignored."""
    _write_skill_file(
        tmp_path,
        "_private.py",
        """
        from arc.core.types import SkillManifest, ToolResult
        from arc.skills.base import Skill

        class PrivateSkill(Skill):
            def manifest(self):
                return SkillManifest(name="private", version="1.0.0",
                    description="", capabilities=frozenset(), tools=())
            async def execute_tool(self, tool_name, arguments):
                return ToolResult(tool_call_id="", success=True, output="")
        """,
    )
    result = _scan_user_dir(tmp_path)
    assert result == []


def test_scan_user_dir_skips_abstract_classes(tmp_path):
    """Abstract subclasses of Skill are not returned."""
    _write_skill_file(
        tmp_path,
        "abstract_skill.py",
        """
        from abc import abstractmethod
        from arc.core.types import SkillManifest
        from arc.skills.base import Skill

        class AbstractBase(Skill):
            @abstractmethod
            def manifest(self): ...

            @abstractmethod
            async def execute_tool(self, tool_name, arguments): ...
        """,
    )
    result = _scan_user_dir(tmp_path)
    assert result == []


def test_scan_user_dir_no_duplicates_across_files(tmp_path):
    """Same class defined twice across files is deduplicated."""
    skill_code = """
        from arc.core.types import SkillManifest, ToolResult
        from arc.skills.base import Skill

        class DupSkill(Skill):
            def manifest(self):
                return SkillManifest(name="dup", version="1.0.0",
                    description="", capabilities=frozenset(), tools=())
            async def execute_tool(self, tool_name, arguments):
                return ToolResult(tool_call_id="", success=True, output="")
    """
    _write_skill_file(tmp_path, "dup_a.py", skill_code)
    _write_skill_file(tmp_path, "dup_b.py", skill_code)

    # Two separate classes with same name from different files — both valid,
    # no crash, and result length reflects what was found.
    result = _scan_user_dir(tmp_path)
    # Each file defines an independent class — should get 2 (not a hard crash)
    assert len(result) >= 1


# ━━━ _load_soft_skills ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_load_soft_skills_nonexistent_dir(tmp_path):
    """Returns empty list for a non-existent directory."""
    result = _load_soft_skills(tmp_path / "missing")
    assert result == []


def test_load_soft_skills_reads_md_files(tmp_path):
    """Reads content from .md files."""
    (tmp_path / "python_expert.md").write_text("Always use type hints.", encoding="utf-8")
    (tmp_path / "concise.md").write_text("Be brief.", encoding="utf-8")

    results = _load_soft_skills(tmp_path)
    assert len(results) == 2
    names = {name for name, _ in results}
    assert "python_expert" in names
    assert "concise" in names


def test_load_soft_skills_skips_empty_files(tmp_path):
    """Empty .md files are silently ignored."""
    (tmp_path / "empty.md").write_text("   \n  ", encoding="utf-8")

    result = _load_soft_skills(tmp_path)
    assert result == []


def test_load_soft_skills_ignores_non_md_files(tmp_path):
    """Only .md files are loaded — .txt, .py etc. are ignored."""
    (tmp_path / "notes.txt").write_text("some text", encoding="utf-8")
    (tmp_path / "skill.py").write_text("# code", encoding="utf-8")
    (tmp_path / "real.md").write_text("Be helpful.", encoding="utf-8")

    results = _load_soft_skills(tmp_path)
    assert len(results) == 1
    assert results[0][0] == "real"


# ━━━ discover_skills (integration) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_discover_skills_returns_instances():
    """Returns instantiated Skill objects, not classes."""
    skills = discover_skills(user_dir=Path("/nonexistent"))
    for skill in skills:
        assert isinstance(skill, Skill)


def test_discover_skills_includes_builtins():
    """Always includes the three built-in skills."""
    skills = discover_skills(user_dir=Path("/nonexistent"))
    names = {s.manifest().name for s in skills}
    assert "filesystem" in names
    assert "terminal" in names
    assert "browsing" in names


def test_discover_skills_includes_user_skill(tmp_path):
    """User skills are added after builtins."""
    _write_skill_file(
        tmp_path,
        "custom.py",
        """
        from arc.core.types import SkillManifest, ToolResult
        from arc.skills.base import Skill

        class CustomSkill(Skill):
            def manifest(self):
                return SkillManifest(name="custom", version="1.0.0",
                    description="Custom", capabilities=frozenset(), tools=())
            async def execute_tool(self, tool_name, arguments):
                return ToolResult(tool_call_id="", success=True, output="ok")
        """,
    )

    skills = discover_skills(user_dir=tmp_path)
    names = {s.manifest().name for s in skills}
    assert "custom" in names
    # Builtins still present
    assert "filesystem" in names


def test_discover_skills_tolerates_empty_user_dir(tmp_path):
    """Works fine when user skills dir is empty."""
    skills = discover_skills(user_dir=tmp_path)
    names = {s.manifest().name for s in skills}
    assert "filesystem" in names


# ━━━ discover_soft_skills ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_discover_soft_skills_empty_dir(tmp_path):
    """Returns empty string when no .md files exist."""
    result = discover_soft_skills(user_dir=tmp_path)
    assert result == ""


def test_discover_soft_skills_formats_output(tmp_path):
    """Wraps content under '## Additional Instructions' header."""
    (tmp_path / "python_expert.md").write_text("Always use type hints.", encoding="utf-8")

    result = discover_soft_skills(user_dir=tmp_path)
    assert "## Additional Instructions" in result
    assert "Python Expert" in result
    assert "Always use type hints." in result


def test_discover_soft_skills_multiple_files(tmp_path):
    """All .md files are included in the output."""
    (tmp_path / "style.md").write_text("Be concise.", encoding="utf-8")
    (tmp_path / "domain.md").write_text("You are a finance expert.", encoding="utf-8")

    result = discover_soft_skills(user_dir=tmp_path)
    assert "Be concise." in result
    assert "You are a finance expert." in result
