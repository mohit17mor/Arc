"""Tests for arc/skills/builtin/code_intel.py"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from arc.skills.builtin.code_intel import (
    CodeIntelSkill,
    _collect_source_files,
    _detect_language,
    _extract_symbols,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def skill():
    return CodeIntelSkill()


@pytest.fixture
def sample_project(tmp_path):
    """Create a small sample project for testing."""
    # Python file
    (tmp_path / "app.py").write_text(
        'import os\n\n'
        'class Server:\n'
        '    def __init__(self, host: str):\n'
        '        self.host = host\n\n'
        '    def start(self) -> None:\n'
        '        pass\n\n'
        'def create_app(config: dict) -> Server:\n'
        '    return Server(config["host"])\n',
        encoding="utf-8",
    )
    # Another Python file
    (tmp_path / "utils.py").write_text(
        'def validate_config(config: dict) -> bool:\n'
        '    return "host" in config\n\n'
        'def format_url(host: str, port: int) -> str:\n'
        '    return f"http://{host}:{port}"\n',
        encoding="utf-8",
    )
    # Nested directory
    sub = tmp_path / "routes"
    sub.mkdir()
    (sub / "auth.py").write_text(
        'class AuthRouter:\n'
        '    def login(self, username: str) -> str:\n'
        '        return "token"\n\n'
        '    def logout(self) -> None:\n'
        '        pass\n',
        encoding="utf-8",
    )
    # Non-Python file (should be collected but symbols may vary)
    (sub / "index.js").write_text(
        'function handleRequest(req, res) {\n'
        '  res.send("ok");\n'
        '}\n',
        encoding="utf-8",
    )
    # Directory that should be skipped
    skip = tmp_path / "node_modules"
    skip.mkdir()
    (skip / "junk.py").write_text("# should be skipped", encoding="utf-8")

    return tmp_path


# ── Unit tests ───────────────────────────────────────────────────────────────


class TestDetectLanguage:
    def test_python(self):
        assert _detect_language(Path("foo.py")) == "python"

    def test_javascript(self):
        assert _detect_language(Path("bar.js")) == "javascript"

    def test_typescript(self):
        assert _detect_language(Path("baz.ts")) == "typescript"

    def test_rust(self):
        assert _detect_language(Path("lib.rs")) == "rust"

    def test_unknown(self):
        assert _detect_language(Path("readme.txt")) is None

    def test_case_insensitive(self):
        assert _detect_language(Path("App.PY")) == "python"


class TestCollectSourceFiles:
    def test_finds_source_files(self, sample_project):
        files = _collect_source_files(sample_project)
        names = {f.name for f in files}
        assert "app.py" in names
        assert "utils.py" in names
        assert "auth.py" in names
        assert "index.js" in names

    def test_skips_node_modules(self, sample_project):
        files = _collect_source_files(sample_project)
        names = {f.name for f in files}
        assert "junk.py" not in names

    def test_respects_max_files(self, sample_project):
        files = _collect_source_files(sample_project, max_files=2)
        assert len(files) <= 2

    def test_empty_dir(self, tmp_path):
        files = _collect_source_files(tmp_path)
        assert files == []


class TestExtractSymbols:
    def test_extracts_python_class_and_functions(self, sample_project):
        symbols = _extract_symbols(sample_project / "app.py", "python")
        joined = "\n".join(symbols)
        assert "class Server:" in joined
        assert "def create_app" in joined

    def test_extracts_nested_methods(self, sample_project):
        symbols = _extract_symbols(sample_project / "routes" / "auth.py", "python")
        joined = "\n".join(symbols)
        assert "class AuthRouter:" in joined
        assert "def login" in joined

    def test_returns_empty_for_unknown_language(self, tmp_path):
        (tmp_path / "data.txt").write_text("hello", encoding="utf-8")
        symbols = _extract_symbols(tmp_path / "data.txt", "unknown_lang")
        assert symbols == []


# ── Integration tests (full skill) ──────────────────────────────────────────


@pytest.mark.asyncio
class TestRepoMap:
    async def test_returns_project_structure(self, skill, sample_project):
        result = await skill.execute_tool("repo_map", {"path": str(sample_project)})
        assert result.success
        assert "Repository map:" in result.output
        assert "Server" in result.output
        assert "create_app" in result.output
        assert "AuthRouter" in result.output

    async def test_invalid_path(self, skill):
        result = await skill.execute_tool("repo_map", {"path": "/nonexistent/path"})
        assert not result.success
        assert "Not a directory" in result.error

    async def test_empty_dir(self, skill, tmp_path):
        result = await skill.execute_tool("repo_map", {"path": str(tmp_path)})
        assert result.success
        assert "No source files" in result.output


@pytest.mark.asyncio
class TestFindSymbol:
    async def test_finds_class(self, skill, sample_project):
        result = await skill.execute_tool("find_symbol", {
            "name": "Server",
            "path": str(sample_project),
        })
        assert result.success
        assert "class Server:" in result.output
        assert "def __init__" in result.output

    async def test_finds_function(self, skill, sample_project):
        result = await skill.execute_tool("find_symbol", {
            "name": "validate_config",
            "path": str(sample_project),
        })
        assert result.success
        assert "def validate_config" in result.output

    async def test_finds_nested_method_class(self, skill, sample_project):
        result = await skill.execute_tool("find_symbol", {
            "name": "AuthRouter",
            "path": str(sample_project),
        })
        assert result.success
        assert "class AuthRouter:" in result.output
        assert "def login" in result.output

    async def test_symbol_not_found(self, skill, sample_project):
        result = await skill.execute_tool("find_symbol", {
            "name": "NonexistentFunction",
            "path": str(sample_project),
        })
        assert result.success
        assert "not found" in result.output.lower()

    async def test_empty_name(self, skill, sample_project):
        result = await skill.execute_tool("find_symbol", {
            "name": "",
            "path": str(sample_project),
        })
        assert not result.success


@pytest.mark.asyncio
class TestSearchCode:
    async def test_finds_matches(self, skill, sample_project):
        result = await skill.execute_tool("search_code", {
            "pattern": "config",
            "path": str(sample_project),
        })
        assert result.success
        assert "match" in result.output.lower()
        # Should find in both app.py and utils.py
        assert "app.py" in result.output or "utils.py" in result.output

    async def test_case_insensitive(self, skill, sample_project):
        result = await skill.execute_tool("search_code", {
            "pattern": "SERVER",
            "path": str(sample_project),
        })
        assert result.success
        assert "match" in result.output.lower()

    async def test_no_matches(self, skill, sample_project):
        result = await skill.execute_tool("search_code", {
            "pattern": "xyzzy_nonexistent_42",
            "path": str(sample_project),
        })
        assert result.success
        assert "No matches" in result.output

    async def test_empty_pattern(self, skill, sample_project):
        result = await skill.execute_tool("search_code", {
            "pattern": "",
            "path": str(sample_project),
        })
        assert not result.success


@pytest.mark.asyncio
class TestManifest:
    async def test_manifest(self, skill):
        m = skill.manifest()
        assert m.name == "code_intel"
        tool_names = [t.name for t in m.tools]
        assert "repo_map" in tool_names
        assert "find_symbol" in tool_names
        assert "search_code" in tool_names

    async def test_unknown_tool(self, skill):
        result = await skill.execute_tool("nonexistent", {})
        assert not result.success
