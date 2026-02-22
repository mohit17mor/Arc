"""Tests for built-in skills."""

import pytest
from pathlib import Path
from arc.skills.builtin.filesystem import FilesystemSkill


@pytest.fixture
def fs_skill(tmp_path):
    """Filesystem skill with temp workspace."""
    skill = FilesystemSkill(workspace=tmp_path)
    return skill


@pytest.mark.asyncio
async def test_read_file(fs_skill, tmp_path):
    """read_file returns file contents."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, Arc!")

    result = await fs_skill.execute_tool("read_file", {"path": "test.txt"})
    assert result.success is True
    assert result.output == "Hello, Arc!"


@pytest.mark.asyncio
async def test_read_file_not_found(fs_skill):
    """read_file returns error for missing file."""
    result = await fs_skill.execute_tool("read_file", {"path": "nonexistent.txt"})
    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_write_file(fs_skill, tmp_path):
    """write_file creates file with content."""
    result = await fs_skill.execute_tool(
        "write_file",
        {"path": "output.txt", "content": "Test content"},
    )
    assert result.success is True

    created = tmp_path / "output.txt"
    assert created.exists()
    assert created.read_text() == "Test content"


@pytest.mark.asyncio
async def test_write_file_creates_dirs(fs_skill, tmp_path):
    """write_file creates parent directories."""
    result = await fs_skill.execute_tool(
        "write_file",
        {"path": "deep/nested/file.txt", "content": "Nested!"},
    )
    assert result.success is True

    created = tmp_path / "deep" / "nested" / "file.txt"
    assert created.exists()


@pytest.mark.asyncio
async def test_list_directory(fs_skill, tmp_path):
    """list_directory shows files and folders."""
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.txt").write_text("b")
    (tmp_path / "subdir").mkdir()

    result = await fs_skill.execute_tool("list_directory", {"path": "."})
    assert result.success is True
    assert "file1.txt" in result.output
    assert "file2.txt" in result.output
    assert "subdir" in result.output
    assert "[DIR]" in result.output
    assert "[FILE]" in result.output


@pytest.mark.asyncio
async def test_list_directory_empty(fs_skill, tmp_path):
    """list_directory handles empty directory."""
    result = await fs_skill.execute_tool("list_directory", {"path": "."})
    assert result.success is True
    assert "empty" in result.output.lower()


@pytest.mark.asyncio
async def test_list_directory_not_found(fs_skill):
    """list_directory returns error for missing directory."""
    result = await fs_skill.execute_tool("list_directory", {"path": "nonexistent"})
    assert result.success is False
    assert "not found" in result.error.lower()