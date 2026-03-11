"""
Code Intelligence Skill — AST-aware code navigation and understanding.

Uses tree-sitter for parsing and grep-ast for context-aware code search.
Gives agents the ability to understand codebases structurally, not just
as raw text files.

Tools:
    repo_map(path)            — condensed project structure with symbols
    find_symbol(name, path)   — find where a function/class is defined
    search_code(pattern, path) — AST-aware grep with scope context

Dependencies:
    pip install grep-ast tree-sitter-language-pack
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# File extensions → tree-sitter language names
_LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "javascript",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "c_sharp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".json": "json",
    ".sh": "bash", ".bash": "bash",
}

# Directories to always skip
_SKIP_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "target", "out", ".idea", ".vscode",
}

# AST node types that represent function/class definitions per language
_DEF_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition",
                   "arrow_function", "export_statement"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                   "interface_declaration", "type_alias_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition",
            "interface_declaration", "type_alias_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "enum_item", "trait_item"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"class_declaration", "method_declaration", "interface_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {"class_declaration", "method_declaration", "interface_declaration"},
    "ruby": {"method", "class", "module"},
}

# Max files to scan for repo_map (safety limit)
_MAX_FILES = 500
_MAX_FILE_SIZE = 200_000  # 200KB — skip huge generated files


def _detect_language(filepath: Path) -> str | None:
    """Detect tree-sitter language from file extension."""
    return _LANG_MAP.get(filepath.suffix.lower())


def _collect_source_files(root: Path, max_files: int = _MAX_FILES) -> list[Path]:
    """Walk a directory and collect source files, skipping irrelevant dirs."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in sorted(filenames):
            fp = Path(dirpath) / fname
            if _detect_language(fp) and fp.stat().st_size < _MAX_FILE_SIZE:
                files.append(fp)
                if len(files) >= max_files:
                    return files
    return files


def _extract_symbols(filepath: Path, lang: str) -> list[str]:
    """Extract top-level symbol signatures from a source file using tree-sitter."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return []

    try:
        code = filepath.read_bytes()
        parser = get_parser(lang)
        tree = parser.parse(code)
    except Exception as e:
        logger.debug(f"Failed to parse {filepath}: {e}")
        return []

    def_types = _DEF_TYPES.get(lang, set())
    if not def_types:
        return []

    symbols: list[str] = []
    lines = code.decode("utf-8", errors="replace").splitlines()

    def walk(node, depth=0):
        if depth > 3:  # don't go too deep (skip nested functions etc.)
            return
        if node.type in def_types:
            # Get the first line of the definition as the signature
            start_line = node.start_point[0]
            if start_line < len(lines):
                sig = lines[start_line].strip()
                # Clean up trailing colons, braces for display
                indent = "  " * depth
                symbols.append(f"{indent}{sig}")
        for child in node.children:
            child_depth = depth + 1 if node.type in def_types else depth
            walk(child, child_depth)

    walk(tree.root_node)
    return symbols


class CodeIntelSkill(Skill):
    """
    Code intelligence skill using tree-sitter and grep-ast.

    Provides AST-aware code navigation for any supported language.
    """

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="code_intel",
            version="1.0.0",
            description=(
                "AST-aware code intelligence — understand codebases structurally. "
                "Use repo_map to get an overview of a project, find_symbol to locate "
                "definitions, and search_code for context-aware code search."
            ),
            capabilities=frozenset([Capability.FILE_READ]),
            tools=(
                ToolSpec(
                    name="repo_map",
                    description=(
                        "Generate a condensed map of a code repository showing all files "
                        "with their key symbols (classes, functions, methods). "
                        "Call this FIRST when working on a coding task to understand "
                        "the project structure. Returns ~500-2000 tokens for a typical project."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Root directory of the project to map. "
                                    "Use '.' for current working directory."
                                ),
                            },
                        },
                        "required": ["path"],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
                ToolSpec(
                    name="find_symbol",
                    description=(
                        "Find where a function, class, or method is defined in the codebase "
                        "and return the complete source code of that symbol with context. "
                        "More precise than grep — returns the full function/class body, "
                        "not just the line where the name appears."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The function, class, or method name to find.",
                            },
                            "path": {
                                "type": "string",
                                "description": "Root directory to search in. Use '.' for cwd.",
                            },
                        },
                        "required": ["name", "path"],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
                ToolSpec(
                    name="search_code",
                    description=(
                        "Search for a pattern in code with AST-aware context. "
                        "Unlike plain grep, results show the enclosing function/class "
                        "scope so you understand WHERE in the code the match occurs. "
                        "Use for finding usages, tracing call chains, locating error strings."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Text pattern or regex to search for.",
                            },
                            "path": {
                                "type": "string",
                                "description": "Root directory to search in. Use '.' for cwd.",
                            },
                        },
                        "required": ["pattern", "path"],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
            ),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name == "repo_map":
            return self._repo_map(arguments.get("path", "."))
        if tool_name == "find_symbol":
            return self._find_symbol(
                arguments.get("name", ""),
                arguments.get("path", "."),
            )
        if tool_name == "search_code":
            return self._search_code(
                arguments.get("pattern", ""),
                arguments.get("path", "."),
            )
        return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

    # ── repo_map ─────────────────────────────────────────────────────────────

    def _repo_map(self, path: str) -> ToolResult:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        files = _collect_source_files(root)
        if not files:
            return ToolResult(success=True, output="No source files found in this directory.")

        lines: list[str] = []
        file_count = 0
        symbol_count = 0

        for fp in files:
            lang = _detect_language(fp)
            if not lang:
                continue
            try:
                rel = fp.relative_to(root)
            except ValueError:
                rel = fp

            symbols = _extract_symbols(fp, lang)
            file_count += 1
            symbol_count += len(symbols)

            if symbols:
                lines.append(f"\n{rel}")
                for sym in symbols:
                    lines.append(f"  {sym}")
            else:
                lines.append(f"\n{rel}  (no extractable symbols)")

        header = f"Repository map: {file_count} files, {symbol_count} symbols\n"
        return ToolResult(success=True, output=header + "\n".join(lines))

    # ── find_symbol ──────────────────────────────────────────────────────────

    def _find_symbol(self, name: str, path: str) -> ToolResult:
        if not name:
            return ToolResult(success=False, output="", error="Symbol name is required.")

        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            return ToolResult(
                success=False, output="",
                error="tree-sitter-language-pack not installed. Run: pip install grep-ast",
            )

        files = _collect_source_files(root)
        results: list[str] = []

        for fp in files:
            lang = _detect_language(fp)
            if not lang:
                continue

            try:
                code = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Quick text check before expensive AST parse
            if name not in code:
                continue

            try:
                parser = get_parser(lang)
                tree = parser.parse(code.encode())
            except Exception:
                continue

            def_types = _DEF_TYPES.get(lang, set())
            code_lines = code.splitlines()

            def find_in_node(node):
                if node.type in def_types:
                    # Check if this definition's name matches
                    start_line = node.start_point[0]
                    end_line = node.end_point[0]
                    if start_line < len(code_lines) and name in code_lines[start_line]:
                        try:
                            rel = fp.relative_to(root)
                        except ValueError:
                            rel = fp
                        # Extract the full definition body
                        body = "\n".join(
                            f"{i+1:4d} | {code_lines[i]}"
                            for i in range(start_line, min(end_line + 1, len(code_lines)))
                        )
                        results.append(f"── {rel} (line {start_line + 1}) ──\n{body}")
                for child in node.children:
                    find_in_node(child)

            find_in_node(tree.root_node)

        if not results:
            return ToolResult(
                success=True,
                output=f"Symbol '{name}' not found in {len(files)} files under {path}",
            )

        return ToolResult(
            success=True,
            output=f"Found {len(results)} definition(s) of '{name}':\n\n" + "\n\n".join(results),
        )

    # ── search_code ──────────────────────────────────────────────────────────

    def _search_code(self, pattern: str, path: str) -> ToolResult:
        if not pattern:
            return ToolResult(success=False, output="", error="Search pattern is required.")

        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        try:
            from grep_ast import TreeContext
        except ImportError:
            return ToolResult(
                success=False, output="",
                error="grep-ast not installed. Run: pip install grep-ast",
            )

        files = _collect_source_files(root)
        results: list[str] = []
        match_count = 0

        for fp in files:
            lang = _detect_language(fp)
            if not lang:
                continue

            try:
                code = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Quick text check
            if pattern.lower() not in code.lower():
                continue

            # Find matching line numbers
            matching_lines: list[int] = []
            for i, line in enumerate(code.splitlines()):
                if pattern.lower() in line.lower():
                    matching_lines.append(i)

            if not matching_lines:
                continue

            match_count += len(matching_lines)

            # Use TreeContext to show matches with AST-aware context
            try:
                tc = TreeContext(
                    filename=str(fp),
                    code=code,
                    color=False,
                    line_number=True,
                    show_top_of_file_parent_scope=True,
                    margin=1,
                )
                tc.add_lines_of_interest(matching_lines)
                tc.add_context()
                formatted = tc.format()
                if formatted and formatted.strip():
                    try:
                        rel = fp.relative_to(root)
                    except ValueError:
                        rel = fp
                    results.append(f"── {rel} ({len(matching_lines)} matches) ──\n{formatted}")
            except Exception as e:
                # Fallback to simple grep-style output
                try:
                    rel = fp.relative_to(root)
                except ValueError:
                    rel = fp
                lines_out = []
                code_lines = code.splitlines()
                for ln in matching_lines[:10]:
                    lines_out.append(f"  {ln+1:4d} | {code_lines[ln]}")
                results.append(f"── {rel} ({len(matching_lines)} matches) ──\n" + "\n".join(lines_out))

            # Cap results to avoid huge outputs
            if len(results) >= 20:
                break

        if not results:
            return ToolResult(
                success=True,
                output=f"No matches for '{pattern}' in {len(files)} files under {path}",
            )

        header = f"Found {match_count} match(es) across {len(results)} file(s):\n\n"
        return ToolResult(success=True, output=header + "\n\n".join(results))
