"""Tests for the SkillRouter — two-tier tool selection."""

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.core.types import Capability, SkillManifest, ToolSpec
from arc.skills.base import tool, FunctionSkill, Skill
from arc.skills.manager import SkillManager
from arc.skills.router import SkillRouter, USE_SKILL_TOOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubSkill(Skill):
    """Minimal skill for testing, supports always_available flag."""

    def __init__(self, name: str, description: str, tool_names: list[str],
                 always_available: bool = False):
        self._name = name
        self._description = description
        self._tool_names = tool_names
        self._always_available = always_available

    def manifest(self) -> SkillManifest:
        tools = tuple(
            ToolSpec(
                name=tn,
                description=f"Tool {tn} from {self._name}",
                parameters={"type": "object", "properties": {}, "required": []},
            )
            for tn in self._tool_names
        )
        return SkillManifest(
            name=self._name,
            version="1.0.0",
            description=self._description,
            tools=tools,
            always_available=self._always_available,
        )

    async def execute_tool(self, tool_name, arguments):
        from arc.core.types import ToolResult
        return ToolResult(success=True, output=f"executed {tool_name}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kernel():
    return Kernel(config=ArcConfig())


@pytest.fixture
async def populated_manager(kernel):
    """Manager with 4 skills: 2 always-on, 2 on-demand."""
    manager = SkillManager(kernel)

    # Always-on
    await manager.register(
        StubSkill("terminal", "Execute shell commands", ["execute"],
                  always_available=True)
    )
    await manager.register(
        StubSkill("filesystem", "Read/write files", ["read_file", "write_file"],
                  always_available=True)
    )

    # On-demand
    await manager.register(
        StubSkill("browsing", "Search & read web pages",
                  ["web_search", "web_read", "http_get"])
    )
    await manager.register(
        StubSkill("browser_control", "Interactive browser automation",
                  ["browser_go", "browser_act", "browser_look"])
    )

    return manager


@pytest.fixture
def router(populated_manager):
    return SkillRouter(populated_manager)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_router_construction(router):
    assert router is not None
    assert router.activated_skills == set()


# ---------------------------------------------------------------------------
# get_active_tool_specs — initial state
# ---------------------------------------------------------------------------

def test_initial_specs_contain_always_on_tools(router):
    """Before any activation, specs include always-on tools + use_skill."""
    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}

    # Always-on tools present
    assert "execute" in names
    assert "read_file" in names
    assert "write_file" in names

    # On-demand tools NOT present
    assert "web_search" not in names
    assert "browser_go" not in names


def test_initial_specs_contain_use_skill(router):
    """use_skill meta-tool is present when there are on-demand skills."""
    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}
    assert USE_SKILL_TOOL in names


def test_use_skill_enum_lists_on_demand_skills(router):
    """use_skill parameter enum contains only on-demand skills."""
    specs = router.get_active_tool_specs()
    use_skill = next(s for s in specs if s.name == USE_SKILL_TOOL)
    enum_values = use_skill.parameters["properties"]["skill_name"]["enum"]

    assert "browsing" in enum_values
    assert "browser_control" in enum_values
    # always-on skills should NOT appear in the enum
    assert "terminal" not in enum_values
    assert "filesystem" not in enum_values


def test_use_skill_description_contains_skill_info(router):
    """use_skill description includes skill names and their tools."""
    specs = router.get_active_tool_specs()
    use_skill = next(s for s in specs if s.name == USE_SKILL_TOOL)
    desc = use_skill.description

    assert "browsing" in desc
    assert "web_search" in desc
    assert "browser_control" in desc
    assert "browser_go" in desc


# ---------------------------------------------------------------------------
# Token savings — the whole point
# ---------------------------------------------------------------------------

def test_initial_spec_count_is_smaller_than_total(router, populated_manager):
    """Router should send fewer tool specs than the flat list."""
    flat_count = len(populated_manager.get_all_tool_specs())
    routed_count = len(router.get_active_tool_specs())

    # Flat: 9 tools total. Routed: 3 always-on + 1 use_skill = 4
    assert routed_count < flat_count
    assert routed_count == 4  # execute, read_file, write_file, use_skill


# ---------------------------------------------------------------------------
# activate()
# ---------------------------------------------------------------------------

def test_activate_adds_tools(router):
    """Activating a skill adds its tools to the next call."""
    msg = router.activate("browsing")
    assert "activated" in msg.lower()
    assert "web_search" in msg

    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}
    assert "web_search" in names
    assert "web_read" in names
    assert "http_get" in names


def test_activate_removes_skill_from_use_skill_enum(router):
    """After activation, the skill vanishes from use_skill enum."""
    router.activate("browsing")
    specs = router.get_active_tool_specs()
    use_skill = next(s for s in specs if s.name == USE_SKILL_TOOL)
    enum_values = use_skill.parameters["properties"]["skill_name"]["enum"]

    # browsing activated → no longer in enum
    assert "browsing" not in enum_values
    # browser_control still available
    assert "browser_control" in enum_values


def test_activate_all_on_demand_removes_use_skill(router):
    """When all on-demand skills are activated, use_skill disappears."""
    router.activate("browsing")
    router.activate("browser_control")

    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}
    assert USE_SKILL_TOOL not in names

    # All tools now present
    assert "web_search" in names
    assert "browser_go" in names


def test_activate_unknown_skill_returns_error(router):
    """Activating a non-existent skill returns an error message."""
    msg = router.activate("nonexistent")
    assert "unknown" in msg.lower()
    assert "browsing" in msg  # suggests available skills


def test_activate_always_available_returns_info(router):
    """Activating an always-on skill tells the LLM it's already active."""
    msg = router.activate("terminal")
    assert "already available" in msg.lower()


def test_activate_already_activated_returns_info(router):
    """Activating a skill twice returns an informational message."""
    router.activate("browsing")
    msg = router.activate("browsing")
    assert "already activated" in msg.lower()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_activations(router):
    """reset() removes all activated skills."""
    router.activate("browsing")
    router.activate("browser_control")
    assert len(router.activated_skills) == 2

    router.reset()
    assert router.activated_skills == set()

    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}
    assert "web_search" not in names
    assert "browser_go" not in names


# ---------------------------------------------------------------------------
# excluded_skills
# ---------------------------------------------------------------------------

def test_excluded_skills_hidden_from_router(populated_manager):
    """Excluded skills don't appear in use_skill or as always-on."""
    router = SkillRouter(populated_manager, excluded_skills=frozenset({"terminal"}))
    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}

    assert "execute" not in names


def test_excluded_on_demand_skill_not_in_menu(populated_manager):
    """Excluded on-demand skill doesn't appear in use_skill."""
    router = SkillRouter(
        populated_manager, excluded_skills=frozenset({"browsing"})
    )
    specs = router.get_active_tool_specs()
    use_skill = next(s for s in specs if s.name == USE_SKILL_TOOL)
    enum_values = use_skill.parameters["properties"]["skill_name"]["enum"]

    assert "browsing" not in enum_values
    assert "browser_control" in enum_values


def test_excluded_skill_cannot_be_activated(populated_manager):
    """Trying to activate an excluded skill returns an error."""
    router = SkillRouter(
        populated_manager, excluded_skills=frozenset({"browsing"})
    )
    msg = router.activate("browsing")
    assert "unknown" in msg.lower()


# ---------------------------------------------------------------------------
# is_use_skill_call
# ---------------------------------------------------------------------------

def test_is_use_skill_call(router):
    assert router.is_use_skill_call(USE_SKILL_TOOL) is True
    assert router.is_use_skill_call("execute") is False
    assert router.is_use_skill_call("web_search") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_skills_always_available_no_use_skill(kernel):
    """If every skill is always_available, use_skill doesn't appear."""
    manager = SkillManager(kernel)
    await manager.register(
        StubSkill("a", "Skill A", ["tool_a"], always_available=True)
    )
    await manager.register(
        StubSkill("b", "Skill B", ["tool_b"], always_available=True)
    )

    router = SkillRouter(manager)
    specs = router.get_active_tool_specs()
    names = {s.name for s in specs}

    assert "tool_a" in names
    assert "tool_b" in names
    assert USE_SKILL_TOOL not in names


@pytest.mark.asyncio
async def test_no_skills_registered(kernel):
    """Router with empty manager returns no specs."""
    manager = SkillManager(kernel)
    router = SkillRouter(manager)
    specs = router.get_active_tool_specs()
    assert specs == []


@pytest.mark.asyncio
async def test_all_skills_excluded(populated_manager):
    """Excluding every skill returns no specs."""
    all_names = frozenset(populated_manager.skill_names)
    router = SkillRouter(populated_manager, excluded_skills=all_names)
    specs = router.get_active_tool_specs()
    assert specs == []
