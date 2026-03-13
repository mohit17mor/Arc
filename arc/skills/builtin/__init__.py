"""Built-in skills that ship with Arc."""

from arc.skills.builtin.filesystem import FilesystemSkill
from arc.skills.builtin.terminal import TerminalSkill
from arc.skills.builtin.browsing import BrowsingSkill
from arc.skills.builtin.worker import WorkerSkill
from arc.skills.builtin.browser_control import BrowserControlSkill
from arc.skills.builtin.planning import PlanningSkill
from arc.skills.builtin.screen_record import ScreenRecordSkill

__all__ = [
    "FilesystemSkill",
    "TerminalSkill",
    "BrowsingSkill",
    "WorkerSkill",
    "BrowserControlSkill",
    "PlanningSkill",
    "ScreenRecordSkill",
]
