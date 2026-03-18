import asyncio

import pytest

from arc.core.run_control import RunCancelledError, RunControlAction, RunControlManager, RunStatus


@pytest.mark.asyncio
async def test_register_run_and_list_active_runs():
    manager = RunControlManager()

    handle = manager.start_run(kind="agent", source="main", metadata={"label": "chat"})

    snapshot = manager.get_run(handle.run_id)
    assert snapshot is not None
    assert snapshot.kind == "agent"
    assert snapshot.source == "main"
    assert snapshot.status == RunStatus.RUNNING

    active = manager.list_runs(active_only=True)
    assert [run.run_id for run in active] == [handle.run_id]


@pytest.mark.asyncio
async def test_cancelled_run_fails_checkpoint_and_becomes_cancelled():
    manager = RunControlManager()
    handle = manager.start_run(kind="agent", source="main")

    changed = manager.request(handle.run_id, RunControlAction.CANCEL)
    assert changed is True

    with pytest.raises(RunCancelledError):
        await handle.checkpoint()

    snapshot = manager.get_run(handle.run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.CANCELLED
    assert snapshot.requested_action == RunControlAction.CANCEL


@pytest.mark.asyncio
async def test_finish_run_marks_completed_when_not_cancelled():
    manager = RunControlManager()
    handle = manager.start_run(kind="workflow", source="workflow")

    await handle.finish_completed()

    snapshot = manager.get_run(handle.run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_pause_blocks_checkpoint_until_resume():
    manager = RunControlManager()
    handle = manager.start_run(kind="agent", source="main")

    assert manager.request(handle.run_id, RunControlAction.PAUSE) is True

    task = asyncio.create_task(handle.checkpoint())
    await asyncio.sleep(0.01)
    assert not task.done()

    assert manager.request(handle.run_id, RunControlAction.RESUME) is True
    await asyncio.wait_for(task, timeout=0.2)

    snapshot = manager.get_run(handle.run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.RUNNING
