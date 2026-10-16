"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

import os
import signal
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and the timeout path signals it."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1]
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    conn = board
    killed = []
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("clock_shift", [-5, 5])
def test_macos_clock_adjustment_does_not_reap_or_duplicate_live_worker(board, monkeypatch, clock_shift):
    """A long-lived dispatcher's boot-clock cache differs from a fresh worker's."""
    import psutil
    import psutil._psosx as osx

    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    tid = _claimed_running(board, pid=pid, started_at=recorded)
    wall_start = psutil.Process(pid).create_time()
    monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.INIT_BOOT_TIME + clock_shift)
    assert abs(psutil.Process(pid).create_time() - wall_start) >= 1
    assert kbd._process_fingerprint(pid) == recorded
    assert kbd._worker_alive(pid, recorded)
    assert kbd.detect_crashed_workers(board) == []
    assert kb.release_stale_claims(board) == 0
    assert kb.get_task(board, tid).status == "running"


@pytest.mark.platforms("macos")
def test_macos_fingerprint_agrees_in_fresh_interpreter():
    import subprocess
    import sys

    pid = os.getpid()
    actual = subprocess.check_output(
        [sys.executable, "-c", "from hermes_cli.kanban_db_dispatch import _process_fingerprint; "
         f"print(_process_fingerprint({pid}))"], text=True, timeout=20,
    ).strip()
    assert actual == kbd._process_fingerprint(pid)
    assert "|darwin-v1:" in actual


@pytest.mark.platforms("macos")
def test_macos_recycled_identity_is_never_signalled(board, monkeypatch):
    from types import SimpleNamespace
    import psutil

    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    tid = _claimed_running(board, pid=pid, started_at=recorded, max_runtime=1)
    start = psutil.Process(pid)._ident[1]
    monkeypatch.setattr(psutil, "Process", lambda p: SimpleNamespace(_ident=(p, start + 0.001)))
    assert not kbd._worker_alive(pid, recorded)
    killed = []
    assert tid in kbd.enforce_max_runtime(board, signal_fn=lambda p, s: killed.append((p, s)))
    assert killed == []


@pytest.mark.platforms("macos")
def test_macos_legacy_fingerprints_keep_the_legacy_comparator():
    pid = os.getpid()
    legacy = kbd._legacy_process_fingerprint(pid)
    assert "|darwin-v1:" not in legacy
    assert kbd._worker_alive(pid, legacy)
    assert kbd._pid_recycled(pid, "foreign-boot|" + legacy.split("|", 1)[1])


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("identity", [None, (1, 42), (os.getpid(), float("nan")), (os.getpid(), float("inf"))])
def test_macos_unavailable_or_invalid_identity_fails_closed(monkeypatch, identity):
    from types import SimpleNamespace
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda p: SimpleNamespace(_ident=identity))
    assert kbd._process_fingerprint(os.getpid()) is None


def test_temporarily_unreadable_identity_holds_claim_and_refuses_signals(board, monkeypatch):
    pid = os.getpid()
    recorded = kbd._process_fingerprint(pid)
    tid = _claimed_running(board, pid=pid, started_at=recorded, max_runtime=1)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda p: None)
    killed = []
    signal_fn = lambda p, s: killed.append((p, s))  # noqa: E731
    assert kbd._worker_alive(pid, recorded)
    assert kbd.detect_crashed_workers(board) == []
    assert kbd.enforce_max_runtime(board, signal_fn=signal_fn) == []
    assert kb.release_stale_claims(board, signal_fn=signal_fn) == 0
    task = kb.get_task(board, tid)
    result = kbd._terminate_reclaimed_worker(pid, task.claim_lock, started_at=recorded, signal_fn=signal_fn)
    assert result["signal_refused"] and not result["terminated"]
    assert task.status == "running" and killed == []


@pytest.mark.platforms("posix")
def test_timeout_terminates_only_owned_child_before_releasing_claim(board):
    import subprocess
    import sys

    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as child:
        try:
            recorded = kbd._process_fingerprint(child.pid)
            assert recorded is not None
            tid = _claimed_running(board, pid=child.pid, started_at=recorded, max_runtime=1)
            assert tid in kbd.enforce_max_runtime(board)
            assert child.wait(timeout=5) == -signal.SIGTERM
            assert kb.get_task(board, tid).status != "running"
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


def test_timeout_keeps_claim_if_worker_survives_signals(board, monkeypatch):
    pid = os.getpid()
    tid = _claimed_running(board, pid=pid, started_at=kbd._process_fingerprint(pid), max_runtime=1)
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *args: False)
    killed = []
    assert kbd.enforce_max_runtime(board, signal_fn=lambda p, s: killed.append((p, s))) == []
    assert killed and kb.get_task(board, tid).status == "running"


def test_identity_lost_after_sigterm_never_authorizes_sigkill_or_retry(board, monkeypatch):
    pid = os.getpid()
    tid = _claimed_running(board, pid=pid, started_at=kbd._process_fingerprint(pid), max_runtime=1)
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *args: False)
    killed = []

    def signal_fn(p, s):
        killed.append((p, s))
        monkeypatch.setattr(kbd, "_process_fingerprint", lambda p: None)

    assert kbd.enforce_max_runtime(board, signal_fn=signal_fn) == []
    assert killed == [(pid, signal.SIGTERM)]
    assert kb.get_task(board, tid).status == "running"
