"""Clock-stable Kanban worker identities; unknown identity never grants signals."""

from __future__ import annotations

import math
import sys
from typing import Optional


def _legacy_process_fingerprint(pid: int) -> Optional[str]:
    """Original boot witness plus wall-clock/start-tick identity for old rows."""
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time

    start = get_process_start_time(int(pid))
    if start is None:
        return None
    return f"{current_instantiation_epoch()}|{start}"


def _process_fingerprint(pid: int) -> Optional[str]:
    """Use psutil's own clock-stable PID-reuse identity on macOS.

    Public create_time() adjusts against an interpreter-local boot-clock cache;
    a fresh worker and old dispatcher can disagree about the SAME process.
    psutil 7.x uses its monotonic _ident for its own signal/PID-reuse guard.
    Preserve its full precision and version the representation. An unsupported
    or unreadable private interface fails capture closed, never falls back to
    adjusted wall time or a relaxed signalling tolerance.
    """
    if sys.platform != "darwin":
        return _legacy_process_fingerprint(pid)
    try:
        import psutil
    except ImportError:
        return None
    from gateway.drain_control import current_instantiation_epoch

    try:
        owner_pid, start = psutil.Process(int(pid))._ident
        start = float(start)
        if owner_pid != int(pid) or not math.isfinite(start) or start <= 0:
            return None
        return f"{current_instantiation_epoch()}|darwin-v1:{start.hex()}"
    except (psutil.Error, AttributeError, TypeError, ValueError, OverflowError):
        return None


def _worker_identity_matches(pid: Optional[int], started_at) -> Optional[bool]:
    """Exact match, proven mismatch, or unknown. Unknown holds claims, never signals."""
    # Preserve the dispatcher facade's test/compatibility seams after extraction.
    from hermes_cli import kanban_db_dispatch as dispatch

    if started_at is None or not pid:
        return True
    if started_at == dispatch.UNVERIFIED_WORKER_FINGERPRINT:
        return None
    if isinstance(started_at, str) and "|" in started_at:
        current = dispatch._process_fingerprint(int(pid))
        if (sys.platform == "darwin" and current is not None
                and "|darwin-v1:" in current and "|darwin-v1:" not in started_at):
            current = dispatch._legacy_process_fingerprint(int(pid))
        return None if current is None else current == started_at
    from gateway.status import _start_times_agree
    from gateway.status import get_process_start_time

    current = get_process_start_time(int(pid))
    if current is None:
        return None
    try:
        return _start_times_agree(current, started_at)
    except (TypeError, ValueError):
        return False
