"""Keep source-checkout tests independent of a surrounding desktop payload."""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_source_payload_manifest(monkeypatch):
    """The executing checkout is source, while temporary payloads remain real.

    Installed checkouts can live directly inside the operator's Hermes home.
    PM's adjacent-payload discovery would then read that home's manifest even
    after HERMES_HOME is isolated. Select the unbundled source contract for
    this checkout only; payload tests must construct their own temporary tree.
    The real-home I/O guard stays unchanged and continues to reject direct I/O.
    """
    from pm import environments

    checkout = Path(__file__).resolve().parents[2]
    read_manifest = environments._payload_manifest

    def isolated_manifest(root):
        if Path(root).resolve() == checkout:
            return None
        return read_manifest(root)

    monkeypatch.setattr(environments, "_payload_manifest", isolated_manifest)
