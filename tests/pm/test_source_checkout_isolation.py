"""Installed source tests never borrow their surrounding operator payload."""
import json
from pathlib import Path

from pm import environments


def test_executing_checkout_uses_unbundled_source_contract():
    checkout = Path(__file__).resolve().parents[2]
    assert environments.payload_venv(checkout) is None
    assert environments.payload_command_dir(checkout) is None


def test_temporary_payload_retains_real_manifest_resolution(tmp_path):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "repo": "repo", "venv": "environment",
        "runtime": {"commands": {"hermes": "bin/hermes"}},
    }), encoding="utf-8")
    assert environments.payload_venv(checkout) == tmp_path / "environment"
    assert environments.payload_command_dir(checkout) == tmp_path / "bin"
