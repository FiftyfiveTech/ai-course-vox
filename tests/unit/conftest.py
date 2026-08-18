"""Fixtures every unit test needs. Nothing here touches the network, the mic, or real weights.

The two log redirects are autouse rather than opt-in on purpose: a test that forgets them appends
to the real runs/*.jsonl, and a gate that reads those logs would then be reading test noise. Both
work because `telemetry._append` looks the path up at call time — see the comment on it.
"""
import pytest

from src import config, telemetry


@pytest.fixture(autouse=True)
def calls_log(tmp_path, monkeypatch):
    """Never append to the real runs/calls.jsonl from a test."""
    path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(telemetry, "CALLS_LOG", path)
    return path


@pytest.fixture(autouse=True)
def turns_log(tmp_path, monkeypatch):
    """Never append to the real runs/turns.jsonl from a test."""
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(telemetry, "TURNS_LOG", path)
    return path


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch):
    """A VOX_*_MODEL left in the shell must not change what the default-resolution tests see."""
    for env in config.STAGE_ENV.values():
        monkeypatch.delenv(env, raising=False)
