"""Fixtures every unit test needs. Nothing here touches the network, the mic, or real weights.

The two log redirects are autouse rather than opt-in on purpose: a test that forgets them appends
to the real runs/*.jsonl, and a gate that reads those logs would then be reading test noise. Both
work because `telemetry._append` looks the path up at call time — see the comment on it.
"""
import pytest

from schemas.turn_state import TurnState
from src import config, cooldown, embeddings, retrieval, sources, state, telemetry


def fake_state(reply, intent="capture", next_action="reply", confidence=0.9, **entities):
    """A TurnState as `state.build` would have returned it, without the call (VOX-019).

    Here rather than in one test file because every test that drives `loop.one_turn` down the
    un-retrieved path now needs one — that path is the extractor since the VOX-019/020 merge.
    """
    return TurnState(intent=intent, entities=entities, confidence=confidence,
                     next_action=next_action, reply=reply)


@pytest.fixture(autouse=True)
def no_real_state_extraction(monkeypatch):
    """No unit test may reach the real structured-state extractor.

    `state.build` posts to the provider directly — it is the one LLM path in the repo with no arm
    indirection and so no offline fallback, which means a test that forgets to stub it does not
    fail: it makes a real call, spends real free tier, and passes or fails on what a remote model
    happened to say. It did, on the three loop tests that were still stubbing `nlu.reply` after the
    live plain path became this. A test that wants the path stubs it with `fake_state`.
    """
    def refuse(transcript, turn_id, model_id=None):
        raise AssertionError(
            "a unit test reached the real state.build — stub it: "
            "monkeypatch.setattr(loop.state, 'build', lambda *a, **kw: fake_state('...'))")

    monkeypatch.setattr(state, "build", refuse)


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
def no_cooldowns():
    """A 429 in one test must not leave an arm parked for the next.

    Autouse for the same reason the log redirects are: the registry is process-global, so a test
    that forgets to clear it does not fail — it silently makes a *later* test skip the remote arm
    and pass for the wrong reason. Cleared on the way in and the way out.
    """
    cooldown.clear()
    yield
    cooldown.clear()


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch):
    """A VOX_*_MODEL left in the shell must not change what the default-resolution tests see."""
    for env in config.STAGE_ENV.values():
        monkeypatch.delenv(env, raising=False)


# The code defaults, as `src/config.py` writes them — the second argument to each `os.environ.get`.
# Duplicated here on purpose: the whole point is a value the environment cannot reach, so it cannot
# be read back out of a module that already read the environment. `test_session.py` has a test that
# fails if these two places disagree.
SESSION_DEFAULTS = {"SESSION_MINUTES": 3.0, "SESSION_QUIET_LIMIT": 2}


@pytest.fixture(autouse=True)
def no_ambient_tunables(monkeypatch):
    """A VOX_* tunable in `.env` must not change what a unit test measures.

    `no_env_override` above does this for the model names by deleting the variables, which works
    because arms are resolved per call. It cannot work for these: `config.py` reads them once at
    import, into module constants that `src/loop.py` then imports by name — so by the time a fixture
    runs, `VOX_SESSION_QUIET_LIMIT=1` is already two constants deep and deleting the variable
    changes nothing.

    Not hypothetical. `.env` on the demo machine carries a demo profile (`VOX_SESSION_QUIET_LIMIT=1`
    so one silent listen ends a session instead of two, for a demo where a muted headset would
    otherwise cost a minute of dead air), and it failed three tests in this file — a suite whose
    result depended on whose machine it ran on. `.env` is gitignored and per-machine and the profile
    is correct; a test that only passes without one is the bug.
    """
    from src import loop                    # imported here: this file must not need the mic modules
    for name, default in SESSION_DEFAULTS.items():
        monkeypatch.setattr(config, name, default)
        monkeypatch.setattr(loop, name, default)


@pytest.fixture(autouse=True)
def chunks_file(tmp_path, monkeypatch):
    """Never read the real runs/chunks.jsonl from a test.

    Autouse for the same reason the log redirects are, in the other direction: `retrieval.retrieve()`
    with no index falls back to config.CHUNKS_FILE, and on this machine that file has a real 215-chunk
    corpus in it. A test that forgot to pass its own index would not fail — it would pass, here, and
    only here. Pointed at a path that does not exist, so the fallback raises instead.
    """
    path = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(config, "CHUNKS_FILE", path)
    monkeypatch.setattr(sources, "CHUNKS_FILE", path)
    monkeypatch.setattr(retrieval, "_INDEX", None)
    return path


@pytest.fixture(autouse=True)
def no_real_encoder(tmp_path, monkeypatch):
    """No unit test may load the sentence encoder or write the real vector cache.

    Autouse and off-by-default for the same reason `chunks_file` is: `retrieval.build()` reads
    `HYBRID_RETRIEVAL` from config, and on a developer machine that is on — so a test that reached
    the default path would quietly download ~130 MB of weights on a cold cache and then load them,
    turning a 9-second suite into a network-dependent one. It did, once, which is why this exists.

    A test that wants the dense half asks for it explicitly and supplies a fake encoder — see
    tests/unit/test_hybrid.py. Nothing here can reach real weights by forgetting something.
    """
    monkeypatch.setattr(config, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(retrieval, "HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    monkeypatch.setattr(retrieval, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    monkeypatch.setattr(embeddings, "_LOADED", {})
