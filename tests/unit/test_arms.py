"""VOX-006: the arms are a table, and every row in it is reachable and free.

No network, no mic, no weights. Two kinds of assertion here:

  the registry     is coherent — every arm has an implemented backend, a unique name, and a
                   provider on the free-tier list. These are the mistakes that otherwise surface
                   as a failure hours later on a first call, or worse, as a bill.
  the dispatch     honours the name it was given, and the call record carries the HF repo id.
                   That is the ticket's verification criterion, checked with a fake backend so it
                   holds without spending a request.

The measured latencies for each arm come from `make arms`; nothing here claims a number.
"""
import json

import pytest

from src import arms, config, telemetry
from src.config import ARMS, LLM_ARMS, STT_ARMS, TTS_ARMS, resolve

ALL_ARMS = [(stage, arm) for stage, stage_arms in ARMS.items() for arm in stage_arms]
ARM_IDS = [f"{stage}:{arm.alias}" for stage, arm in ALL_ARMS]


# `calls_log` and `no_env_override` are autouse in tests/unit/conftest.py.

# --- the criterion -------------------------------------------------------------------------

def test_the_stages_have_the_arms_the_ticket_asks_for():
    assert len(STT_ARMS) >= 3, "VOX-006 wants 3 STT arms"
    assert len(TTS_ARMS) >= 2, "VOX-006 wants 2 TTS arms"
    assert len(LLM_ARMS) >= 2, "VOX-006 wants 2 LLM arms"


def test_local_and_hosted_are_both_selectable():
    """"Local and hosted both callable by flag" — not one stage local and the rest hosted."""
    providers = {arm.provider for _, arm in ALL_ARMS}
    assert "local" in providers
    assert providers - {"local"}, "no hosted arm registered"
    assert {a.provider for a in STT_ARMS} > {"local"}, "stt must offer both, it is the swap the gate shows"


# --- the registry is coherent --------------------------------------------------------------

@pytest.mark.parametrize("stage,arm", ALL_ARMS, ids=ARM_IDS)
def test_every_arm_runs_on_a_free_tier(stage, arm):
    """The zero-spend constraint, checked at import instead of on a first call."""
    assert arm.provider in telemetry.FREE_TIERS, (
        f"{arm.id} names a provider that is not a known free tier — stop and ask")


@pytest.mark.parametrize("stage,arm", ALL_ARMS, ids=ARM_IDS)
def test_every_arm_has_an_implemented_backend(stage, arm):
    """A row with no adapter is an arm that only looks callable."""
    assert arm.backend in arms._MODULES[stage].BACKENDS


@pytest.mark.parametrize("stage,arm", ALL_ARMS, ids=ARM_IDS)
def test_a_remote_arm_names_its_credential_and_a_local_one_needs_none(stage, arm):
    """The rule is about credentials, not about HTTP.

    This was phrased as `provider == "local"` implies no api_base, which held only while every local
    arm was loaded in-process. The ollama arm broke that: it runs on this machine and still speaks
    HTTP, to a daemon on localhost. What has not changed — and is the part worth asserting, because
    it is the zero-spend constraint in miniature — is that nothing running locally costs a key.
    """
    if arm.local:
        assert arm.key_env is None, f"{arm.id} runs locally and must not need a credential"
        assert arm.key() is None
    else:
        assert arm.api_base and arm.key_env

    if arm.provider == "local":
        assert arm.api_base is None, "an in-process arm has nothing to connect to"


@pytest.mark.parametrize("stage", tuple(ARMS))
def test_ids_and_aliases_are_unique_per_stage(stage):
    stage_arms = ARMS[stage]
    assert len({a.id for a in stage_arms}) == len(stage_arms)
    assert len({a.alias for a in stage_arms}) == len(stage_arms)


def test_every_tts_arm_declares_its_sample_rate():
    """Kokoro is 24 kHz and SpeechT5 16 kHz; playback cannot assume one of them."""
    for arm in TTS_ARMS:
        assert arm.extra["sample_rate"] > 0


# --- resolution ----------------------------------------------------------------------------

def test_none_resolves_to_the_stage_default():
    assert resolve("stt") is STT_ARMS[0]
    assert resolve("llm") is LLM_ARMS[0]
    assert resolve("tts") is TTS_ARMS[0]


@pytest.mark.parametrize("stage,arm", ALL_ARMS, ids=ARM_IDS)
def test_an_arm_resolves_by_full_id_and_by_alias(stage, arm):
    assert resolve(stage, arm.id) is arm
    assert resolve(stage, arm.alias) is arm


def test_an_unambiguous_bare_repo_id_resolves():
    """A repo id is what CLAUDE.md says we speak; it should be enough when only one arm serves it."""
    assert resolve("tts", "microsoft/speecht5_tts").alias == "speecht5"


def test_an_ambiguous_bare_repo_id_is_refused_and_names_the_candidates(monkeypatch):
    twin = config.Arm(repo_id=LLM_ARMS[0].repo_id, provider="groq", provider_model="x",
                      backend="openai-chat", alias="twin", api_base="https://x", key_env="GROQ_API_KEY")
    monkeypatch.setitem(config.ARMS, "llm", LLM_ARMS + (twin,))
    with pytest.raises(RuntimeError) as e:
        resolve("llm", LLM_ARMS[0].repo_id)
    assert LLM_ARMS[0].id in str(e.value) and twin.id in str(e.value)


def test_an_unknown_model_id_lists_the_known_ones():
    with pytest.raises(RuntimeError) as e:
        resolve("stt", "openai/whisper-does-not-exist")
    for arm in STT_ARMS:
        assert arm.id in str(e.value)


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError):
        resolve("asr", None)


def test_the_env_var_sets_the_default_and_the_flag_still_wins(monkeypatch):
    monkeypatch.setenv("VOX_STT_MODEL", STT_ARMS[-1].alias)
    assert resolve("stt") is STT_ARMS[-1]
    assert resolve("stt", STT_ARMS[0].alias) is STT_ARMS[0]


# --- dispatch and the log ------------------------------------------------------------------

def fake_backend(seen):
    def fn(arm, payload, rec):
        seen.append((arm, payload))
        rec["chars"] = 7
        return "faked"
    return fn


def test_dispatch_runs_the_backend_the_named_arm_asks_for(monkeypatch):
    seen = []
    arm = STT_ARMS[2]                      # openai/whisper-base@local — a different backend
    monkeypatch.setitem(arms._MODULES["stt"].BACKENDS, arm.backend, fake_backend(seen))
    assert arms.stt([0.0] * 16, arm.alias, turn_id="t1") == "faked"
    assert seen[0][0] is arm


def test_the_call_record_carries_the_hf_repo_id_and_zero_cost(monkeypatch, calls_log):
    """The ticket's verification: the model id shows up in the log, per arm."""
    for arm in STT_ARMS:
        monkeypatch.setitem(arms._MODULES["stt"].BACKENDS, arm.backend, fake_backend([]))
        arms.stt([0.0] * 16, arm.id, turn_id=f"turn-{arm.alias}")

    logged = [json.loads(x) for x in calls_log.read_text(encoding="utf-8").splitlines()]
    assert [r["model_id"] for r in logged] == [a.repo_id for a in STT_ARMS]
    assert all(r["provider"] == a.provider for r, a in zip(logged, STT_ARMS))
    assert all(r["cost_usd"] == 0.0 and r["ok"] and r["stage"] == "stt" for r in logged)


def test_a_backend_that_raises_is_logged_before_it_propagates(monkeypatch, calls_log):
    def boom(arm, payload, rec):
        raise TimeoutError("provider took too long")
    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, TTS_ARMS[0].backend, boom)
    with pytest.raises(TimeoutError):
        arms.tts("hello", TTS_ARMS[0].id, turn_id="t2")

    rec = json.loads(calls_log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["ok"] is False and "TimeoutError" in rec["error"]
    assert rec["model_id"] == TTS_ARMS[0].repo_id


def test_synthesis_returns_the_rate_it_actually_produced(monkeypatch):
    arm = TTS_ARMS[-1]
    monkeypatch.setitem(arms._MODULES["tts"].BACKENDS, arm.backend,
                        lambda a, text, rec: [0.0] * 8)
    speech = arms.tts("hello", arm.id, turn_id="t3")
    assert speech.sample_rate == arm.extra["sample_rate"]
    assert speech.sample_rate != TTS_ARMS[0].extra["sample_rate"], "the two arms must differ here"


def test_warm_is_a_noop_for_a_hosted_arm():
    """Nothing to load and nothing to raise — a hosted arm must not need a loader entry."""
    assert arms.warm("llm", LLM_ARMS[0].id) is LLM_ARMS[0]


def test_available_lists_every_arm():
    assert len(arms.available()) == sum(len(v) for v in ARMS.values())
    assert arms.available("tts") == list(TTS_ARMS)
