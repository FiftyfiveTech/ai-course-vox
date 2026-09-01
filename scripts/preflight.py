"""Ask each provider what it still serves, and check the arm table against the answer.

    uv run python scripts/preflight.py              every remote arm, catalogue only
    uv run python scripts/preflight.py --call       and one real 1-token call per arm
    uv run python scripts/preflight.py --stage llm

Why this exists. On 2026-08-26 NIM retired `meta-llama/Llama-3.1-8B-Instruct`, the default LLM arm.
Nothing in the repo noticed until `make demo` reached the LLM stage of a live turn and died on a 410
whose body said "has reached its end of life". The arm table is a set of claims about other people's
catalogues, and those claims expire without telling us.

So this asks. One `GET /v1/models` per remote provider — no tokens, no spend, two HTTP calls for the
whole table — and compares `arm.provider_model` against the ids that came back. Exits non-zero if
any arm is missing, so it can sit in front of a session or in the gating table and answer "is the
table still true?" with a status code.

`--call` is the second half of the same question, because a catalogue listing is necessary and not
sufficient: NIM lists 82 models and answers 404 "Not found for account" for most of them, so being
on the list does not mean this key may use it. A 1-token call is the only thing that settles it.
Kept behind a flag because it does spend tokens and can trip a free tier's pace limit — the
catalogue pass is the one that is free to run every time.

Not `check_arms.py`. That script calls every arm for real, loads torch and the local weights, and
takes minutes — it is VOX-006's evidence. This one is meant to run before a demo, so it imports
httpx and nothing else.
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARMS                                                 # noqa: E402

TIMEOUT_S = 15


def catalogue(arm):
    """-> (set of model ids the provider serves, error string). Exactly one is truthy.

    Keyed by api_base rather than by provider name so two providers on one base — or one provider
    reached under two names — ask once. The cache matters: without it a five-arm stage on two
    providers makes five listing calls to answer one question.
    """
    if arm.api_base in _CACHE:
        return _CACHE[arm.api_base]
    try:
        key = arm.key()
    except RuntimeError as e:
        # A missing credential is a STOP-and-ask under CLAUDE.md, not something to report as a dead
        # model — the arm may be perfectly alive and simply unreachable from this machine.
        result = (None, f"no credential: {str(e).splitlines()[0]}")
    else:
        try:
            r = httpx.get(f"{arm.api_base}/models", timeout=TIMEOUT_S,
                          headers={"Authorization": f"Bearer {key}"} if key else {})
            r.raise_for_status()
            result = ({d["id"] for d in r.json().get("data", [])}, None)
        except Exception as e:
            result = (None, f"{type(e).__name__}: {str(e)[:80]}")
    _CACHE[arm.api_base] = result
    return result


_CACHE = {}


# Which backends `probe` knows how to call for one token. An STT arm speaks /audio/transcriptions
# and answers 400 "does not support chat completions" to a chat body, which reads like a broken arm
# and is only a broken probe — so those say so instead of guessing.
CHAT_BACKENDS = {"openai-chat", "ollama-chat"}


def probe(arm):
    """-> a one-line verdict from one real minimal call. Only run under --call."""
    if arm.backend not in CHAT_BACKENDS:
        return f"not probed ({arm.backend} needs a real payload; catalogue is the check)"
    body = {"model": arm.provider_model, "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(arm.extra.get("request") or {})
    try:
        r = httpx.post(f"{arm.api_base}/chat/completions", json=body, timeout=TIMEOUT_S,
                       headers=arm.auth_headers({"Content-Type": "application/json"}))
    except Exception as e:
        return f"{type(e).__name__}"
    if r.status_code == 200:
        return "answers"
    from src.errors import _detail
    return f"HTTP {r.status_code} {_detail(r)[:60]}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(ARMS), help="one stage instead of all of them")
    ap.add_argument("--call", action="store_true",
                    help="also make one 1-token call per arm (spends tokens; can hit a pace limit)")
    args = ap.parse_args(argv)

    stages = [args.stage] if args.stage else list(ARMS)
    rows, missing = [], []
    for stage in stages:
        for arm in ARMS[stage]:
            # Local arms make no claim about anyone's catalogue. `make setup` and the loop's own
            # warm-up are what check those, and reporting them here as "skipped" would put eleven
            # uninteresting rows in front of the four that matter.
            if arm.local:
                continue
            served, err = catalogue(arm)
            if err is not None:
                verdict = f"?  {err}"
            elif arm.provider_model in served:
                verdict = "on catalogue"
            else:
                verdict = f"GONE — not among the {len(served)} models {arm.provider} serves"
                missing.append(arm)
            if args.call and err is None:
                verdict += f"  |  {probe(arm)}"
            rows.append((stage, arm, verdict))

    width = max((len(a.alias) for _, a, _ in rows), default=5)
    for stage, arm, verdict in rows:
        print(f"{stage:<5} {arm.alias:<{width}}  {arm.provider_model:<34} {verdict}")

    if missing:
        print(f"\n{len(missing)} arm(s) name a model their provider no longer serves. Re-point them "
              f"in src/config.py:")
        for arm in missing:
            print(f"  {arm.id}  (alias {arm.alias})")
        print("\nWhat each provider does still serve is the listing above's denominator; "
              "`--call` says which of those this key may actually use.")
        return 1
    print(f"\n{len(rows)} remote arm(s) checked, all still on their provider's catalogue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
