"""The sentence encoder behind the dense half of retrieval — one stage module, like stt and tts.

    arms.embed(["some text", ...], turn_id=...) -> (n, dim) float32, L2-normalised

Same shape as `src/stt.py` and `src/tts.py`: a `BACKENDS` table of `fn(arm, payload, rec)` adapters
and a `LOADERS` table so `arms.warm()` can pull the weights in before anything is timed. Which
encoder runs is a row in `config.EMBED_ARMS` and a `--embed` flag, so swapping BGE for MiniLM is a
config change here exactly as swapping Kokoro for Piper is.

It is a model, so it goes through `arms.embed()` and therefore through the cost logger — which is
the one thing `src/retrieval.py` was previously able to say it did not need. BM25 is arithmetic and
had no provider to name; an encoder has weights, a repo id and a latency, and pretending otherwise
would leave the only unlogged model call in the codebase sitting inside the turn.

Three decisions worth knowing before reading the code:

**Pooling and the query prefix come off the arm, not from a default here.** BGE was trained with CLS
pooling and an asymmetric instruction on the query side; MiniLM with mean pooling and no
instruction. Getting that wrong does not degrade a model slightly — it asks for vectors from a
space the model was never trained to put anything in, and the failure is silent because cosines
between garbage are still numbers between -1 and 1. So `pooling` and `query_prefix` live in
`config.EMBED_ARMS` beside the repo id, and this module reads them rather than deciding them.

**Documents and queries are encoded differently, and the difference is one flag.** `is_query=True`
prepends the arm's prefix; chunks never get it. With BGE that asymmetry is the model's design. With
MiniLM the prefix is empty and the two paths are identical, which is why the flag is a parameter and
not two functions.

**Vectors are L2-normalised on the way out, once.** A cosine between unit vectors is a dot product,
so normalising here makes every downstream similarity a matrix multiply and removes any chance of a
caller comparing a normalised vector with a raw one. `DENSE_SCORE_FLOOR` is a cosine, and it is only
a cosine if this holds.
"""
import numpy as np
import torch

from src.config import EMBED_BATCH

# One loaded encoder per repo id, for the same reason `nlu.load_ollama` keeps the model resident:
# a from_pretrained inside a turn is ~1 s of model load charged to the retrieval stage.
_LOADED = {}


def _load(arm):
    """-> (tokenizer, model) for `arm`, loaded once per process and kept."""
    if arm.repo_id not in _LOADED:
        from transformers import AutoModel, AutoTokenizer      # local: ~1 s of imports
        tok = AutoTokenizer.from_pretrained(arm.provider_model)
        model = AutoModel.from_pretrained(arm.provider_model)
        model.eval()
        _LOADED[arm.repo_id] = (tok, model)
    return _LOADED[arm.repo_id]


def load_encoder(arm):
    """`arms.warm()`'s hook: pull the weights before anything is timed. -> the loaded model."""
    return _load(arm)[1]


def fetch_encoder(arm=None):
    """Download the encoder weights into the HF cache. Used by `make setup`, never by a turn.

    Separate from `load_encoder` so a setup step can fail loudly about the network while a turn
    stays offline — `_load` hits the cache and only reaches for the network if something is missing,
    which is a download inside a measured stage.
    """
    from src.config import resolve
    return _load(arm or resolve("embed"))[1]


def _pool(arm, out, mask):
    """-> one vector per input, pooled the way this arm was trained. See the module docstring."""
    how = arm.extra.get("pooling", "mean")
    hidden = out.last_hidden_state
    if how == "cls":
        return hidden[:, 0]
    if how != "mean":
        raise RuntimeError(
            f"{arm.id} asks for {how!r} pooling, which this backend does not implement. "
            f"Known: cls, mean."
        )
    # Mean over real tokens only. Including the padding would make a vector depend on the length of
    # the *longest other text in the batch*, so the same sentence would embed differently depending
    # on what it was batched with — and at index time that is a silent, batch-order-dependent index.
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)


def encode_transformers(arm, texts, rec):
    """The `transformers-embed` backend. -> (n, dim) float32, L2-normalised.

    `texts` is a list of strings, or `{"texts": [...], "is_query": bool}` when the caller needs the
    query side of an asymmetric encoder.
    """
    is_query = False
    if isinstance(texts, dict):
        is_query, texts = bool(texts.get("is_query")), texts["texts"]
    if isinstance(texts, str):
        texts = [texts]

    prefix = arm.extra.get("query_prefix", "") if is_query else ""
    prepared = [prefix + (t or "") for t in texts]

    tok, model = _load(arm)
    out = []
    with torch.no_grad():
        for i in range(0, len(prepared), EMBED_BATCH):
            batch = tok(prepared[i:i + EMBED_BATCH], padding=True, truncation=True,
                        return_tensors="pt")
            pooled = _pool(arm, model(**batch), batch["attention_mask"])
            out.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
    vectors = torch.cat(out).to(torch.float32).numpy() if out else np.zeros((0, dim(arm)), "float32")

    rec["texts"] = len(prepared)
    rec["chars"] = sum(len(t) for t in prepared)
    rec["dim"] = int(vectors.shape[1]) if len(vectors) else dim(arm)
    rec["is_query"] = is_query
    return vectors


def dim(arm):
    """-> the arm's vector width, as declared in the registry."""
    return int(arm.extra.get("dim", 384))


BACKENDS = {"transformers-embed": encode_transformers}
LOADERS = {"transformers-embed": load_encoder}
