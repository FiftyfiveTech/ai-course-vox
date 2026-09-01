"""Entity-capture scorer (VOX-015).

Exact-match per entity against gold labels, per-category precision/recall,
and an overall capture rate. Normalisation follows ENTITY_SPEC.md §2.

Usage:
    from src.scorer import score_utterance, score_dataset
"""
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Normalisation (ENTITY_SPEC.md §2.1)
# ---------------------------------------------------------------------------

def normalise(value):
    """Normalise a scalar string value for exact-match comparison.

    Steps from ENTITY_SPEC.md §2.1:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Collapse internal whitespace runs to a single space
      4. Remove trailing punctuation (full stops, commas)
    Duration and date values are already in canonical form in the gold labels
    (e.g. "1h", "2026-08-18") so no further expansion is needed here — the
    NLU output must produce the same canonical form.
    """
    if value is None:
        return None
    s = str(value).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,")
    return s


def normalise_value(value):
    """Normalise a field value, handling both scalars and lists."""
    if isinstance(value, list):
        return sorted(normalise(v) for v in value)
    return normalise(value)


# ---------------------------------------------------------------------------
# Per-utterance scoring
# ---------------------------------------------------------------------------

def score_utterance(gold: dict, extracted: dict) -> dict:
    """Compare extracted entities against gold for one utterance.

    Args:
        gold: gold label dict (from labels.json), e.g.
              {"intent": "book_meeting", "person": ["priya"], "duration": "1h"}
        extracted: NLU output dict in the same schema

    Returns a dict with:
        correct        number of gold slots matched exactly
        total_gold     number of gold slots
        false_positives  fields in extracted not in gold
        slot_results   per-field breakdown {"field": "correct"|"miss"|"wrong"}
    """
    slot_results = {}
    correct = 0
    false_positives = []

    for field, gold_val in gold.items():
        norm_gold = normalise_value(gold_val)
        if field not in extracted:
            slot_results[field] = "miss"
        else:
            ext_val = extracted[field]
            # Gold may be a single-element list (e.g. person: ["kiran"]) while the model
            # returns a scalar. Coerce scalar → list so "kiran" matches ["kiran"].
            if isinstance(gold_val, list) and not isinstance(ext_val, list):
                ext_val = [ext_val]
            norm_ext = normalise_value(ext_val)
            if norm_ext == norm_gold:
                slot_results[field] = "correct"
                correct += 1
            else:
                slot_results[field] = "wrong"

    for field in extracted:
        if field not in gold:
            false_positives.append(field)

    return {
        "correct": correct,
        "total_gold": len(gold),
        "false_positives": false_positives,
        "slot_results": slot_results,
    }


# ---------------------------------------------------------------------------
# Dataset-level scoring
# ---------------------------------------------------------------------------

def score_dataset(gold_records: list, extracted_records: list) -> dict:
    """Score a full dataset.

    Args:
        gold_records: list of {"id": ..., "gold": {...}} dicts
        extracted_records: list of {"id": ..., "extracted": {...}} dicts
                           (extracted keyed by utterance id)

    Returns:
        overall_capture_rate   correct_slots / total_gold_slots
        per_category           {category: {precision, recall, correct, total_gold}}
        utterance_results      per-utterance breakdown
        total_correct          int
        total_gold             int
        total_false_positives  int
    """
    extracted_by_id = {r["id"]: r.get("extracted", {}) for r in extracted_records}

    total_correct = 0
    total_gold = 0
    total_fp = 0
    utterance_results = []

    # category is stored at the top level of gold records alongside "gold"
    category_stats = defaultdict(lambda: {"correct": 0, "total_gold": 0, "fp": 0})

    for gold_rec in gold_records:
        uid = gold_rec["id"]
        category = gold_rec.get("category", "unknown")
        gold = gold_rec.get("gold", {})
        extracted = extracted_by_id.get(uid, {})

        result = score_utterance(gold, extracted)
        result["id"] = uid
        result["category"] = category
        utterance_results.append(result)

        total_correct += result["correct"]
        total_gold += result["total_gold"]
        total_fp += len(result["false_positives"])

        category_stats[category]["correct"] += result["correct"]
        category_stats[category]["total_gold"] += result["total_gold"]
        category_stats[category]["fp"] += len(result["false_positives"])

    overall_capture_rate = total_correct / total_gold if total_gold else 0.0

    per_category = {}
    for cat, stats in category_stats.items():
        cg = stats["total_gold"]
        cc = stats["correct"]
        per_category[cat] = {
            "correct": cc,
            "total_gold": cg,
            "capture_rate": round(cc / cg, 4) if cg else 0.0,
            "false_positives": stats["fp"],
        }

    return {
        "overall_capture_rate": round(overall_capture_rate, 4),
        "total_correct": total_correct,
        "total_gold": total_gold,
        "total_false_positives": total_fp,
        "per_category": per_category,
        "utterance_results": utterance_results,
    }
