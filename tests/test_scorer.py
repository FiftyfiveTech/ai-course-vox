"""Unit tests for the entity-capture scorer (VOX-015).

All expected values are hand-computed against the fixture data below.
No model, no network, no files — pure arithmetic on dicts.
"""
import pytest

from src.scorer import normalise, normalise_value, score_utterance, score_dataset


# ---------------------------------------------------------------------------
# normalise()
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercase(self):
        assert normalise("Priya") == "priya"

    def test_strips_whitespace(self):
        assert normalise("  priya  ") == "priya"

    def test_collapses_internal_whitespace(self):
        assert normalise("design  team") == "design team"

    def test_strips_trailing_period(self):
        assert normalise("priya.") == "priya"

    def test_strips_trailing_comma(self):
        assert normalise("rahul,") == "rahul"

    def test_none_returns_none(self):
        assert normalise(None) is None

    def test_canonical_duration_unchanged(self):
        assert normalise("1h") == "1h"

    def test_canonical_time_unchanged(self):
        assert normalise("15:00") == "15:00"

    def test_canonical_date_unchanged(self):
        assert normalise("2026-08-18") == "2026-08-18"


class TestNormaliseValue:
    def test_list_is_sorted(self):
        assert normalise_value(["sneha", "rahul"]) == ["rahul", "sneha"]

    def test_list_is_lowercased(self):
        assert normalise_value(["Priya"]) == ["priya"]

    def test_scalar_passes_through(self):
        assert normalise_value("1h") == "1h"


# ---------------------------------------------------------------------------
# score_utterance()
# ---------------------------------------------------------------------------

class TestScoreUtterance:
    def test_perfect_match(self):
        gold = {"intent": "book_meeting", "person": ["priya"], "duration": "1h",
                "date": "2026-08-18", "time": "15:00"}
        extracted = {"intent": "book_meeting", "person": ["priya"], "duration": "1h",
                     "date": "2026-08-18", "time": "15:00"}
        result = score_utterance(gold, extracted)
        assert result["correct"] == 5
        assert result["total_gold"] == 5
        assert result["false_positives"] == []
        assert all(v == "correct" for v in result["slot_results"].values())

    def test_miss_when_field_absent(self):
        gold = {"intent": "book_meeting", "duration": "1h"}
        extracted = {"intent": "book_meeting"}   # duration missing
        result = score_utterance(gold, extracted)
        assert result["correct"] == 1
        assert result["total_gold"] == 2
        assert result["slot_results"]["duration"] == "miss"

    def test_wrong_when_value_differs(self):
        gold = {"intent": "book_meeting", "time": "15:00"}
        extracted = {"intent": "book_meeting", "time": "14:00"}
        result = score_utterance(gold, extracted)
        assert result["correct"] == 1
        assert result["slot_results"]["time"] == "wrong"

    def test_false_positive_tracked(self):
        gold = {"intent": "greet"}
        extracted = {"intent": "greet", "person": ["priya"]}   # person not in gold
        result = score_utterance(gold, extracted)
        assert result["correct"] == 1
        assert result["false_positives"] == ["person"]

    def test_normalisation_applied_during_comparison(self):
        # Extracted has different case/whitespace — should still match after normalise()
        gold = {"team": "design team"}
        extracted = {"team": "Design  Team"}
        result = score_utterance(gold, extracted)
        assert result["slot_results"]["team"] == "correct"

    def test_list_order_irrelevant(self):
        # Gold has ["priya", "rahul"], extracted has ["Rahul", "Priya"]
        gold = {"person": ["priya", "rahul"]}
        extracted = {"person": ["Rahul", "Priya"]}
        result = score_utterance(gold, extracted)
        assert result["slot_results"]["person"] == "correct"

    def test_empty_extraction(self):
        gold = {"intent": "log_hours", "duration": "4h", "project": "vox project"}
        result = score_utterance(gold, {})
        assert result["correct"] == 0
        assert result["total_gold"] == 3
        assert all(v == "miss" for v in result["slot_results"].values())

    def test_ambiguous_intent_only(self):
        gold = {"intent": "ambiguous"}
        extracted = {"intent": "ambiguous"}
        result = score_utterance(gold, extracted)
        assert result["correct"] == 1
        assert result["total_gold"] == 1


# ---------------------------------------------------------------------------
# score_dataset() — hand-computed fixture
#
# Fixture: 3 utterances, 2 categories
#
# utt_A (entity): gold has 3 slots; extracted gets 2 right, 1 wrong
#   correct=2, total_gold=3
#
# utt_B (entity): gold has 2 slots; extracted gets all right
#   correct=2, total_gold=2
#
# utt_C (greet):  gold has 1 slot; extracted misses it
#   correct=0, total_gold=1
#
# Overall: correct=4, total_gold=6  => capture_rate = 4/6 = 0.6667
# entity:  correct=4, total_gold=5  => capture_rate = 4/5 = 0.8
# greet:   correct=0, total_gold=1  => capture_rate = 0/1 = 0.0
# ---------------------------------------------------------------------------

GOLD_RECORDS = [
    {"id": "utt_A", "category": "entity",
     "gold": {"intent": "log_hours", "duration": "4h", "project": "vox project"}},
    {"id": "utt_B", "category": "entity",
     "gold": {"intent": "book_meeting", "time": "15:00"}},
    {"id": "utt_C", "category": "greet",
     "gold": {"intent": "greet"}},
]

EXTRACTED_RECORDS = [
    {"id": "utt_A", "extracted": {"intent": "log_hours", "duration": "4h",
                                   "project": "wrong project"}},  # project wrong
    {"id": "utt_B", "extracted": {"intent": "book_meeting", "time": "15:00"}},  # perfect
    {"id": "utt_C", "extracted": {}},                                            # total miss
]


class TestScoreDataset:
    def setup_method(self):
        self.result = score_dataset(GOLD_RECORDS, EXTRACTED_RECORDS)

    def test_overall_capture_rate(self):
        # 4 correct out of 6 gold slots
        assert self.result["total_correct"] == 4
        assert self.result["total_gold"] == 6
        assert self.result["overall_capture_rate"] == round(4 / 6, 4)

    def test_per_category_entity(self):
        entity = self.result["per_category"]["entity"]
        assert entity["correct"] == 4
        assert entity["total_gold"] == 5
        assert entity["capture_rate"] == round(4 / 5, 4)

    def test_per_category_greet(self):
        greet = self.result["per_category"]["greet"]
        assert greet["correct"] == 0
        assert greet["total_gold"] == 1
        assert greet["capture_rate"] == 0.0

    def test_utterance_results_count(self):
        assert len(self.result["utterance_results"]) == 3

    def test_missing_extraction_handled(self):
        # utt_C had no extracted record — scorer should handle gracefully
        utt_c = next(r for r in self.result["utterance_results"] if r["id"] == "utt_C")
        assert utt_c["correct"] == 0
        assert utt_c["slot_results"]["intent"] == "miss"

    def test_false_positives_not_counted_as_correct(self):
        assert self.result["total_false_positives"] == 0
