"""
tests/test_candidate_generator.py
Tests unitaires pour src/candidate_generator.py
"""

import pytest
import pandas as pd
from src.candidate_generator import (
    _most_common_value,
    generate_candidates,
    generate_all_candidates,
)


# ---------------------------------------------------------------------------
# _most_common_value
# ---------------------------------------------------------------------------

class TestMostCommonValue:

    def test_basic(self):
        s = pd.Series(["Chicago", "Chicago", "Houston"])
        val, count, total = _most_common_value(s)
        assert val == "Chicago"
        assert count == 2
        assert total == 3

    def test_all_same(self):
        s = pd.Series(["IL", "IL", "IL"])
        val, count, total = _most_common_value(s)
        assert val == "IL" and count == 3 and total == 3

    def test_nan_ignored(self):
        s = pd.Series(["Chicago", None, "Chicago"])
        val, count, total = _most_common_value(s)
        assert val == "Chicago" and total == 2

    def test_empty(self):
        val, count, total = _most_common_value(pd.Series([], dtype=str))
        assert val is None and count == 0 and total == 0

    def test_all_nan(self):
        s = pd.Series([None, None])
        val, count, total = _most_common_value(s)
        assert val is None


# ---------------------------------------------------------------------------
# generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:

    def setup_method(self):
        self.df = pd.DataFrame({
            "ZIP":  ["60601", "60602", "60603", "77001", "77002"],
            "CITY": ["Chicago", "Chicago", "Chicago", "Houston", "Houston"],
            "STATE": ["IL", "IL", "IL", "TX", "TX"],
        })
        # Pattern "606" → lignes 0,1,2
        self.pmap = {"606": [0, 1, 2], "770": [3, 4]}

    def test_finds_zip_chicago(self):
        cands = generate_candidates(
            self.df, "ZIP", self.pmap, "startswith", ["CITY"], min_support=2
        )
        assert any(
            c["pattern_X"] == "606" and c["pattern_Y"] == "Chicago"
            for c in cands
        )

    def test_no_self_dependency(self):
        cands = generate_candidates(
            self.df, "ZIP", self.pmap, "startswith", ["ZIP"], min_support=2
        )
        assert all(c["col_Y"] != "ZIP" for c in cands)

    def test_match_type_propagated(self):
        cands = generate_candidates(
            self.df, "ZIP", {"606": [0, 1, 2]}, "startswith", ["CITY"], min_support=2
        )
        assert cands[0]["match_type_X"] == "startswith"
        assert cands[0]["match_type_Y"] == "exact"

    def test_first_token_becomes_startswith(self):
        cands = generate_candidates(
            self.df, "ZIP", {"606": [0, 1, 2]}, "first_token", ["CITY"], min_support=2
        )
        assert cands[0]["match_type_X"] == "startswith"

    def test_min_support_filter(self):
        pmap = {"606": [0, 1, 2], "X": [0]}   # "X" → support 1
        cands = generate_candidates(
            self.df, "ZIP", pmap, "startswith", ["CITY"], min_support=2
        )
        assert all(c["pattern_X"] != "X" for c in cands)

    def test_raw_confidence_value(self):
        cands = generate_candidates(
            self.df, "ZIP", {"606": [0, 1, 2]}, "startswith", ["CITY"], min_support=2
        )
        c = next(c for c in cands if c["pattern_X"] == "606")
        assert c["raw_confidence"] == pytest.approx(1.0)

    def test_multiple_targets(self):
        cands = generate_candidates(
            self.df, "ZIP", self.pmap, "startswith", ["CITY", "STATE"], min_support=2
        )
        targets = {c["col_Y"] for c in cands}
        assert "CITY" in targets and "STATE" in targets


# ---------------------------------------------------------------------------
# generate_all_candidates
# ---------------------------------------------------------------------------

class TestGenerateAllCandidates:

    def setup_method(self):
        self.df = pd.DataFrame({
            "ZIP":  ["60601", "60602", "60603", "77001"],
            "CITY": ["Chicago", "Chicago", "Chicago", "Houston"],
        })
        self.pattern_maps = {
            "ZIP": {
                "startswith": {"606": [0, 1, 2], "770": [3]},
            }
        }

    def test_generates_candidates(self):
        cands = generate_all_candidates(self.df, self.pattern_maps, min_support=2)
        assert len(cands) > 0

    def test_no_duplicates(self):
        cands = generate_all_candidates(self.df, self.pattern_maps, min_support=2)
        keys = [(c["col_X"], str(c["pattern_X"]), c["match_type_X"],
                 c["col_Y"], str(c["pattern_Y"])) for c in cands]
        assert len(keys) == len(set(keys))

    def test_empty_pattern_maps(self):
        cands = generate_all_candidates(self.df, {}, min_support=2)
        assert cands == []
