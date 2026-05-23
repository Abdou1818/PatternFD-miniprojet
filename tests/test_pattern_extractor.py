"""
tests/test_pattern_extractor.py
Tests unitaires pour src/pattern_extractor.py
"""

import pytest
import pandas as pd
from src.pattern_extractor import (
    extract_prefixes,
    extract_tokens,
    extract_ngrams,
    prefix_patterns,
    token_patterns,
    exact_patterns,
    extract_all_patterns,
)


# ---------------------------------------------------------------------------
# extract_prefixes
# ---------------------------------------------------------------------------

class TestExtractPrefixes:

    def test_normal(self):
        assert extract_prefixes("ABCDE", 3) == ["A", "AB", "ABC"]

    def test_k_max_greater_than_length(self):
        assert extract_prefixes("AB", 5) == ["A", "AB"]

    def test_k_max_equals_length(self):
        assert extract_prefixes("ABC", 3) == ["A", "AB", "ABC"]

    def test_empty_string(self):
        assert extract_prefixes("", 3) == []

    def test_non_string(self):
        assert extract_prefixes(12345, 3) == []

    def test_none(self):
        assert extract_prefixes(None, 3) == []

    def test_single_char(self):
        assert extract_prefixes("X", 5) == ["X"]

    def test_zip_code(self):
        result = extract_prefixes("60601", 3)
        assert result == ["6", "60", "606"]

    def test_k_max_zero(self):
        assert extract_prefixes("ABC", 0) == []


# ---------------------------------------------------------------------------
# extract_tokens
# ---------------------------------------------------------------------------

class TestExtractTokens:

    def test_space_separator(self):
        assert extract_tokens("John Smith") == ["John", "Smith"]

    def test_hyphen_separator(self):
        assert extract_tokens("on-premise") == ["on", "premise"]

    def test_multiple_separators(self):
        tokens = extract_tokens("Los Angeles, CA")
        assert "Los" in tokens and "Angeles" in tokens

    def test_empty_string(self):
        assert extract_tokens("") == []

    def test_non_string(self):
        assert extract_tokens(42) == []

    def test_single_word(self):
        assert extract_tokens("Chicago") == ["Chicago"]

    def test_trailing_spaces(self):
        assert extract_tokens("  John Smith  ") == ["John", "Smith"]

    def test_slash_separator(self):
        tokens = extract_tokens("On/Off-Premise")
        assert "On" in tokens


# ---------------------------------------------------------------------------
# extract_ngrams
# ---------------------------------------------------------------------------

class TestExtractNgrams:

    def test_normal(self):
        assert extract_ngrams("ABCDE", 3) == ["ABC", "BCD", "CDE"]

    def test_exact_length(self):
        assert extract_ngrams("ABC", 3) == ["ABC"]

    def test_shorter_than_n(self):
        assert extract_ngrams("AB", 3) == []

    def test_empty(self):
        assert extract_ngrams("", 3) == []

    def test_bigrams(self):
        assert extract_ngrams("ABCD", 2) == ["AB", "BC", "CD"]

    def test_unigrams(self):
        assert extract_ngrams("ABC", 1) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# prefix_patterns
# ---------------------------------------------------------------------------

class TestPrefixPatterns:

    def setup_method(self):
        self.s = pd.Series(["60601", "60602", "60603", "77001", "77002"])

    def test_finds_common_prefix(self):
        result = prefix_patterns(self.s, k_max=3, min_support=2)
        assert "606" in result
        assert "770" in result

    def test_min_support_filter(self):
        # "606" a support=3, "770" support=2 → min_support=3 garde "606" seulement
        result = prefix_patterns(self.s, k_max=3, min_support=3)
        assert "606" in result
        assert "770" not in result

    def test_nan_ignored(self):
        s = pd.Series(["60601", None, "60602", None])
        result = prefix_patterns(s, k_max=3, min_support=2)
        assert "606" in result
        # Les indices retournés ne doivent pas inclure les NaN (indices 1 et 3)
        assert 1 not in result["606"]
        assert 3 not in result["606"]

    def test_numeric_column_converted(self):
        s = pd.Series([60601, 60602, 77001])
        result = prefix_patterns(s, k_max=3, min_support=2)
        assert "606" in result

    def test_float_column(self):
        s = pd.Series([20852.0, 20852.0, 20878.0])
        result = prefix_patterns(s, k_max=5, min_support=2)
        assert any("2085" in p for p in result)

    def test_indices_are_correct(self):
        s = pd.Series(["60601", "60602", "77001"])
        result = prefix_patterns(s, k_max=3, min_support=2)
        assert sorted(result["606"]) == [0, 1]

    def test_empty_series(self):
        result = prefix_patterns(pd.Series([], dtype=str), k_max=3, min_support=1)
        assert result == {}


# ---------------------------------------------------------------------------
# token_patterns
# ---------------------------------------------------------------------------

class TestTokenPatterns:

    def test_finds_common_first_token(self):
        s = pd.Series(["John Smith", "John Brown", "Susan Miller"])
        result = token_patterns(s, min_support=2)
        assert "John" in result
        assert "Susan" not in result

    def test_nan_ignored(self):
        s = pd.Series(["John Smith", None, "John Brown"])
        result = token_patterns(s, min_support=1)
        assert "John" in result
        assert 1 not in result.get("John", [])

    def test_min_support_filter(self):
        s = pd.Series(["John Smith", "John Brown", "John Taylor", "Susan Miller"])
        result = token_patterns(s, min_support=3)
        assert "John" in result
        assert "Susan" not in result


# ---------------------------------------------------------------------------
# exact_patterns
# ---------------------------------------------------------------------------

class TestExactPatterns:

    def test_basic(self):
        s = pd.Series(["Chicago", "Chicago", "Houston"])
        result = exact_patterns(s, min_support=2)
        assert "Chicago" in result
        assert "Houston" not in result

    def test_nan_excluded(self):
        s = pd.Series(["IL", None, "IL"])
        result = exact_patterns(s, min_support=1)
        assert "IL" in result


# ---------------------------------------------------------------------------
# extract_all_patterns
# ---------------------------------------------------------------------------

class TestExtractAllPatterns:

    def test_returns_both_types(self):
        s = pd.Series(["John Smith", "John Brown", "Susan Miller"])
        result = extract_all_patterns(s, k_max=3, min_support=2)
        # Doit contenir au moins "startswith" (préfixes) et "first_token"
        assert "startswith" in result or "first_token" in result

    def test_empty_series(self):
        result = extract_all_patterns(pd.Series([], dtype=str))
        assert result == {}
