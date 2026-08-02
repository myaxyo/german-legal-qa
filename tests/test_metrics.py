"""Unit tests for evaluation metrics."""

import pytest

from glqa.evaluation.metrics import (
    citation_accuracy,
    exact_match,
    extract_citations,
    normalize_text,
    rouge_l,
    token_f1,
)


class TestNormalize:
    def test_lowercase(self):
        assert normalize_text("BGB") == "bgb"

    def test_paragraph_normalization(self):
        assert "§823" in normalize_text("§ 823")
        assert "§123" in normalize_text("§  123")

    def test_whitespace(self):
        assert normalize_text("  foo   bar  ") == "foo bar"


class TestExactMatch:
    def test_identical(self):
        assert exact_match("§823 BGB", "§823 BGB") == 1.0

    def test_normalized_match(self):
        assert exact_match("§ 823 BGB", "§823 bgb") == 1.0

    def test_no_match(self):
        assert exact_match("§823 BGB", "§824 BGB") == 0.0


class TestTokenF1:
    def test_perfect_match(self):
        text = "Dies ist ein Test"
        assert token_f1(text, text) == 1.0

    def test_partial_overlap(self):
        pred = "Dies ist ein guter Test"
        ref = "Dies ist ein schlechter Test"
        score = token_f1(pred, ref)
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert token_f1("Hund Katze", "Apfel Birne") == 0.0

    def test_empty(self):
        assert token_f1("", "") == 1.0


class TestRougeL:
    def test_identical(self):
        text = "Der Schuldner hat den Schaden zu ersetzen"
        assert rouge_l(text, text) == 1.0

    def test_subsequence(self):
        pred = "Der Schuldner hat den entstandenen Schaden vollständig zu ersetzen"
        ref = "Der Schuldner hat den Schaden zu ersetzen"
        score = rouge_l(pred, ref)
        assert score > 0.7

    def test_no_match(self):
        assert rouge_l("abc def", "xyz uvw") == 0.0


class TestCitationExtraction:
    def test_simple_paragraph(self):
        cites = extract_citations("Gemäß §823 BGB besteht ein Anspruch")
        assert any("823" in c and "BGB" in c for c in cites)

    def test_article(self):
        cites = extract_citations("Art. 5 GG schützt die Meinungsfreiheit")
        assert any("5" in c and "GG" in c for c in cites)

    def test_paragraph_with_absatz(self):
        cites = extract_citations("§123 Abs. 1 BGB")
        assert len(cites) >= 1

    def test_no_citations(self):
        cites = extract_citations("Es gibt keine Paragraphen hier")
        assert len(cites) == 0


class TestCitationAccuracy:
    def test_perfect(self):
        pred = "Gemäß §823 BGB..."
        ref = "Nach §823 BGB..."
        assert citation_accuracy(pred, ref) == 1.0

    def test_missing_citation(self):
        pred = "Der Anspruch besteht."
        ref = "Nach §823 BGB besteht der Anspruch."
        assert citation_accuracy(pred, ref) == 0.0

    def test_no_expected_citations(self):
        pred = "Keine Paragraphen."
        ref = "Auch keine Paragraphen."
        assert citation_accuracy(pred, ref) == 1.0
