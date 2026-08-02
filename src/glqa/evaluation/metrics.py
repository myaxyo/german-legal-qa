"""Evaluation metrics for German Legal QA.

Metrics:
  - Exact Match: does the answer contain the correct statute reference?
  - Token F1: overlap between predicted and reference answer tokens
  - ROUGE-L: longest common subsequence based scoring
  - BERTScore: semantic similarity using multilingual BERT
  - Citation Accuracy: does the model cite the correct §-reference?
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Text-level Metrics
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize German legal text for comparison."""
    text = text.lower()
    # Normalize paragraph references
    text = re.sub(r"§\s*(\d+)", r"§\1", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    """Check if normalized prediction matches reference exactly."""
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference.

    Tokenizes on whitespace and computes precision/recall/F1.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L (longest common subsequence) F1 score."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    lcs_length = lcs_table[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Citation Accuracy
# ---------------------------------------------------------------------------

# Regex to extract German legal references (§123 BGB, Art. 5 GG, etc.)
CITATION_PATTERN = re.compile(
    r"(?:§+\s*\d+[a-z]?(?:\s*(?:Abs\.?\s*\d+|S\.?\s*\d+|Nr\.?\s*\d+))*\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöü]*)"
    r"|(?:Art\.?\s*\d+(?:\s*(?:Abs\.?\s*\d+))*\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöü]*)"
)


def extract_citations(text: str) -> set[str]:
    """Extract legal citations (§ references) from text."""
    matches = CITATION_PATTERN.findall(text)
    # Normalize: remove extra spaces
    return {re.sub(r"\s+", " ", m).strip() for m in matches}


def citation_accuracy(prediction: str, reference: str) -> float:
    """Measure how well the predicted citations match the reference.

    Returns F1 over extracted statute citations.
    """
    pred_cites = extract_citations(prediction)
    ref_cites = extract_citations(reference)

    if not ref_cites:
        # No citations expected; give full score if none predicted
        return 1.0 if not pred_cites else 0.5

    if not pred_cites:
        return 0.0

    # Compute F1 over citation sets
    common = pred_cites & ref_cites
    precision = len(common) / len(pred_cites) if pred_cites else 0
    recall = len(common) / len(ref_cites) if ref_cites else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BERTScore (wraps the bert-score library)
# ---------------------------------------------------------------------------


def compute_bert_scores(
    predictions: list[str],
    references: list[str],
    model_type: str = "bert-base-multilingual-cased",
) -> dict[str, float]:
    """Compute BERTScore for a batch of predictions vs references.

    Returns dict with 'precision', 'recall', 'f1' (averaged).
    """
    from bert_score import score

    P, R, F1 = score(
        predictions,
        references,
        model_type=model_type,
        lang="de",
        verbose=False,
    )

    return {
        "bert_score_precision": P.mean().item(),
        "bert_score_recall": R.mean().item(),
        "bert_score_f1": F1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    compute_bertscore: bool = True,
) -> dict[str, float]:
    """Compute all metrics over a list of prediction/reference pairs.

    Returns a dict mapping metric_name → average score.
    """
    n = len(predictions)
    assert n == len(references), "Predictions and references must have same length"

    scores = {
        "exact_match": [],
        "token_f1": [],
        "rouge_l": [],
        "citation_accuracy": [],
    }

    for pred, ref in zip(predictions, references):
        scores["exact_match"].append(exact_match(pred, ref))
        scores["token_f1"].append(token_f1(pred, ref))
        scores["rouge_l"].append(rouge_l(pred, ref))
        scores["citation_accuracy"].append(citation_accuracy(pred, ref))

    results = {k: float(np.mean(v)) for k, v in scores.items()}
    results["num_samples"] = n

    # BERTScore (batched, slower)
    if compute_bertscore and n > 0:
        bert_results = compute_bert_scores(predictions, references)
        results.update(bert_results)

    return results
