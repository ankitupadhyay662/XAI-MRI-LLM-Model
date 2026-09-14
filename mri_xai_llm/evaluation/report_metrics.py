"""Text-similarity metrics (BLEU / ROUGE / BERTScore) for generated reports.

TEXT_METRIC_CAVEAT (READ THIS BEFORE USING THESE METRICS):
BLEU/ROUGE/BERTScore measure textual similarity only and do NOT establish
clinical correctness of a report; a report can score well on text metrics
while containing clinically significant errors, and vice versa. Use
finding-level sensitivity/specificity and hallucination-rate metrics (see
evaluation.classification) alongside these for meaningful evaluation.
"""
from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

TEXT_METRIC_CAVEAT = (
    "BLEU/ROUGE/BERTScore measure textual similarity only and do NOT establish "
    "clinical correctness of a report; a report can score well on text metrics "
    "while containing clinically significant errors, and vice versa. Use "
    "finding-level sensitivity/specificity and hallucination-rate metrics (see "
    "evaluation.classification) alongside these for meaningful evaluation."
)


def _compute_bleu(reference: str, hypothesis: str) -> float | None:
    try:
        import sacrebleu

        return float(sacrebleu.sentence_bleu(hypothesis, [reference]).score)
    except ImportError:
        pass
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        return float(
            sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)
        )
    except ImportError:
        logger.warning("Neither sacrebleu nor nltk is installed; BLEU score unavailable.")
        return None


def _compute_rouge(reference: str, hypothesis: str) -> dict | None:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            key: {"precision": s.precision, "recall": s.recall, "fmeasure": s.fmeasure}
            for key, s in scores.items()
        }
    except ImportError:
        logger.warning("rouge_score is not installed; ROUGE scores unavailable.")
        return None


def _compute_bertscore(reference: str, hypothesis: str) -> dict | None:
    try:
        import bert_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, f1 = bert_score.score(
                [hypothesis], [reference], lang="en", verbose=False
            )
        return {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
        }
    except ImportError:
        logger.warning("bert_score is not installed; BERTScore unavailable.")
        return None
    except Exception as exc:  # noqa: BLE001 - bert_score can fail at runtime (model download, etc.)
        logger.warning("BERTScore computation failed (%s); returning None.", exc)
        return None


def compute_text_metrics(reference: str, hypothesis: str) -> dict:
    """Compute BLEU, ROUGE, and BERTScore between a reference and hypothesis report.

    See TEXT_METRIC_CAVEAT: these are text-similarity metrics only and do not
    establish clinical correctness. Optional heavy dependencies (sacrebleu/nltk,
    rouge_score, bert_score) are imported lazily; any missing dependency yields
    None for that metric rather than raising.
    """
    return {
        "bleu": _compute_bleu(reference, hypothesis),
        "rouge": _compute_rouge(reference, hypothesis),
        "bertscore": _compute_bertscore(reference, hypothesis),
        "caveat": TEXT_METRIC_CAVEAT,
    }
