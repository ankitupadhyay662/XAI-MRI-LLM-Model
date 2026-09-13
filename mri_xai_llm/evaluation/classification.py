"""Classification and finding-level evaluation metrics.

Covers both standard classification metrics (via sklearn) and finding-level
metrics that quantify hallucination behavior (unsupported/invented findings),
which is central to evaluating a hallucination-controlled reporting pipeline.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    multilabel: bool = False,
) -> dict:
    """Compute accuracy/precision/recall/f1 (+ optional AUROC/AUPRC and
    per-class sensitivity/specificity) for single-label or multi-label
    (finding taxonomy) classification.

    y_true, y_pred: for single-label, 1D arrays of class indices/labels.
                    for multilabel, 2D binary indicator arrays [n_samples, n_classes].
    y_score: predicted probabilities/scores, same shape as y_pred for multilabel,
             or [n_samples, n_classes] (or 1D for binary) for single-label.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict = {}

    if multilabel:
        metrics["accuracy"] = float((y_true == y_pred).mean())
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        n_classes = y_true.shape[1]
        sensitivity, specificity = [], []
        for c in range(n_classes):
            tn, fp, fn, tp = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1]).ravel()
            sensitivity.append(float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"))
            specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"))
        metrics["sensitivity_per_class"] = sensitivity
        metrics["specificity_per_class"] = specificity

        if y_score is not None:
            y_score = np.asarray(y_score)
            try:
                metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_score, average="macro"))
            except ValueError:
                metrics["roc_auc_macro"] = None
            try:
                metrics["average_precision_macro"] = float(
                    average_precision_score(y_true, y_score, average="macro")
                )
            except ValueError:
                metrics["average_precision_macro"] = None
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        metrics["precision_per_class"] = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
        metrics["recall_per_class"] = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
        metrics["f1_per_class"] = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
        metrics["labels"] = labels

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sensitivity, specificity = [], []
        total = cm.sum()
        for i, _label in enumerate(labels):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            sensitivity.append(float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"))
            specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"))
        metrics["sensitivity_per_class"] = sensitivity
        metrics["specificity_per_class"] = specificity

        if y_score is not None:
            y_score = np.asarray(y_score)
            try:
                if y_score.ndim == 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
                    metrics["average_precision"] = float(average_precision_score(y_true, y_score))
                else:
                    metrics["roc_auc_macro"] = float(
                        roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                    )
                    metrics["average_precision_macro"] = None
            except ValueError:
                metrics["roc_auc"] = None

    return metrics


def finding_level_metrics(
    gt_findings: list[set[str]],
    pred_findings: list[set[str]],
) -> dict:
    """Per-sample finding-level sensitivity/specificity/FNR and hallucination
    (unsupported-finding) rate, aggregated across the taxonomy.

    gt_findings / pred_findings: parallel lists, one set of finding labels per
    sample (e.g. {"Mass/Lesion", "Edema"}).
    """
    if len(gt_findings) != len(pred_findings):
        raise ValueError("gt_findings and pred_findings must be the same length")

    taxonomy = sorted(set().union(*gt_findings, *pred_findings)) if gt_findings else []

    tp = fp = fn = tn = 0
    total_pred = 0
    total_unsupported_pred = 0

    for gt, pred in zip(gt_findings, pred_findings):
        for label in taxonomy:
            in_gt = label in gt
            in_pred = label in pred
            if in_gt and in_pred:
                tp += 1
            elif in_gt and not in_pred:
                fn += 1
            elif not in_gt and in_pred:
                fp += 1
            else:
                tn += 1
        total_pred += len(pred)
        total_unsupported_pred += len(pred - gt)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else float("nan")
    unsupported_finding_rate = total_unsupported_pred / total_pred if total_pred > 0 else 0.0
    hallucination_rate = unsupported_finding_rate  # explicit alias, same concept

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "false_negative_rate": float(false_negative_rate),
        "unsupported_finding_rate": float(unsupported_finding_rate),
        "hallucination_rate": float(hallucination_rate),
        "n_samples": len(gt_findings),
        "taxonomy_size": len(taxonomy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
