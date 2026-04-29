"""Evaluation metrics shared by the notebook experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_val,
    p_val,
    y_train=None,
    p_train=None,
    target_tpr: float = 0.80,
    fixed_threshold: float | None = None,
) -> dict[str, float]:
    """Compute ranking and fixed-threshold triage metrics.

    The partial AUC follows the project convention: ranking negatives by
    ``-score`` and integrating up to ``FPR = 1 - target_tpr``.
    """
    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=float)
    out: dict[str, float] = {}

    out['ROC_AUC'] = (
        float(roc_auc_score(y_val, p_val))
        if 0 < y_val.sum() < len(y_val)
        else np.nan
    )
    out['PR_AUC'] = (
        float(average_precision_score(y_val, p_val))
        if y_val.sum() > 0
        else np.nan
    )

    max_fpr = 1.0 - target_tpr
    fpr_arr, tpr_arr, _ = roc_curve(1 - y_val, -p_val)
    stop = np.searchsorted(fpr_arr, max_fpr, side='right')
    key = f'pAUC@TPR>={target_tpr:.2f}'
    if 0 < stop < len(fpr_arr):
        tpr_cut = np.append(
            tpr_arr[:stop],
            np.interp(
                max_fpr,
                [fpr_arr[stop - 1], fpr_arr[stop]],
                [tpr_arr[stop - 1], tpr_arr[stop]],
            ),
        )
        fpr_cut = np.append(fpr_arr[:stop], max_fpr)
        out[key] = float(auc(fpr_cut, tpr_cut))
    else:
        out[key] = np.nan

    if fixed_threshold is not None:
        threshold = float(fixed_threshold)
    else:
        ref_y = np.asarray(y_train, dtype=int) if y_train is not None and p_train is not None else y_val
        ref_p = np.asarray(p_train, dtype=float) if y_train is not None and p_train is not None else p_val
        pos_ref = ref_p[ref_y == 1]
        threshold = float(np.quantile(pos_ref, 1.0 - target_tpr)) if len(pos_ref) > 0 else 1.0

    yhat = (p_val >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, yhat, labels=[0, 1]).ravel()
    out['thr_for_target_TPR'] = threshold
    out['TPR_at_thr'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    out['FPR_at_thr'] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    out['Precision_at_thr'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    if y_train is not None and p_train is not None:
        y_tr = np.asarray(y_train, dtype=int)
        p_tr = np.asarray(p_train, dtype=float)
        out['train_ROC_AUC'] = (
            float(roc_auc_score(y_tr, p_tr))
            if 0 < y_tr.sum() < len(y_tr)
            else np.nan
        )
        yhat_tr = (p_tr >= threshold).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_tr, yhat_tr, labels=[0, 1]).ravel()
        out['train_TPR_at_thr'] = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else np.nan
        out['train_FPR_at_thr'] = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else np.nan
        out['train_Precision_at_thr'] = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else np.nan

    return out


def fold_context(y, tr_idx, va_idx) -> dict[str, int]:
    """Small fold summary used in experiment result tables."""
    return {
        'n_train': len(tr_idx),
        'n_val': len(va_idx),
        'n_pos_train': int(y[tr_idx].sum()),
        'n_pos_val': int(y[va_idx].sum()),
    }
