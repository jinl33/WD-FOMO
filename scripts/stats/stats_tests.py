from __future__ import annotations

import math

import numpy as np
from scipy import stats

def paired_t_test(a, b):
    a, b = _paired_arrays(a, b)
    res = stats.ttest_rel(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)

def wilcoxon_signed_rank(a, b):
    a, b = _paired_arrays(a, b)
    if np.allclose(a, b):
        return 0.0, 1.0
    res = stats.wilcoxon(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()

def bonferroni_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    return np.minimum(1.0, p * len(p)).tolist()

def _auc_components(scores_pos, scores_neg):
    diff = scores_pos[:, None] - scores_neg[None, :]
    psi = (diff > 0).astype(float) + 0.5 * (diff == 0)
    return psi.mean(axis=1), psi.mean(axis=0)

def delong_auc_test(y_true, score_a, score_b):
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if not (np.isin(y, [0, 1]).all() and 0 < y.sum() < len(y)):
        raise ValueError("y_true must be binary with both classes present")

    pos = y == 1
    neg = ~pos
    v10_a, v01_a = _auc_components(a[pos], a[neg])
    v10_b, v01_b = _auc_components(b[pos], b[neg])

    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    v10 = np.vstack([v10_a, v10_b])
    v01 = np.vstack([v01_a, v01_b])
    s10 = np.cov(v10, ddof=1)
    s01 = np.cov(v01, ddof=1)
    s = s10 / n_pos + s01 / n_neg

    auc_a, auc_b = float(v10_a.mean()), float(v10_b.mean())
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ s @ contrast)
    diff = auc_a - auc_b
    if var <= 0:
        z = 0.0 if abs(diff) < 1e-12 else math.copysign(math.inf, diff)
        p = 1.0 if z == 0.0 else 0.0
    else:
        z = diff / math.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))
    return {"auc_a": auc_a, "auc_b": auc_b, "auc_diff": diff, "z": z, "p_value": float(p)}

def auroc(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("both classes must be present")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))

def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

def _paired_arrays(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must have equal shape, got {a.shape} and {b.shape}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN in paired inputs; resolve before testing")
    return a, b
