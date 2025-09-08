#!/usr/bin/env python3
"""
afm_sampling_tools.py
─────────────────────
BornAgain-independent utilities for AFM particle sampling:

I/O
  • load_lineprofiles(txt)  -> x_cols, y_cols (nm)
  • extract_hsub_and_dmin() -> hsub_nm, dmin_nm  (raw paired arrays)

2D Quantile Grid (correlation-preserving)
  • quantile_grid_edges(dmin, hsub, Qx, Qy) -> x_edges, y_edges
  • summarize_pairs_quantile_grid_with_edges(dmin, hsub, x_edges, y_edges, reducer)
       -> diam_rep, height_rep, weight_rep, labels
  • summarize_pairs_quantile_grid(dmin, hsub, Qx, Qy, reducer)
       -> diam_rep, height_rep, weight_rep
  • quantile_cell_counts(dmin, hsub, x_edges, y_edges) -> counts, weights_matrix

K-Medoids (PAM; deterministic)
  • kmedoids_pam(X, k, scale) -> medoid_indices, labels
  • summarize_pairs_kmedoids(dmin, hsub, K, scale)
       -> diam_rep, height_rep, weight_rep, labels

Visualization (matplotlib)
  • visualize_quantile_grid(dmin, hsub, x_edges?, y_edges?, Qx, Qy, reducer, ...)
  • visualize_kmedoids(dmin, hsub, diam_rep, height_rep, labels, weight_rep?, ...)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict, List
import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

# ============================ I/O ============================

def load_lineprofiles(txt_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read whitespace-delimited file with interleaved x,y columns.
    Assumptions:
      - 2 header rows to skip
      - '-' as NA
      - x,y in meters; converts to nm
    Returns:
      x_cols, y_cols: arrays (N, K) with columns as profiles (nm)
    """
    df = (
        pd.read_csv(
            txt_path, delim_whitespace=True, skiprows=2, na_values="-", header=None
        ).dropna(axis=1, how="all")
    )
    if df.shape[1] % 2 != 0:
        raise ValueError(f"{txt_path}: uneven number of columns (expect x,y pairs).")
    x_cols = df.iloc[:, ::2].to_numpy(dtype=float) * 1e9  # m -> nm
    y_cols = df.iloc[:, 1::2].to_numpy(dtype=float) * 1e9
    return x_cols, y_cols

def _select_indices(n_avail: int, total_plots: int | None, select: str = "first", seed: int | None = None):
    """
    Choose which profile indices to plot.

    select ∈ {"first", "even", "random"}
      - "first": first total_plots profiles
      - "even" : evenly spaced across the full set
      - "random": random unique profiles (set seed for reproducibility)
    """
    if total_plots is None or total_plots >= n_avail:
        return list(range(n_avail))

    total = max(1, int(total_plots))
    if select == "first":
        idx = np.arange(total, dtype=int)
    elif select == "even":
        idx = np.linspace(0, n_avail - 1, num=total, dtype=int)
        idx = np.unique(idx)  # guard in very small n_avail cases
    elif select == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_avail, size=total, replace=False))
    else:
        raise ValueError(f"Unknown selection mode: {select}")
    return idx.tolist()


def plot_profiles_in_pages(
    x_cols, y_cols,
    *,
    frac: float = 0.2,
    per_fig: int = 3,
    total_plots: int | None = None,
    select: str = "first",
    seed: int | None = None,
    indices: list[int] | None = None,
    figsize=(7.5, 6.5),
    annotate_dx: bool = True,
    show_baseline: bool = False,
    sharex: bool = False,
    save_prefix: str | None = None,
    dpi: int = 300,
):
    """
    Paginate line profile plots: up to `per_fig` subplots per figure.
    Also limit the TOTAL number of plotted profiles via `total_plots`.

    If `indices` is provided, it overrides selection logic.
    Otherwise, indices are chosen by `_select_indices(...)`.
    """
    n_avail = x_cols.shape[1]

    if indices is None:
        chosen = _select_indices(n_avail, total_plots, select=select, seed=seed)
    else:
        chosen = [i for i in indices if 0 <= i < n_avail]
        if not chosen:
            raise ValueError("No valid indices provided.")

    num_pages = ceil(len(chosen) / per_fig)
    figs, pages = [], []

    for page_idx in range(num_pages):
        chunk = chosen[page_idx * per_fig : (page_idx + 1) * per_fig]

        fig, axes, res = plot_first_profiles_with_fraction_markers(
            x_cols, y_cols,
            frac=frac,
            nplot=len(chunk),
            indices=chunk,
            figsize=figsize,
            annotate_dx=annotate_dx,
            show_baseline=show_baseline,
            sharex=sharex,
            savepath=None,
            dpi=dpi,
        )

        fig.suptitle(
            f"Line profiles (page {page_idx+1}/{num_pages}) — indices {chunk}",
            y=0.995, fontsize=11
        )

        if save_prefix:
            fig.savefig(f"{save_prefix}_page{page_idx+1}.png", dpi=dpi)
            fig.savefig(f"{save_prefix}_page{page_idx+1}.pdf", dpi=dpi)

        figs.append(fig)
        pages.append((chunk, res))

    return figs, pages

def plot_first_profiles_with_fraction_markers(
    x_cols: np.ndarray,
    y_cols: np.ndarray,
    frac: float = 0.2,
    nplot: int = 5,
    indices: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (7.5, 6.5),
    annotate_dx: bool = True,
    show_baseline: bool = False,
    sharex: bool = False,
    savepath: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot the first N (or specified) line profiles and mark:
      • peak (xP, yP)
      • substrate baseline = ½(min_L + min_R)
      • target = baseline + frac * (yP - baseline)
      • left/right crossing xL/xR at the target
      • Δx = xR - xL annotation (if both crossings exist)

    Returns:
      fig, axes, results_dict
        where results_dict has keys:
          'xL', 'xR', 'xP', 'yP', 'baseline', 'hsub', 'dx_frac', 'target'
          (each is a list aligned with plotted profiles)
    """

    def _interp_cross(x, y, i, j, target):
        x1, x2 = float(x[i]), float(x[j])
        y1, y2 = float(y[i]), float(y[j])
        if y2 == y1:
            return 0.5 * (x1 + x2)
        t = (target - y1) / (y2 - y1)
        t = np.clip(t, 0.0, 1.0)
        return x1 + t * (x2 - x1)

    # choose which columns to plot
    n_avail = x_cols.shape[1]
    if indices is None:
        indices = list(range(min(nplot, n_avail)))
    else:
        indices = [i for i in indices if 0 <= i < n_avail]
        if len(indices) == 0:
            raise ValueError("No valid indices to plot.")

    rows = len(indices)
    fig, axes = plt.subplots(rows, 1, figsize=figsize, sharex=sharex)
    if rows == 1:
        axes = [axes]

    out_xL, out_xR, out_xP, out_yP = [], [], [], []
    out_base, out_hsub, out_dx, out_target = [], [], [], []

    for ax, j in zip(axes, indices):
        x = x_cols[:, j]
        y = y_cols[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

        if y.size < 3:
            ax.set_title(f"Profile {j+1} (insufficient data)")
            ax.set_ylabel("Height (nm)")
            ax.grid(alpha=0.25)
            out_xL.append(np.nan); out_xR.append(np.nan)
            out_xP.append(np.nan); out_yP.append(np.nan)
            out_base.append(np.nan); out_hsub.append(np.nan)
            out_dx.append(np.nan); out_target.append(np.nan)
            continue

        # Peak
        p = int(np.nanargmax(y))
        xP, yP = float(x[p]), float(y[p])

        # Baseline from minima around the peak
        idxL_min = int(np.nanargmin(y[:p+1]))
        idxR_min = int(p + np.nanargmin(y[p:]))
        baseline = 0.5 * (float(y[idxL_min]) + float(y[idxR_min]))
        hsub = yP - baseline
        target = baseline + frac * max(hsub, 0.0)

        # Left crossing
        xL = np.nan
        for i in range(p-1, -1, -1):
            y_i, y_ip1 = y[i], y[i+1]
            if (y_i - target) == 0:
                xL = float(x[i]); break
            if (y_i - target) * (y_ip1 - target) < 0 or (y_ip1 - target) == 0:
                xL = _interp_cross(x, y, i, i+1, target)
                break

        # Right crossing
        xR = np.nan
        for i in range(p, len(y)-1):
            y_i, y_ip1 = y[i], y[i+1]
            if (y_i - target) == 0:
                xR = float(x[i]); break
            if (y_i - target) * (y_ip1 - target) < 0 or (y_ip1 - target) == 0:
                xR = _interp_cross(x, y, i, i+1, target)
                break

        dx_val = (xR - xL) if (np.isfinite(xL) and np.isfinite(xR)) else np.nan

        # ---- plotting
        ax.plot(x, y, lw=1.4, label=f"Profile {j+1}")
        ax.axhline(target, ls="--", lw=1.0, color="tab:gray", alpha=0.8, label=f"{int(frac*100)}% of h_sub")

        # Peak marker
        ax.plot([xP], [yP], marker="o", ms=5, color="tab:red", label="peak")

        # Left/right markers
        if np.isfinite(xL):
            ax.plot([xL], [target], marker="s", ms=5, color="tab:blue", label="left cross")
        if np.isfinite(xR):
            ax.plot([xR], [target], marker="^", ms=5, color="tab:green", label="right cross")

        # Δx segment + label
        if np.isfinite(dx_val):
            ax.plot([xL, xR], [target, target], lw=1.2, color="k", alpha=0.8)
            if annotate_dx:
                ax.text((xL + xR) / 2.0, target,
                        f"Δx={dx_val:.1f} nm",
                        ha="center", va="bottom", fontsize=9)

        if show_baseline:
            ax.axhline(baseline, ls=":", lw=0.9, color="tab:purple", alpha=0.7, label="baseline")

        ax.set_title(f"Profile {j+1}")
        ax.set_ylabel("Height (nm)")
        ax.grid(alpha=0.25)

        # collect outputs
        out_xL.append(xL); out_xR.append(xR)
        out_xP.append(xP); out_yP.append(yP)
        out_base.append(baseline); out_hsub.append(hsub)
        out_dx.append(dx_val); out_target.append(target)

        # deduplicate legend entries per subplot
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="best", fontsize=9)

    axes[-1].set_xlabel("x (nm)")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi)

    results = {
        "xL": out_xL,
        "xR": out_xR,
        "xP": out_xP,
        "yP": out_yP,
        "baseline": out_base,
        "hsub": out_hsub,
        "dx_frac": out_dx,
        "target": out_target,
    }
    return fig, axes, results


def extract_hsub_and_dmin(x_cols: np.ndarray, y_cols: np.ndarray, frac: float = 0.2
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each profile (column) returns:
      hsub_nm: peak height above baseline, baseline = ½(min_L + min_R)
      dx20_nm: full width at `frac` (default 0.2) of hsub above the baseline
               i.e., distance between left/right x where y = baseline + frac*hsub

    Notes:
      - Uses first crossings moving outward from the peak on each side.
      - Linear interpolation is used between samples for sub-pixel accuracy.
      - If a crossing isn't found on either side, dx20 is NaN for that profile.
    """
    def _interp_cross(x, y, i, j, target):
        x1, x2 = float(x[i]), float(x[j])
        y1, y2 = float(y[i]), float(y[j])
        if y2 == y1:
            return (x1 + x2) * 0.5
        t = (target - y1) / (y2 - y1)
        t = np.clip(t, 0.0, 1.0)
        return x1 + t * (x2 - x1)

    hsub, dx20 = [], []
    for x, y in zip(x_cols.T, y_cols.T):
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if y.size < 3:
            hsub.append(np.nan); dx20.append(np.nan); continue

        p = int(np.nanargmax(y))  # peak index

        # Baseline from minima to the left/right of the peak
        idxL_min = int(np.nanargmin(y[:p+1]))
        idxR_min = int(p + np.nanargmin(y[p:]))
        baseline = 0.5 * (float(y[idxL_min]) + float(y[idxR_min]))
        h = float(y[p]) - baseline
        hsub.append(h)

        if not np.isfinite(h) or h <= 0.0:
            dx20.append(np.nan); continue

        target = baseline + frac * h

        # Left crossing (scan outward from the peak)
        xL = np.nan
        for i in range(p-1, -1, -1):
            y_i, y_ip1 = y[i], y[i+1]
            if (y_i - target) == 0:
                xL = float(x[i]); break
            if (y_i - target) * (y_ip1 - target) < 0 or (y_ip1 - target) == 0:
                xL = _interp_cross(x, y, i, i+1, target)
                break

        # Right crossing
        xR = np.nan
        for i in range(p, len(y)-1):
            y_i, y_ip1 = y[i], y[i+1]
            if (y_i - target) == 0:
                xR = float(x[i]); break
            if (y_i - target) * (y_ip1 - target) < 0 or (y_ip1 - target) == 0:
                xR = _interp_cross(x, y, i, i+1, target)
                break

        dx20.append(abs(xR - xL) if np.isfinite(xL) and np.isfinite(xR) else np.nan)

    return np.asarray(hsub, float), np.asarray(dx20, float)


# ========================= 2D QUANTILE GRID =========================

def _safe_quantile_edges(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Quantile edges with safeguards:
      - ensures strictly increasing edges by tiny jitter if needed
      - falls back to linspace if variance is ~0
    """
    v = np.asarray(v)
    if not np.isfinite(v).any():
        return np.array([0.0, 1.0])
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.array([0.0, 1.0])
    if np.nanstd(finite) == 0:
        return np.linspace(float(finite.min()), float(finite.max()) + 1e-9, len(q))
    edges = np.quantile(finite, q)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def quantile_grid_edges(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    Qx: int = 6,
    Qy: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return quantile grid edges along diameter (x) and height (y)."""
    m = np.isfinite(dmin_nm) & np.isfinite(hsub_nm)
    x = dmin_nm[m]; y = hsub_nm[m]
    Qx = max(1, int(Qx)); Qy = max(1, int(Qy))
    qx = np.linspace(0, 1, Qx + 1)
    qy = np.linspace(0, 1, Qy + 1)
    x_edges = _safe_quantile_edges(x, qx)
    y_edges = _safe_quantile_edges(y, qy)
    return x_edges, y_edges


def summarize_pairs_quantile_grid_with_edges(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    reducer: Literal["median", "mean"] = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Summarize using provided quantile edges (ensures viz & summary match).

    Returns:
      diam_rep   : (C,)  representative diameters (nm)
      height_rep : (C,)  representative heights (nm)
      weight_rep : (C,)  non-empty cell weights (sum=1)
      labels     : (N,)  linearized cell index per original sample (for viz)
    """
    m = np.isfinite(dmin_nm) & np.isfinite(hsub_nm)
    x = dmin_nm[m]; y = hsub_nm[m]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    Qx = len(x_edges) - 1
    Qy = len(y_edges) - 1
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, Qx - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, Qy - 1)
    labels = yi + Qy * xi  # linear cell id per point

    diam_rep, height_rep, weight_rep = [], [], []
    N = x.size
    for i in range(Qx):
        for j in range(Qy):
            sel = (xi == i) & (yi == j)
            if not np.any(sel):
                continue
            if reducer == "median":
                d_rep = float(np.median(x[sel])); h_rep = float(np.median(y[sel]))
            else:
                d_rep = float(np.mean(x[sel]));   h_rep = float(np.mean(y[sel]))
            w = float(sel.sum()) / N
            diam_rep.append(d_rep); height_rep.append(h_rep); weight_rep.append(w)

    diam_rep   = np.asarray(diam_rep, dtype=float)
    height_rep = np.asarray(height_rep, dtype=float)
    weight_rep = np.asarray(weight_rep, dtype=float)

    # exact normalization
    s = weight_rep.sum()
    if s > 0:
        weight_rep /= s
        weight_rep[-1] += 1.0 - weight_rep.sum()

    return diam_rep, height_rep, weight_rep, labels


def summarize_pairs_quantile_grid(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    Qx: int = 6,
    Qy: int = 6,
    reducer: Literal["median", "mean"] = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper that computes edges then summarizes.
    Returns:
      diam_rep, height_rep, weight_rep  (weights sum to 1)
    """
    xe, ye = quantile_grid_edges(dmin_nm, hsub_nm, Qx=Qx, Qy=Qy)
    diam_rep, height_rep, weight_rep, _ = summarize_pairs_quantile_grid_with_edges(
        dmin_nm, hsub_nm, xe, ye, reducer=reducer
    )
    return diam_rep, height_rep, weight_rep


def quantile_cell_counts(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (counts_matrix, weights_matrix) for the quantile grid.
    weights_matrix sums to 1 over all cells.
    """
    m = np.isfinite(dmin_nm) & np.isfinite(hsub_nm)
    x, y = dmin_nm[m], hsub_nm[m]
    Qx = len(x_edges) - 1
    Qy = len(y_edges) - 1
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, Qx - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, Qy - 1)
    counts = np.zeros((Qx, Qy), dtype=int)
    for i, j in zip(xi, yi):
        counts[i, j] += 1
    weights = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
    return counts, weights

# ========================= K-MEDOIDS (PAM) =========================

def _zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization with epsilon guard (deterministic)."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    Z = (X - mu) / sigma
    return Z, mu, sigma

def _pairwise_euclidean(Z: np.ndarray) -> np.ndarray:
    """Dense pairwise Euclidean distance matrix (NxN)."""
    s = np.sum(Z**2, axis=1, keepdims=True)
    D2 = s + s.T - 2.0 * (Z @ Z.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, out=D2)

def _assign(D: np.ndarray, medoids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest and second-nearest medoid distances + labels."""
    M = D[:, medoids]
    order = np.argsort(M, axis=1)
    d1 = M[np.arange(M.shape[0]), order[:, 0]]
    d2 = M[np.arange(M.shape[0]), order[:, 1]] if M.shape[1] > 1 else np.full(M.shape[0], np.inf)
    labels = order[:, 0]
    return labels, d1, d2

def _pam_build(D: np.ndarray, k: int) -> np.ndarray:
    """BUILD phase: first medoid = min total distance; then greedy additions."""
    N = D.shape[0]
    total = np.sum(D, axis=1)
    medoids = [int(np.argmin(total))]
    d1 = D[:, medoids[0]].copy()
    while len(medoids) < k:
        best_gain = -np.inf; best_idx = None
        in_M = np.zeros(N, dtype=bool); in_M[medoids] = True
        for i in range(N):
            if in_M[i]:
                continue
            gain = np.sum(d1 - np.minimum(d1, D[:, i]))
            if gain > best_gain:
                best_gain = gain; best_idx = i
        medoids.append(int(best_idx))
        d1 = np.minimum(d1, D[:, best_idx])
    return np.asarray(medoids, dtype=int)

def _pam_swap(D: np.ndarray, medoids: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """SWAP phase: try medoid/non-medoid swaps that reduce total cost."""
    N = D.shape[0]
    medoids = medoids.copy()
    in_M = np.zeros(N, dtype=bool); in_M[medoids] = True
    for _ in range(max_iter):
        labels, d1, d2 = _assign(D, medoids)
        current_cost = float(np.sum(d1))
        improved = False
        for mi, m in enumerate(medoids):
            for h in range(N):
                if in_M[h]:
                    continue
                Djh = D[:, h]
                use_d2 = (labels == mi)
                new_d = np.where(use_d2, np.minimum(d2, Djh), np.minimum(d1, Djh))
                new_cost = float(np.sum(new_d))
                if new_cost + 1e-12 < current_cost:
                    in_M[m] = False; in_M[h] = True
                    medoids[mi] = h
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return medoids

def kmedoids_pam(
    X: np.ndarray,
    k: int,
    scale: bool = True,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic K-medoids (PAM).
    Returns:
      medoid_indices : (k,) indices into X of chosen medoids
      labels         : (N,) cluster label per point (0..k-1)
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    k = max(1, min(int(k), N))
    Z = _zscore(X)[0] if scale else X.copy()
    D = _pairwise_euclidean(Z)
    medoids = _pam_build(D, k)
    medoids = _pam_swap(D, medoids, max_iter=max_iter)
    labels, _, _ = _assign(D, medoids)
    return medoids, labels

def summarize_pairs_kmedoids(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    K: int = 25,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce paired samples to K representative *observed* particles via K-medoids.
    Returns:
      diam_rep  : (K_actual,) nm  medoid diameters (from raw data)
      height_rep: (K_actual,) nm  medoid heights (from raw data)
      weight_rep: (K_actual,)     cluster weights (sum=1)
      labels    : (N_filtered,)    cluster label per original sample (for viz)
    """
    m = np.isfinite(dmin_nm) & np.isfinite(hsub_nm)
    X = np.column_stack([dmin_nm[m], hsub_nm[m]])
    if len(X) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    medoid_idx, labels = kmedoids_pam(X, k=K, scale=scale)
    # representatives are real observed pairs
    diam_rep   = X[medoid_idx, 0]
    height_rep = X[medoid_idx, 1]
    # weights by occupancy
    uniq, counts = np.unique(labels, return_counts=True)
    weight_rep = counts.astype(float) / counts.sum()
    # exact normalization
    s = weight_rep.sum()
    if s > 0:
        weight_rep /= s
        weight_rep[-1] += 1.0 - weight_rep.sum()
    return diam_rep, height_rep, weight_rep, labels

# ========================= Visualization =========================

def visualize_quantile_grid(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    x_edges: Optional[np.ndarray] = None,
    y_edges: Optional[np.ndarray] = None,
    Qx: int = 6,
    Qy: int = 6,
    reducer: Literal["median", "mean"] = "median",
    figsize: Tuple[float, float] = (7.5, 6.0),
    scatter_alpha: float = 0.35,
    annotate_weights: bool = False,
    show: bool = True,
):
    """
    Plot raw points, overlay quantile grid, and draw representatives sized by weight.
    Returns: (fig, ax, (diam_rep, height_rep, weight_rep), (x_edges, y_edges))
    """
    import matplotlib.pyplot as plt

    m = np.isfinite(dmin_nm) & np.isfinite(hsub_nm)
    x = dmin_nm[m]; y = hsub_nm[m]

    if x_edges is None or y_edges is None:
        x_edges, y_edges = quantile_grid_edges(x, y, Qx=Qx, Qy=Qy)

    # summarize using the same edges we visualize
    diam_rep, height_rep, weight_rep, _ = summarize_pairs_quantile_grid_with_edges(
        x, y, x_edges, y_edges, reducer=reducer
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=16, alpha=scatter_alpha, color="tab:gray", edgecolors="none", label="Raw data")

    for xv in x_edges[1:-1]:
        ax.axvline(xv, color="k", lw=0.8, alpha=0.25)
    for yv in y_edges[1:-1]:
        ax.axhline(yv, color="k", lw=0.8, alpha=0.25)

    if weight_rep.size:
        w = weight_rep / weight_rep.max()
        sizes = 600 * np.sqrt(w)
        ax.scatter(diam_rep, height_rep, s=sizes, color="tab:blue",
                   edgecolors="white", linewidths=0.9, label="Grid reps")
        if annotate_weights:
            for dx, hy, ww in zip(diam_rep, height_rep, weight_rep):
                ax.text(dx, hy, f"{100*ww:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Δx between minima (nm)  [≈ diameter]")
    ax.set_ylabel("Background-subtracted height (nm)")
    ax.set_title(f"2D Quantile Grid (Qx={len(x_edges)-1}, Qy={len(y_edges)-1})")
    ax.grid(alpha=0.2); ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax, (diam_rep, height_rep, weight_rep), (x_edges, y_edges)


def visualize_kmedoids(
    dmin_nm: np.ndarray,
    hsub_nm: np.ndarray,
    diam_rep: np.ndarray,
    height_rep: np.ndarray,
    labels: np.ndarray,
    weight_rep: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (7.5, 6.0),
    annotate_weights: bool = True,
    show: bool = True,
):
    """
    Scatter raw points colored by cluster; overlay medoids (stars) sized by weight.
    Returns: (fig, ax)
    """
    import matplotlib.pyplot as plt

    m = np.isfinite(dmin_nm) & np.isfinite(hsub_nm)
    x = dmin_nm[m]; y = hsub_nm[m]
    if labels.shape[0] != x.shape[0]:
        raise ValueError("labels length must match number of finite raw pairs.")

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x, y, c=labels, s=16, alpha=0.45, cmap="tab20", edgecolors="none", label="Raw data")

    if weight_rep is None or len(weight_rep) == 0:
        sizes = 300.0
    else:
        w = weight_rep / weight_rep.max()
        sizes = 800.0 * np.sqrt(w)

    ax.scatter(diam_rep, height_rep, marker="*", s=sizes, c="k",
               edgecolors="white", linewidths=0.9, label="Medoids")

    if annotate_weights and (weight_rep is not None) and len(weight_rep) == len(diam_rep):
        for dx, hy, ww in zip(diam_rep, height_rep, weight_rep):
            ax.text(dx, hy, f"{100*ww:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Δx between minima (nm)  [≈ diameter]")
    ax.set_ylabel("Background-subtracted height (nm)")
    ax.set_title(f"K-medoids summary (K={len(diam_rep)})")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Cluster")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax