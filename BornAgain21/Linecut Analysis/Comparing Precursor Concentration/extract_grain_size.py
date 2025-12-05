# Utility to plot the full line profile first (so you can choose anchor points),
# plus a linear-baseline subtraction using user-provided anchors and x-limits.
# This cell will:
# 1) Load your uploaded linecut file
# 2) Plot the full profile (linear y) so you can visually pick anchors
#
# After you pick anchors, you can call `subtract_linear_baseline(...)` with your choices.

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Literal, Optional

# ---------- Core function ----------
def subtract_linear_baseline(
    x: np.ndarray,
    y: np.ndarray,
    points: Sequence[float],
    points_kind: Literal["x", "index"] = "x",
    xlim: Optional[Tuple[float, float]] = None,
):
    """
    Subtract a linear baseline defined by user-selected points, then crop by x-limits.
    See docstring in the previous message for details.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same length.")
    if len(points) < 2:
        raise ValueError("Provide at least two points to define the baseline.")

    if points_kind == "x":
        xp = np.asarray(points, dtype=float)
        yp = np.interp(xp, x, y, left=np.nan, right=np.nan)
        m = np.isfinite(yp)
        if m.sum() < 2:
            raise ValueError("At least two selected x-points must lie within the data range.")
        xp, yp = xp[m], yp[m]
    elif points_kind == "index":
        idx = np.asarray(points, dtype=int)
        if np.any((idx < 0) | (idx >= x.size)):
            raise ValueError("Index in 'points' is out of bounds.")
        xp = x[idx].astype(float)
        yp = y[idx].astype(float)
    else:
        raise ValueError("points_kind must be 'x' or 'index'.")

    # Crop window
    if xlim is not None:
        xmin, xmax = map(float, xlim)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        mask = (x >= xmin) & (x <= xmax)
    else:
        mask = np.ones_like(x, dtype=bool)

    # Fit line from anchors (2-point exact or LS fit for >2)
    if xp.size == 2:
        x1, x2 = float(xp[0]), float(xp[1])
        y1, y2 = float(yp[0]), float(yp[1])
        if np.isclose(x2, x1):
            raise ValueError("Selected points must have different x to define a line.")
        m_line = (y2 - y1) / (x2 - x1)
        b_line = y1 - m_line * x1
    else:
        A = np.column_stack([xp, np.ones_like(xp)])
        m_line, b_line = np.linalg.lstsq(A, yp, rcond=None)[0]

    xw = x[mask]
    yw = y[mask]
    baseline = m_line * xw + b_line
    ycorr = yw - baseline
    return xw, yw, baseline, ycorr

def fwhm_from_peak(x: np.ndarray, ycorr: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute FWHM of a single, positive peak in (x, ycorr).

    Returns
    -------
    fwhm : float
        Full width at half maximum (same x-units as input).
    x_left : float
        Left half-maximum crossing (interpolated).
    x_right : float
        Right half-maximum crossing (interpolated).
    x_peak : float
        Peak position (x at maximum of ycorr).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(ycorr).ravel()

    if x.size != y.size or x.ndim != 1:
        raise ValueError("x and ycorr must be 1D arrays of the same length.")

    # Use the *local* maximum within the window
    i_max = int(np.nanargmax(y))
    y_max = float(y[i_max])
    if not np.isfinite(y_max) or y_max <= 0:
        raise ValueError("Peak maximum must be positive and finite for FWHM.")

    half = 0.5 * y_max

    # ---- Left crossing (search left from peak) ----
    i_left = None
    for i in range(i_max, 0, -1):
        if y[i-1] <= half <= y[i]:
            # linear interpolation between points (i-1) and i
            frac = (half - y[i-1]) / (y[i] - y[i-1] + 1e-300)
            x_left = x[i-1] + frac * (x[i] - x[i-1])
            i_left = i
            break
    if i_left is None:
        raise RuntimeError("Could not find left half-maximum crossing inside the window.")

    # ---- Right crossing (search right from peak) ----
    i_right = None
    for i in range(i_max, x.size - 1):
        if y[i] >= half >= y[i+1]:
            frac = (half - y[i]) / (y[i+1] - y[i] + 1e-300)
            x_right = x[i] + frac * (x[i+1] - x[i])
            i_right = i
            break
    if i_right is None:
        raise RuntimeError("Could not find right half-maximum crossing inside the window.")

    fwhm = float(x_right - x_left)
    return fwhm, x_left, x_right, float(x[i_max])
# ---------- Load your data and show a full plot ----------
data_path = "C:\BornAgainSimulations/lineprofile_linecut_I_15 deg.txt"
xy = np.loadtxt(data_path)
x = xy[:, 0]
y = xy[:, 1]

plt.figure(figsize=(9, 4.8))
plt.plot(x, y)
plt.xlim(0.05,1)
#plt.ylim(33, 75000)
#plt.yscale('log')
plt.xlabel("x")
plt.ylabel("Intensity (a.u.)")
plt.title("Full line profile – pick two (or more) anchor points for the baseline")
plt.tight_layout()

# Tip (after you choose anchors, run something like):
xw, yw, base, ycorr = subtract_linear_baseline(
     x, y,
     points=[0.0802, 0.1894],    # <-- your chosen anchor x-positions
     points_kind="x",
     xlim=(0.078, 0.1894)      # <-- your crop limits
 )

fwhm, xL, xR, x0 = fwhm_from_peak(xw, ycorr)
print(f"Peak center ~ {x0:.6f}, FWHM ~ {fwhm:.6f}, crossings at {xL:.6f} and {xR:.6f}")
# And then plot the result:
plt.figure(); plt.plot(xw, yw); plt.plot(xw, base); plt.plot(xw, ycorr); 
#plt.yscale('log')
plt.xlabel("x")
plt.ylabel("Intensity (a.u.)")
plt.title("Full line profile – pick two (or more) anchor points for the baseline")
plt.show()