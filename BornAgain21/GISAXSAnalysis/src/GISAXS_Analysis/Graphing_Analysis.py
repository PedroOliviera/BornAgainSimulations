from GISAXS_Analysis import GISAXS_setup_v21 as g
from matplotlib import pyplot as plt
from bornagain import ba_plot as bp
import bornagain as ba
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import numpy as np
from typing import Optional, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Roman numerals for subplot titles
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V"]

import numpy as np
from typing import Optional, Tuple, Union

def _ensure_ascending(x: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make x ascending; flip image columns accordingly if needed."""
    x = np.asarray(x, float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if np.any(np.diff(x) < 0):
        return x[::-1], img[:, ::-1]
    return x, img

def _nearest_index(x: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(x - value)))

def stitch_detector_halves(
    img_left: np.ndarray,
    img_right: np.ndarray,
    x_left: Optional[np.ndarray] = None,
    x_right: Optional[np.ndarray] = None,
    *,
    zero_from: str = "right",   # which side keeps the x=0 column: "left" or "right"
    return_x: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Stitch two detector images along x = 0.

    If x_left/x_right are provided:
        - Keep full negative side from the LEFT image:   x_left < 0 (or <= 0 if zero_from='left')
        - Keep full positive side from the RIGHT image:  x_right > 0 (or >= 0 if zero_from='right')
        - Concatenate along columns (x).
        - Return the stitched x-array if return_x=True.

    If x_left/x_right are NOT provided:
        - Treat the middle column as x=0 for each image.
        - Keep left half of img_left and right half of img_right.
        - No physical x-array is returned unless you want pixel-centered coords.

    Requirements:
        - img_left.shape[0] == img_right.shape[0] (same number of rows).
        - If x arrays are given, their lengths must match the number of columns of the respective image.
    """
    img_left  = np.asarray(img_left,  float)
    img_right = np.asarray(img_right, float)

    if img_left.ndim != 2 or img_right.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")
    if img_left.shape[0] != img_right.shape[0]:
        raise ValueError("Images must have the same number of rows (y).")

    def _ensure_ascending(x, img):
        x = np.asarray(x, float)
        if x.ndim != 1 or x.size != img.shape[1]:
            raise ValueError("x must be 1D with length equal to number of columns.")
        if np.any(np.diff(x) < 0):
            # flip horizontally to make x increasing
            return x[::-1], img[:, ::-1]
        return x, img

    if x_left is not None and x_right is not None:
        # align x orientation with columns
        x_left,  img_left  = _ensure_ascending(x_left,  img_left)
        x_right, img_right = _ensure_ascending(x_right, img_right)

        # decide which side owns x=0
        if zero_from not in {"left", "right"}:
            raise ValueError("zero_from must be 'left' or 'right'.")

        if zero_from == "left":
            left_mask  = (x_left <= 0.0)
            right_mask = (x_right >  0.0)
        else:  # zero_from == "right"
            left_mask  = (x_left <  0.0)
            right_mask = (x_right >= 0.0)

        left_cols  = np.where(left_mask)[0]
        right_cols = np.where(right_mask)[0]
        if left_cols.size == 0:
            raise ValueError("No columns with x<=(or<)0 in left image.")
        if right_cols.size == 0:
            raise ValueError("No columns with x>=(or>)0 in right image.")

        imgL = img_left[:, left_cols]
        imgR = img_right[:, right_cols]
        xL   = x_left[left_cols]
        xR   = x_right[right_cols]

        img_stitched = np.concatenate([imgL, imgR], axis=1)
        x_stitched   = np.concatenate([xL, xR])

        return (img_stitched, x_stitched) if return_x else (img_stitched, None)

    # -------- No x arrays provided: split by middle column (treat as x=0) --------
    ncols_L = img_left.shape[1]
    ncols_R = img_right.shape[1]

    # If even, middle is the boundary between the two central columns.
    # If odd, we give the center column to `zero_from`.
    mid_L = ncols_L // 2
    mid_R = ncols_R // 2

    if zero_from == "left":
        # left keeps its middle column as x=0
        imgL = img_left[:, :mid_L + (ncols_L % 2)]
        imgR = img_right[:, mid_R + (ncols_R % 2):]
    else:  # "right"
        # right keeps its middle column as x=0
        imgL = img_left[:, :mid_L]
        imgR = img_right[:, mid_R:]

    img_stitched = np.concatenate([imgL, imgR], axis=1)

    if return_x:
        # fabricate pixel-centered x with 0 at the stitch point
        nL = imgL.shape[1]
        nR = imgR.shape[1]
        x_left_pix  = np.arange(-nL, 0, dtype=float)
        x_right_pix = np.arange(0, nR, dtype=float)
        x_stitched  = np.concatenate([x_left_pix, x_right_pix])
        return img_stitched, x_stitched

    return img_stitched, None


def normalize2d_by_max(
    img: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,  # (row_start, row_end, col_start, col_end), end exclusive
    mask: Optional[np.ndarray] = None,                # boolean mask same shape as img; True = included
    clip_negatives: bool = False,                     # clip <0 to 0 after normalization
    log_floor: Optional[float] = None,                # e.g., 1e-12 to be log-safe; applied after negative clipping
    return_scale: bool = False,                       # also return the max used for normalization
) -> np.ndarray | Tuple[np.ndarray, float]:
    """
    Normalize a 2D array by the maximum FINITE pixel value.

    - If `roi` is provided, the max is computed within that rectangular region.
    - If `mask` is provided, the max is computed over True-valued pixels.
      (If both roi and mask are given, they are combined: ROI ∩ mask.)
    - NaNs/±inf are ignored when computing the max and remain as-is in the output.
    - Optionally clips negatives to 0 and/or applies a tiny positive floor for log plots.

    Raises
    ------
    ValueError
        If no finite pixels are found in the selected region, or if the max ≤ 0.

    Returns
    -------
    arr_norm or (arr_norm, scale)
        The normalized array (float64). If `return_scale=True`, also returns the max used.
    """
    arr = np.asarray(img, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array.")

    # Build a selection mask of finite values
    finite = np.isfinite(arr)

    # Apply ROI if given
    if roi is not None:
        r0, r1, c0, c1 = roi
        sel = np.zeros_like(finite, dtype=bool)
        sel[max(r0, 0):max(r0, 0) + (r1 - r0), max(c0, 0):max(c0, 0) + (c1 - c0)] = True
        finite &= sel

    # Apply mask if given
    if mask is not None:
        if mask.shape != arr.shape:
            raise ValueError("mask must have the same shape as img.")
        finite &= mask.astype(bool)

    if not np.any(finite):
        raise ValueError("No finite pixels found in the selected region.")

    scale = np.nanmax(arr[finite])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Max finite pixel is non-positive ({scale}); cannot normalize.")

    out = arr / scale

    if clip_negatives:
        out = np.maximum(out, 0.0)
    if log_floor is not None:
        out = np.clip(out, log_floor, None)

    return (out, scale) if return_scale else out

def normalize_by_first_peak(
    x,
    y,
    x_min,
    x_max,
):
    """
    Normalize y by the height of the *first* peak whose x-coordinate lies
    within [x_min, x_max].

    Parameters
    ----------
    x, y : array-like
        1-D coordinate and signal arrays of equal length.
    x_min, x_max : float
        Inclusive x-range in which to look for the peak.  Defaults to
        0.14 ≤ x ≤ 0.16.

    Returns
    -------
    x_out, y_norm : ndarray, ndarray
        (Possibly sorted) x array and y divided by the chosen peak height.

    Raises
    ------
    ValueError
        If the range contains no points or the peak height is zero.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # sort by x if necessary
    if np.any(np.diff(x) < 0):
        order = np.argsort(x)
        x, y = x[order], y[order]

    # indices inside the specified window
    in_window = np.where((x >= x_min) & (x <= x_max))[0]
    if in_window.size == 0:
        raise ValueError(f"No data points with {x_min} ≤ x ≤ {x_max}")

    peak_idx = in_window[np.nanargmax(y[in_window])]

    peak_height = y[peak_idx]
    if peak_height == 0:
        raise ValueError("Peak height is zero; cannot normalize.")

    # fallback: pick highest point in the window
    if peak_idx is None:
        peak_idx = in_window[np.nanargmax(y[in_window])]

    peak_height = y[peak_idx]
    if peak_height == 0:
        raise ValueError("Peak height is zero; cannot normalize.")
    return x, y / peak_height

def plot_qy_linecut(ax, qy, simulation_data, experimental_data, axes_sim, axes_exp, labels, linecut_index, save, savefname):
    ax.set_title(f'Linecut {ROMAN_NUMERALS[linecut_index]}')

    horizontal_slice_1 = qy + 0.0001
    horizontal_slice_2 = qy - 0.0001

    if simulation_data is not None:
        x, y = g.integrate_plt_slices(
            start=horizontal_slice_2,
            stop=horizontal_slice_1,
            data=simulation_data,
            axLim=axes_sim,
            labelname="Simulation",
            num=1,
            horiz_slice=True
        )
        ax.plot(x, y, label=labels[0])

    if experimental_data is not None:
        x, y = g.integrate_plt_slices(
            start=horizontal_slice_2,
            stop=horizontal_slice_1,
            data=experimental_data,
            axLim=axes_exp,
            labelname="Experiment",
            num=1,
            horiz_slice=True
        )
        ax.plot(x, y, label=labels[1])

    if save is True:
        np.savez(f'lineprofile_linecut_{ROMAN_NUMERALS[linecut_index]}_{savefname}.npz', x=x, y=y, x_unit="1/nm", y_unit="a.u.")
        save_data = np.column_stack((x, y))
        np.savetxt(f'lineprofile_linecut_{ROMAN_NUMERALS[linecut_index]}_{savefname}.txt', save_data)

    ax.set_ylabel("Intensity")
    ax.set_xlabel(r'$Q_{y} \;(1/{\rm nm})$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(bottom=10)
    ax.legend()

def plot_qz_linecut(ax, qz, simulation_data, experimental_data, axes_sim, axes_exp, labels, linecut_index, save, savefname):
    ax.set_title(f'Linecut {ROMAN_NUMERALS[linecut_index]}')

    vertical_slice_1 = qz + 0.0001
    vertical_slice_2 = qz - 0.0001

    x_data_all, y_data_all = [], []

    if simulation_data is not None:
        x, y = g.integrate_plt_slices(
            start=vertical_slice_2,
            stop=vertical_slice_1,
            data=simulation_data,
            axLim=axes_sim,
            labelname="Simulation",
            num=1,
            vert_slice=True
        )
        ax.plot(x, y, label=labels[0])
        x_data_all.extend(x)
        y_data_all.extend(y)

    if experimental_data is not None:
        x, y = g.integrate_plt_slices(
            start=vertical_slice_2,
            stop=vertical_slice_1,
            data=experimental_data,
            axLim=axes_exp,
            labelname="Experiment",
            num=1,
            vert_slice=True
        )
        ax.plot(x, y, label=labels[1])
        x_data_all.extend(x)
        y_data_all.extend(y)
    
    if save is True:
        np.savez(f'lineprofile_linecut_{ROMAN_NUMERALS[linecut_index]}_{savefname}.npz', x=x, y=y, x_unit="1/nm", y_unit="a.u.")
        save_data = np.column_stack((x, y))
        np.savetxt(f'lineprofile_linecut_{ROMAN_NUMERALS[linecut_index]}_{savefname}.txt', save_data)

    ax.set_ylabel("Intensity")
    ax.set_xlabel(r'$Q_{z} \;(1/{\rm nm})$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(bottom=10)
    if x_data_all:
        ax.set_xlim(min(x_data_all), max(x_data_all))
    if y_data_all:
        ax.set_ylim(min(y_data_all), max(y_data_all) * 1.1)
    ax.legend()

def linecutsItoV(
    simulation_data=None,
    experimental_data=None,
    L1_qz=None, L2_qy=None, L3_qz=None, L4_qy=None, L5_qz=None,
    axes_exp =None,
    axes_sim = None,
    labels=("Simulation", "Experiment"),
    title="",
    save = False,
    savefname =''
):
    if simulation_data is None and experimental_data is None:
        print("No data provided.")
        return

    # Define active linecuts
    linecuts = [
        ("qy", L1_qz),
        ("qz", L2_qy),
        ("qy", L3_qz),
        ("qz", L4_qy),
        ("qy", L5_qz)
    ]
    active_linecuts = [(kind, val) for kind, val in linecuts if val is not None]
    all_linecuts = [(kind, val) for kind, val in linecuts]
    n = len(active_linecuts)

    if n == 0:
        print("No linecuts defined.")
        return

    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axs = [axs]
    j = 0
    for i, (kind, val) in enumerate(all_linecuts):
        if val is None:
            continue
        elif kind == "qy":
            plot_qy_linecut(axs[j], val, simulation_data, experimental_data, axes_sim, axes_exp, labels, i, save, savefname)
            j+=1
        elif kind == "qz":
            plot_qz_linecut(axs[j], val, simulation_data, experimental_data, axes_sim, axes_exp, labels, i, save, savefname)
            j+=1

    fig.suptitle("Simulation: " + title, fontsize=16)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

def plot2D(
    realData=None,
    simulationData=None,
    realDat_axes=None,
    simData_axes=None,
    graphed_axes=[-1, 1, 0, 2],
    L1_qz=None, L2_qy=None, L3_qz=None, L4_qy=None, L5_qz=None,
    title="",
    zlim=[22, 5e5]
):
    if simulationData is None and realData is None:
        print("No data provided.")
        return None, None

    datasets = []
    axes_list = []
    if simulationData is not None:
        datasets.append(("Simulation", simulationData))
        axes_list.append(simData_axes)
        graphed_axes = simData_axes
    if realData is not None:
        datasets.append(("Experiment", realData))
        axes_list.append(realDat_axes)

    n = len(datasets)
    plt.figure(figsize=(7.5 * n, 6))

    vert_lines = [(L2_qy, 'II'), (L4_qy, 'IV')]
    horiz_lines = [(L1_qz, 'I'), (L3_qz, 'III'), (L5_qz, 'V')]

    exp_ax = None
    sim_ax = None

    for i, ((label, data), axes) in enumerate(zip(datasets, axes_list)):
        plt.subplot(1, n, i + 1)

        im = bp.plot_array(
            data,
            axes_limits=axes,
            intensity_min=zlim[0],
            intensity_max=zlim[1],
            xlabel=r'$Q_{y} \;(1/{\rm nm})$',
            ylabel=r'$Q_{z} \;(1/{\rm nm})$',
            zlabel=None,
            with_cb=True,
            cmap='gist_ncar'
        )
        ax = im.axes  # or: ax = plt.gca()
        if label == "Experiment":
            exp_ax = ax
        else:
            sim_ax = ax

        ax.set_title(label if label == "Experiment" else f"{label}: {title}", fontsize=14)
        ax.set_xlim(graphed_axes[0], graphed_axes[1])
        ax.set_ylim(graphed_axes[2], graphed_axes[3])
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)

        for qy, roman in vert_lines:
            if qy is not None:
                ax.axvline(x=qy, color='red', linewidth=1)
                ax.text(qy, graphed_axes[2], f'{roman}', color='red',
                        fontsize=12, ha='center', va='bottom', rotation=90)

        for qz, roman in horiz_lines:
            if qz is not None:
                ax.axhline(y=qz + 0.005, color='blue', linewidth=1)
                ax.axhline(y=qz - 0.005, color='red', linewidth=1)
                ax.text(graphed_axes[0], qz, f'{roman}', color='black',
                        fontsize=12, ha='left', va='center')

    plt.tight_layout()
    return exp_ax, sim_ax

def plot2D_simulationComparison(
    realData=None,
    simulationData=None,
    realDat_axes=None,
    simData_axes = None,
    graphed_axes=[-1, 1, 0, 2],
    L1_qz=None, L2_qy=None, L3_qz=None, L4_qy=None, L5_qz=None,
    title="",
    zlim=[22, 5e5]
):
    if simulationData is None and realData is None:
        print("No data provided.")
        return

    datasets = []
    axes_list = []

    if simulationData is not None:
        datasets.append(("Custom", simulationData))
        axes_list.append(simData_axes) #g2.get_axes_limits(result, ba.Coords_QSPACE))
        graphed_axes = simData_axes # g2.get_axes_limits(result, ba.Coords_QSPACE)
    if realData is not None:
        datasets.append(("CosineRippleGauss", realData))
        axes_list.append(realDat_axes)

    n = len(datasets)
    plt.figure(figsize=(7.5 * n, 6))

    # Linecut definitions with labels
    vert_lines = [(L2_qy, 'II'), (L4_qy, 'IV')]
    horiz_lines = [(L1_qz, 'I'), (L3_qz, 'III'), (L5_qz, 'V')]

    for i, ((label, data), axes) in enumerate(zip(datasets, axes_list)):
        plt.subplot(1, n, i + 1)

        im = bp.plot_array(
            data,
            axes_limits=axes,
            intensity_min=zlim[0],
            intensity_max=zlim[1],
            xlabel=r'$Q_{y} \;(1/{\rm nm})$',
            ylabel=r'$Q_{z} \;(1/{\rm nm})$',
            zlabel=None,
            with_cb=True,
            cmap='gist_ncar'
        )
        ax = im.axes
        ax.set_title(label if label == "Experiment" else f"{label}: {title}", fontsize=14)

        ax.set_xlim(graphed_axes[0], graphed_axes[1])
        ax.set_ylim(graphed_axes[2], graphed_axes[3])
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)

        # Draw vertical (qy) linecuts
        for qy, roman in vert_lines:
            if qy is not None:
                ax.axvline(x=qy, color='red', linewidth=1)
                ax.text(qy, graphed_axes[2], f'{roman}', color='red',
                        fontsize=12, ha='center', va='bottom', rotation=90)

        # Draw horizontal (qz) linecuts
        for qz, roman in horiz_lines:
            if qz is not None:
                ax.axhline(y=qz + 0.005, color='blue', linewidth=1)
                ax.axhline(y=qz - 0.005, color='red', linewidth=1)
                ax.text(graphed_axes[0], qz, f'{roman}', color='black',
                        fontsize=12, ha='left', va='center')

    plt.tight_layout()
    plt.show()

def yonedaPlot(vert_slice_q, data_npArrays, data_axes_array, data2_npArray=None, data_axes2=None, xmin = float, xmax = float, labels = None):
    """Inputs:
    vert_slice_q: will take max of this vert slice value and use for horizontal slice value
    data_npArrays: array of dataset to be compared
    data_axes: axes of data (g2.get_axes_limits(result, ba.Coords_QSPACE) for simulation) and realData_axes_month for experimental data
    data2_npArrays: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    data2_axes2: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    """
    plt.figure(figsize=(7,5))

    n_datasets = len(data_npArrays)
    cmap = cm.get_cmap("jet", n_datasets)  # evenly spaced colors from jet colormap

    for i, (data, data_axes) in enumerate(zip(data_npArrays, data_axes_array)):
        # Take vertical slice at Qz = vert_slice_q
        x1, y1 = g.plot_slices(data, axesLimits=data_axes, vert_slice=vert_slice_q)
        ind_x1 = np.argmax(y1)         # position of maximum intensity
        hor_slice_q = x1[ind_x1]       # Qy value of that maximum

        step = 0.01
        #hor_slice_q = 0.3
        # Now take horizontal slice through that maximum
        x2, y2 = g.plot_slices(data, axesLimits=data_axes, horiz_slice=hor_slice_q)
        x2, y2 = g.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data, axLim=data_axes, labelname=i, num=20, horiz_slice=True)
        xmin = 0.085
        xmax = 0.137
        #x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = xmin, x_max=xmax)
        x2_norm, y2_norm = x2, y2
        
        y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Get color from colormap
        color = cmap(i)

        # Plot with label and custom color
        plt.plot(x2_norm, y_s, label = labels[i], color=color)

    if data2_npArray is not None:
        x1, y1 = g.plot_slices(data2_npArray, axesLimits=data_axes2, vert_slice=vert_slice_q)
        ind_x1 = np.argmax(y1)         # position of maximum intensity
        hor_slice_q = x1[ind_x1]       # Qy value of that maximum
        
        step = 0.01
        #hor_slice_q = 0.4
        # Now take horizontal slice through that maximum
        x2, y2 = g.plot_slices(data2_npArray, axesLimits=data_axes2, horiz_slice=hor_slice_q)
        x2, y2 = g.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data2_npArray, axLim=data_axes2, labelname=i, num=20, horiz_slice=True)
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = 0.085, x_max=0.137)

        y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Plot with label and custom color
        plt.plot(x2_norm, y_s, label = "Experiment", color='black')


    # Improve legend and axis formatting
    plt.legend(title="Form Factor", fontsize=9, ncol=2)  # 2-column legend if many datasets
    plt.ylim(bottom=2e-6)
    plt.xlim(left=0.055)
    plt.ylabel("Normalized Intensity", fontsize=11)
    plt.xlabel(r"$Q_{y}\;(1/{\rm nm})$", fontsize=11)
    plt.title(rf"Horizontal Slices Along $Q_{{z}}$", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both", ls="--", lw=0.5, alpha=0.6)
    plt.tight_layout()

def hor_slice_comparison(hor_slice_q_array, data_npArrays, data_axes_array, data2_npArray=None, data_axes2=None, xmin = 0.0, xmax = 0.0, labels = None):
    """Inputs:
    vert_slice_q: will take max of this vert slice value and use for horizontal slice value
    data_npArrays: array of dataset to be compared
    data_axes: axes of data (g2.get_axes_limits(result, ba.Coords_QSPACE) for simulation) and realData_axes_month for experimental data
    data2_npArrays: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    data2_axes2: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    """
    plt.figure(figsize=(7,5))

    n_datasets = len(data_npArrays)
    #cmap = cm.get_cmap("rainbow", n_datasets)  # evenly spaced colors from jet colormap
    cmap = ['red', 'green', 'purple', 'blue', 'orange']
    for i, (data, data_axes, hor_slice_q) in enumerate(zip(data_npArrays, data_axes_array, hor_slice_q_array)):
        
        step = 0.01
        # Now take horizontal slice through that maximum
        x2, y2 = g.plot_slices(data, axesLimits=data_axes, horiz_slice=hor_slice_q)
        x2, y2 = g.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data, axLim=data_axes, labelname=i, num=20, horiz_slice=True)
       
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = xmin, x_max=xmax)
        #x2_norm, y2_norm = x2, y2
        
        #y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Get color from colormap
        color = cmap[i]#cmap(i)

        # Plot with label and custom color
        plt.plot(x2_norm, y2_norm, label = labels[i], color=color)

    if data2_npArray is not None:
        
        step = 0.01
        # Now take horizontal slice through that maximum
        x2, y2 = g.plot_slices(data2_npArray, axesLimits=data_axes2, horiz_slice=hor_slice_q)
        #x2, y2 = g.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data2_npArray, axLim=data_axes2, labelname=i, num=20, horiz_slice=True)
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = 0.085, x_max=0.137)

        y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Plot with label and custom color
        plt.plot(x2_norm, y_s, label = "Experiment", color='black')


    # Improve legend and axis formatting
    plt.legend(title="Form Factor", fontsize=9, ncol=2)  # 2-column legend if many datasets
    plt.ylim(bottom=2e-6)
    plt.xlim(left=0.055)
    plt.ylabel("Normalized Intensity", fontsize=11)
    plt.xlabel(r"$Q_{y}\;(1/{\rm nm})$", fontsize=11)
    plt.title(rf"Horizontal Slices Along $Q_{{z}}$", fontsize=12)
    plt.yscale("log")
    #plt.xscale("log")
    plt.grid(which="both", ls="--", lw=0.5, alpha=0.6)
    plt.tight_layout()

def vert_slice_comparison(vert_slice_q_array, data_npArrays, data_axes_array, data2_npArray=None, data_axes2=None, xmin = 0.0, xmax = 0.0, labels = None):
    """Inputs:
    vert_slice_q: will take max of this vert slice value and use for horizontal slice value
    data_npArrays: array of dataset to be compared
    data_axes: axes of data (g2.get_axes_limits(result, ba.Coords_QSPACE) for simulation) and realData_axes_month for experimental data
    data2_npArrays: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    data2_axes2: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    """
    plt.figure(figsize=(7,5))

    n_datasets = len(data_npArrays)
    cmap = cm.get_cmap("rainbow", n_datasets)  # evenly spaced colors from jet colormap
    #cmap = ['red', 'green', 'purple', 'blue', 'orange']
    for i, (data, data_axes, vert_slice_q) in enumerate(zip(data_npArrays, data_axes_array, vert_slice_q_array)):
        
        step = 0.001
        
        x2, y2 = g.plot_slices(data, axesLimits=data_axes, vert_slice=vert_slice_q)
        x2, y2 = g.integrate_plt_slices(start = vert_slice_q - step, stop= vert_slice_q + step, data=data, axLim=data_axes, labelname=i, num=20, vert_slice=True)
        y2 = savgol_filter(y2, window_length=30, polyorder=3, mode="interp")
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = 0.1, x_max=2.5)
        #x2_norm, y2_norm = x2, y2
        

        # Get color from colormap
        color = cmap(i)#cmap[i]

        # Plot with label and custom color
        plt.plot(x2_norm, y2_norm, label = labels[i], color=color)
        #plt.plot(y2_norm)
    if data2_npArray is not None:
        
        # Now take horizontal slice through that maximum
        x2, y2 = g.plot_slices(data2_npArray, axesLimits=data_axes2, vert_slice=vert_slice_q)
        x2, y2 = g.integrate_plt_slices(start = vert_slice_q - step, stop= vert_slice_q + step, data=data2_npArray, axLim=data_axes2, labelname=i, num=20, vert_slice=True)
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = xmin, x_max=xmax)

        y_s = savgol_filter(y2_norm, window_length=30, polyorder=3, mode="interp")

        # Plot with label and custom color
        plt.plot(x2_norm, y_s, label = "Experiment", color='black')


    # Improve legend and axis formatting
    plt.legend(title="Form Factor", fontsize=9, ncol=2)  # 2-column legend if many datasets
    plt.ylim(bottom=2e-6)
    #plt.xlim(left=0.055)
    plt.ylabel("Normalized Intensity", fontsize=11)
    plt.xlabel(r"$Q_{y}\;(1/{\rm nm})$", fontsize=11)
    plt.title(rf"Horizontal Slices Along $Q_{{z}}$", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both", ls="--", lw=0.5, alpha=0.6)
    plt.tight_layout()

def vert_slice_linecut_max_finder(vert_slice_q_array, data_npArrays, data_axes_array, data2_npArray=None, data_axes2=None, xmin = 0.0, xmax = 0.0, labels = None):
    """Inputs:
    vert_slice_q: will take max of this vert slice value and use for horizontal slice value
    data_npArrays: array of dataset to be compared
    data_axes: axes of data (g2.get_axes_limits(result, ba.Coords_QSPACE) for simulation) and realData_axes_month for experimental data
    data2_npArrays: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    data2_axes2: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    """
    plt.figure(figsize=(7,5))

    n_datasets = len(data_npArrays)
    cmap = cm.get_cmap("rainbow", n_datasets)  # evenly spaced colors from jet colormap
    #cmap = ['red', 'green', 'purple', 'blue', 'orange']
    for i, (data, data_axes, vert_slice_q) in enumerate(zip(data_npArrays, data_axes_array, vert_slice_q_array)):
        
        step = 0.001
        
        x2, y2 = g.plot_slices(data, axesLimits=data_axes, vert_slice=vert_slice_q)
        x2, y2 = g.integrate_plt_slices(start = vert_slice_q - step, stop= vert_slice_q + step, data=data, axLim=data_axes, labelname=i, num=20, vert_slice=True)
        #y2 = savgol_filter(y2, window_length=30, polyorder=3, mode="interp")
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = 0.1, x_max=2.5)
        

        peaks, _ = find_peaks(y2_norm, width = [100,500], prominence=[0.001, 1000])
        
        plt.plot(y2_norm)
        plt.plot(peaks, x2_norm[peaks], "x")

def find_two_minima_and_midmax(
    x, y,
    x_range=None,           # (xmin, xmax) or None
    x_unit="",              # label for plot
    fwhm_frac=0.075,        # Gaussian FWHM as fraction of x-span
    min_width_frac=0.06,    # min trough width (fraction of x-span)
    min_dist_frac=0.18,     # min spacing between troughs (fraction of x-span)
    prom_frac=0.006,        # min prominence as fraction of inverted-range
    plot=True, ax=None,qz=None
):
    # 1) prep
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if np.any(np.diff(x) <= 0):
        idx = np.argsort(x); x, y = x[idx], y[idx]
    if x_range is not None:
        xmin, xmax = x_range
        m = (x >= xmin) & (x <= xmax)
        x, y = x[m], y[m]
    if x.size < 10:
        print("Warning: Too few points after range filtering; plotting curve only.")
        # Plot raw/smoothed curve if requested, return zeros
        offset = max(1e-12, -float(np.nanmin(y)) + 1e-12)
        logy = np.log10(y + offset)
        dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        xspan = float(x.max() - x.min()) if x.size > 1 else 1.0
        sigma_samples = max(1.0, (fwhm_frac * xspan) / (2.355 * max(dx, 1e-12)))
        logy_s = gaussian_filter1d(logy, sigma=sigma_samples, mode="nearest") if x.size > 1 else logy
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 4.5))
            else:
                fig = ax.figure
            ax.plot(x, logy, lw=1, label="log10(raw + offset)")
            ax.plot(x, logy_s, lw=2, label=f"log10 smoothed (FWHM≈{fwhm_frac*100:.1f}% span)")
            ax.set_xlabel(f"x ({x_unit})"); ax.set_ylabel("log10(intensity)")
            if qz is not None:
                ax.set_title(f"At qz= {qz}")
            ax.legend(); fig.tight_layout()
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    # 2) log10 with safe offset
    offset = max(1e-12, -float(np.nanmin(y)) + 1e-12)
    logy = np.log10(y + offset)

    # 3) symmetric Gaussian smoothing in x-units (zero phase)
    dx = float(np.median(np.diff(x)))
    xspan = float(x.max() - x.min()) if x.max() > x.min() else 1.0
    sigma_samples = max(1.0, (fwhm_frac * xspan) / (2.355 * max(dx, 1e-12)))
    logy_s = gaussian_filter1d(logy, sigma=sigma_samples, mode="nearest")

    # 4) minima = peaks on inverted smoothed curve
    inv = -logy_s
    rng = float(inv.max() - inv.min())
    prominence = max(1e-12, prom_frac * rng)
    min_width_samples = max(5, int(round((min_width_frac * xspan) / max(dx, 1e-12))))
    min_dist_samples  = max(10, int(round((min_dist_frac  * xspan) / max(dx, 1e-12))))
    mins_idx, props = find_peaks(inv, prominence=prominence,
                                 width=min_width_samples, distance=min_dist_samples)

    # --- minimal change starts here ---
    if mins_idx.size < 2:
        print(f"Warning: Found {mins_idx.size} minima; plotting without minima markers.")
        # Still plot the curves
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 4.5))
            else:
                fig = ax.figure
            ax.plot(x, logy, lw=1, label="log10(raw + offset)")
            ax.plot(x, logy_s, lw=2, label=f"log10 smoothed (FWHM≈{fwhm_frac*100:.1f}% span)")
            ax.set_xlabel(f"x ({x_unit})"); ax.set_ylabel("log10(intensity)")
            if qz is not None:
                ax.set_title(f"At qz= {qz}")
            ax.legend(); fig.tight_layout()

        # Return zeros for any missing extremum.
        if mins_idx.size == 0:
            return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        else:
            # one minimum found -> return it for left_min, zeros for mid and right
            i0 = int(mins_idx[0])
            left_min = (float(x[i0]), float(logy_s[i0]))
            return left_min, (0.0, 0.0), (0.0, 0.0)
    # --- minimal change ends here ---

    # keep top 2 by prominence, then sort by x
    order = np.argsort(props["prominences"])[::-1]
    mins_idx = np.sort(mins_idx[order][:2])

    # 5) mid maximum between the two minima (on smoothed curve)
    i0, i1 = int(mins_idx[0]), int(mins_idx[1])
    jmax = int(i0 + np.argmax(logy_s[i0:i1+1]))

    # 6) package results: (x, log10-intensity) tuples
    left_min  = (float(x[i0]), float(logy_s[i0]))
    right_min = (float(x[i1]), float(logy_s[i1]))
    mid_max   = (float(x[jmax]), float(logy_s[jmax]))

    # 7) plot (only add markers when we have both minima)
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4.5))
        else:
            fig = ax.figure
        ax.plot(x, logy, lw=1, label="log10(raw + offset)")
        ax.plot(x, logy_s, lw=2, label=f"log10 smoothed (FWHM≈{fwhm_frac*100:.1f}% span)")
        ax.plot([left_min[0], right_min[0]], [left_min[1], right_min[1]], "o", ms=9, label="selected minima")
        ax.plot([mid_max[0]], [mid_max[1]], "s", ms=9, label="max between")
        ax.set_xlabel(f"x ({x_unit})"); ax.set_ylabel("log10(intensity)")
        if qz is not None:
            ax.set_title(qz)
        ax.legend(); fig.tight_layout()

    return left_min, mid_max, right_min

