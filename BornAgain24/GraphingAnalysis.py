import GISAXS_V6_ROI as g2
from matplotlib import pyplot as plt
from bornagain import ba_plot as bp
import bornagain as ba
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import numpy as np
from matplotlib.colors import LogNorm

# Roman numerals for subplot titles
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V"]

def normalize_by_first_peak(
    x,
    y,
    x_min=0.14,
    x_max=0.16,
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

    # iterate over window indices to find first local maximum
    peak_idx = None
    for i in in_window:
        # avoid edges where i==0 or i==len(y)-1
        if 0 < i < len(y) - 1 and y[i - 1] < y[i] >= y[i + 1]:
            peak_idx = i
            break

    # fallback: pick highest point in the window
    if peak_idx is None:
        peak_idx = in_window[np.nanargmax(y[in_window])]

    peak_height = y[peak_idx]
    if peak_height == 0:
        raise ValueError("Peak height is zero; cannot normalize.")

    return x, y / peak_height

def plot_qy_linecut(ax, qy, simulation_data, experimental_data, axes_sim, axes_exp, labels, linecut_index):
    ax.set_title(f'Linecut {ROMAN_NUMERALS[linecut_index]}')

    horizontal_slice_1 = qy + 0.05
    horizontal_slice_2 = qy - 0.05

    if simulation_data is not None:
        x, y = g2.integrate_plt_slices(
            start=horizontal_slice_2,
            stop=horizontal_slice_1,
            data=simulation_data,
            axLim=axes_sim,
            labelname="Simulation",
            num=10,
            horiz_slice=True
        )
        ax.plot(x, y, label=labels[0])

    if experimental_data is not None:
        x, y = g2.integrate_plt_slices(
            start=horizontal_slice_2,
            stop=horizontal_slice_1,
            data=experimental_data,
            axLim=axes_exp,
            labelname="Experiment",
            num=10,
            horiz_slice=True
        )
        ax.plot(x, y, label=labels[1])

    ax.set_ylabel("Intensity")
    ax.set_xlabel(r'$Q_{y} \;(1/{\rm nm})$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(bottom=10)
    ax.legend()

def plot_qz_linecut(ax, qz, simulation_data, experimental_data, axes_sim, axes_exp, labels, linecut_index):
    ax.set_title(f'Linecut {ROMAN_NUMERALS[linecut_index]}')

    vertical_slice_1 = qz + 0.05
    vertical_slice_2 = qz - 0.05

    x_data_all, y_data_all = [], []

    if simulation_data is not None:
        x, y = g2.integrate_plt_slices(
            start=vertical_slice_2,
            stop=vertical_slice_1,
            data=simulation_data,
            axLim=axes_sim,
            labelname="Simulation",
            num=10,
            vert_slice=True
        )
        ax.plot(x, y, label=labels[0])
        x_data_all.extend(x)
        y_data_all.extend(y)

    if experimental_data is not None:
        x, y = g2.integrate_plt_slices(
            start=vertical_slice_2,
            stop=vertical_slice_1,
            data=experimental_data,
            axLim=axes_exp,
            labelname="Experiment",
            num=10,
            vert_slice=True
        )
        ax.plot(x, y, label=labels[1])
        x_data_all.extend(x)
        y_data_all.extend(y)

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
    labels=("Simulation", "Experiment"),
    title=""
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
    n = len(active_linecuts)

    if n == 0:
        print("No linecuts defined.")
        return

    axes_sim = g2.get_axes_limits(result, ba.Coords_QSPACE) if simulation_data is not None else None

    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axs = [axs]

    for i, (ax, (kind, val)) in enumerate(zip(axs, active_linecuts)):
        if kind == "qy":
            plot_qy_linecut(ax, val, simulation_data, experimental_data, axes_sim, axes_exp, labels, i)
        elif kind == "qz":
            plot_qz_linecut(ax, val, simulation_data, experimental_data, axes_sim, axes_exp, labels, i)

    fig.suptitle("Simulation: " + title, fontsize=16)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()

def _is_ba_datafield(obj) -> bool:
    # BornAgain v23 Datafield exposes methods like dataArray(), xCenters(), yCenters()
    return hasattr(obj, "dataArray") and callable(getattr(obj, "dataArray", None))

def plot2D(
    realData=None,
    simulationData=None,
    realDat_axes=None,        # [qy_min, qy_max, qz_min, qz_max] for NumPy arrays
    simData_axes=None,        # optional override of graphed_axes for simulation panel
    graphed_axes=None,        # [qy_min, qy_max, qz_min, qz_max] to apply to both panels
    L1_qz=None, L2_qy=None, L3_qz=None, L4_qy=None, L5_qz=None,
    title="",
    zlim=(22, 5e5),           # (vmin, vmax)
    cmap='gist_ncar'
):
    """
    Works with BornAgain v23. Accepts either a BornAgain Datafield or a NumPy 2D array.
    - For Datafield: uses bp.plot_simulation_result(...)
    - For NumPy: uses plt.imshow(..., extent=axes, origin='lower', LogNorm)
    """
    if simulationData is None and realData is None:
        print("No data provided."); return

    panels = []
    if simulationData is not None:
        panels.append(("Simulation", simulationData, simData_axes))
    if realData is not None:
        panels.append(("Experiment", realData, realDat_axes))

    n = len(panels)
    plt.figure(figsize=(7.5 * n, 6))

    # default graphed_axes preference: use sim axes if available, else real, else None
    if graphed_axes is None:
        if simData_axes is not None:
            graphed_axes = simData_axes
        elif realDat_axes is not None:
            graphed_axes = realDat_axes

    # Linecut definitions with labels
    vert_lines = [(L2_qy, 'II'), (L4_qy, 'IV')]
    horiz_lines = [(L1_qz, 'I'), (L3_qz, 'III'), (L5_qz, 'V')]

    for i, (label, data, axes_limits) in enumerate(panels, start=1):
        ax = plt.subplot(1, n, i)

        if _is_ba_datafield(data):
            # BornAgain Datafield path (v23)
            im = bp.plot_simulation_result(
                data,
                cmap=cmap
                # Other bp kwargs can still be passed if needed
            )
            # control color scale after plotting
            im.set_clim(zlim[0], zlim[1])
            ax = im.axes

            # If caller provided axes limits, apply on top (qy on x, qz on y)
            if graphed_axes is not None:
                ax.set_xlim(graphed_axes[0], graphed_axes[1])
                ax.set_ylim(graphed_axes[2], graphed_axes[3])

            # Set standard labels (BornAgain already labels sensibly, but we override)
            ax.set_xlabel(r'$Q_{y} \;(1/{\rm nm})$', fontsize=14)
            ax.set_ylabel(r'$Q_{z} \;(1/{\rm nm})$', fontsize=14)

        else:
            # NumPy array path (experimental or precomputed intensities)
            if axes_limits is None and graphed_axes is None:
                raise ValueError(
                    "For NumPy arrays you must supply axis limits "
                    "[qy_min,qy_max,qz_min,qz_max] via realDat_axes/simData_axes or graphed_axes."
                )
            extent = (graphed_axes or axes_limits)
            if extent is None:
                raise ValueError("Could not determine axes extent for NumPy array.")
            # data assumed to be indexed as [qz, qy] or [rows, cols]; origin='lower'
            im = ax.imshow(
                np.asarray(data),
                origin='lower',
                extent=[extent[0], extent[1], extent[2], extent[3]],
                aspect='auto',
                norm=LogNorm(vmin=zlim[0], vmax=zlim[1]),
                cmap=cmap
            )
            plt.colorbar(im, ax=ax)

            ax.set_xlim((graphed_axes or axes_limits)[0], (graphed_axes or axes_limits)[1])
            ax.set_ylim((graphed_axes or axes_limits)[2], (graphed_axes or axes_limits)[3])
            ax.set_xlabel(r'$Q_{y} \;(1/{\rm nm})$', fontsize=14)
            ax.set_ylabel(r'$Q_{z} \;(1/{\rm nm})$', fontsize=14)

        ax.set_title(label if label == "Experiment" else f"{label}: {title}", fontsize=14)

        # Draw vertical (qy) linecuts
        for qy, roman in vert_lines:
            if qy is not None:
                ax.axvline(x=qy, color='red', linewidth=1)
                ax.text(qy, ax.get_ylim()[0], f'{roman}', color='red',
                        fontsize=12, ha='center', va='bottom', rotation=90)

        # Draw horizontal (qz) linecuts (±0.005 nm⁻¹ bracketing like your original)
        for qz, roman in horiz_lines:
            if qz is not None:
                ax.axhline(y=qz + 0.005, color='blue', linewidth=1)
                ax.axhline(y=qz - 0.005, color='red', linewidth=1)
                ax.text(ax.get_xlim()[0], qz, f'{roman}', color='black',
                        fontsize=12, ha='left', va='center')

    plt.tight_layout()
    plt.show()

def yonedaPlot(vert_slice_q, data_npArrays, data_axes, data2_npArray=None, data_axes2=None):
    """Inputs:
    vert_slice_q: will take max of this vert slice value and use for horizontal slice value
    data_npArrays: array of dataset to be compared
    data_axes: axes of data (g2.get_axes_limits(result, ba.Coords_QSPACE) for simulation) and realData_axes_month for experimental data
    data2_npArrays: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    data2_axes2: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    """
    vert_slice_q = 0.1
    plt.figure(figsize=(7,5))

    n_datasets = len(data_npArrays)
    cmap = cm.get_cmap("jet", n_datasets)  # evenly spaced colors from jet colormap

    for i, data in enumerate(data_npArrays):
        # Take vertical slice at Qz = vert_slice_q
        x1, y1 = g2.plot_slices(data, axesLimits=data_axes, vert_slice=vert_slice_q)
        ind_x1 = np.argmax(y1)         # position of maximum intensity
        hor_slice_q = x1[ind_x1]       # Qy value of that maximum
        print(hor_slice_q)
        
        step = 0.01
        # Now take horizontal slice through that maximum
        x2, y2 = g2.plot_slices(data, axesLimits=data_axes, horiz_slice=hor_slice_q)
        x2, y2 = g2.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data, axLim=data_axes, labelname=i, num=20, horiz_slice=True)
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = 0.085, x_max=0.137)
        #x2_norm, y2_norm = x2, y2
        
        y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Get color from colormap
        color = cmap(i-1)

        # Plot with label and custom color
        plt.plot(x2_norm, y_s, label = rf"$\alpha = {i:.2f}^\circ$", color=color)

    if data2_npArray is not None:
        x1, y1 = g2.plot_slices(data2_npArray, axesLimits=data_axes2, vert_slice=vert_slice_q)
        ind_x1 = np.argmax(y1)         # position of maximum intensity
        hor_slice_q = x1[ind_x1]       # Qy value of that maximum
        
        step = 0.01
        #hor_slice_q = 0.4
        # Now take horizontal slice through that maximum
        x2, y2 = g2.plot_slices(data2_npArray, axesLimits=data_axes2, horiz_slice=hor_slice_q)
        x2, y2 = g2.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data2_npArray, axLim=data_axes2, labelname=i, num=20, horiz_slice=True)
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = 0.085, x_max=0.137)
        
        y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Plot with label and custom color
        plt.plot(x2_norm, y_s, label = "Experiment", color='black')


    # Improve legend and axis formatting
    plt.legend(title="Incidence Angle", fontsize=9, ncol=2)  # 2-column legend if many datasets
    plt.ylim(bottom=2e-6)
    plt.xlim(left=0.055)
    plt.ylabel("Normalized Intensity", fontsize=11)
    plt.xlabel(r"$Q_{y}\;(1/{\rm nm})$", fontsize=11)
    plt.title(rf"Horizontal Slices Along $Q_{{z}}$", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both", ls="--", lw=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()
