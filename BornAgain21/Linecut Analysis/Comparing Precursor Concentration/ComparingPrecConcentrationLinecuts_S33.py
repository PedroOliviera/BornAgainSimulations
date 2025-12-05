from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.cm as cm
import matplotlib.colors as mcolors

exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz\feb'
    
exp_filenames = [f"4_{i}deg.npz" for i in range(10, 21)]  # stop is exclusive

exp_2d_array = []
exp_axes_array = []
labels = [f"{i} deg" for i in range(10, 21)]

'''
for fname in exp_filenames:
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    data_npArrays.append(exp_2d)
'''

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
    print('peak location:')
    print(x[peak_idx])
    return x, y / peak_height

def hor_slice_comparison(hor_slice_q_array, data_npArrays, data_axes_array, data2_npArray=None, data_axes2=None, xmin = 0.0, xmax = 0.0, labels = None):
    """Inputs:
    vert_slice_q: will take max of this vert slice value and use for horizontal slice value
    data_npArrays: array of dataset to be compared
    data_axes: axes of data (g2.get_axes_limits(result, ba.Coords_QSPACE) for simulation) and realData_axes_month for experimental data
    data2_npArrays: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    data2_axes2: designed to add one other dataset that has a different axis e.g. adding one experiment to varying sim parameter
    """
    plt.figure(figsize=(5,5))

    n_datasets = len(data_npArrays)
    cmap = cm.get_cmap("rainbow", n_datasets)  # evenly spaced colors from jet colormap
    #cmap = ['red', 'green', 'purple', 'blue', 'orange']
    for i, (data, data_axes, hor_slice_q) in enumerate(zip(data_npArrays, data_axes_array, hor_slice_q_array)):
        
        step = 0.01
        # Now take horizontal slice through that maximum
        x2, y2 = g.plot_slices(data, axesLimits=data_axes, horiz_slice=hor_slice_q)
        x2, y2 = g.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data, axLim=data_axes, labelname=i, num=100, horiz_slice=True)
       
        x2_norm, y2_norm = normalize_by_first_peak(x2, y2, x_min = xmin, x_max=xmax)
        
        y_2_norm_shifted = y2_norm * 10**(i)
        
        #y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

        # Get color from colormap
        color = cmap(i) #[i]

        # Plot with label and custom color
        plt.plot(x2_norm, y_2_norm_shifted, label = labels[i], color=color, lw=2)



    # Improve legend and axis formatting
    plt.legend(title="Substrate", fontsize=10, ncol=2)  # 2-column legend if many datasets
    plt.ylim(bottom=0.00026)
    plt.xlim(left=0.055, right = 2.5)
    plt.ylabel("Normalized Intensity", fontsize=11)
    plt.xlabel(r"$Q_{y}\;(1/{\rm nm})$", fontsize=11)
    #plt.title(rf"Horizontal Slices Along $Q_{{y}}$", fontsize=12)
    plt.yscale("log")
    #plt.xscale("log")
    plt.grid(which="both", ls="--", lw=0.5, alpha=0.6)
    plt.tight_layout()

#linecuts1 = [0.208 + i * 0.00871 for i in range(11)] #at 20 deg 0.44, 0.3, 0.31, 0.275, 0.31 at 10 deg - 0.282, 0.4, 0.4, 0.35, 0.45
linecuts1 = [0.208, 0.222, 0.2305, 0.2415, 0.2524, 0.258, 0.26, 0.268, 0.276, 0.287, 0.3, 0.31]
linecuts2 = [0.128 for i in range(10,21)]

for linecut1, linecut2, label, fname in zip(linecuts1, linecuts2, labels, exp_filenames):
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    exp_2d_array.append(exp_2d)
    exp_axes_array.append(exp_axes)
    graphing.plot2D(exp_2d, realDat_axes=exp_axes, L1_qz=linecut1, L2_qy=linecut2, zlim=[22,exp_2d.max()])
    graphing.plt.title(label)
    graphing.linecutsItoV(experimental_data=exp_2d, L1_qz=linecut1, L2_qy=linecut2, axes_exp=exp_axes, save=True, savefname = label)
    graphing.plt.title(label)


hor_slice_comparison(hor_slice_q_array=linecuts1, 
                              data_npArrays=exp_2d_array, 
                              data_axes_array=exp_axes_array, 
                              xmin=0.06, xmax=0.125, labels=labels)
#plt.savefig("Substrates_Horizontal_lineucts.pdf", dpi = 300)
#plt.savefig("Substrates_Horizontal_lineucts.png", dpi = 300)

graphing.vert_slice_comparison(vert_slice_q_array=linecuts2, 
                               data_npArrays=exp_2d_array,
                               data_axes_array=exp_axes_array, 
                               xmin=0.1, xmax=0.6, labels=labels)
graphing.plt.show()