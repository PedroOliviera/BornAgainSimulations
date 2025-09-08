from bornagain import deg, nm, R3
import os
from matplotlib import pyplot as plt
from bornagain import ba_plot as bp
import bornagain as ba
import numpy as np
import time
from pathlib import Path
import Graphing_Analysis as graphing
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
import matplotlib.ticker as mticker
import GISAXS_setup_v21 as g

directory1 = r'C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\GISAXS Analysis\Data\GISAS\35'
folder = Path(directory1)
tifs = sorted(folder.glob("*.tif"), key=lambda p: float(p.stem.split("_")[-1]))
filenames = [p.name for p in tifs]
print(filenames)

realData_npArrays = [np.array([]) for i in filenames]
fileLabel = [i for i in range(len(filenames))]

for i, filen in enumerate(filenames): 
    realData_npArrays[i] = g.center_img(g.real_data(filen, directory1))

#For Feb Data
realDat_axes_Feb = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517]

axes = [-1, 1, 0, 2]
#for readData_npArray in realData_npArrays:
#    graphing.plot2D(realDat_axes=realDat_axes_Feb, realData=readData_npArray)
#plt.show()

vert_slice_q = 0.1
plt.figure(figsize=(7,5))

x_max_graph = []
arrays = realData_npArrays[0:8]
n_datasets = len(arrays)+1
cmap = cm.get_cmap("jet", n_datasets)  # evenly spaced colors from jet colormap

for i, data in enumerate(arrays):
    # Take vertical slice at Qz = vert_slice_q
    x1, y1 = g.plot_slices(data, axesLimits=realDat_axes_Feb, vert_slice=vert_slice_q)
    ind_x1 = np.argmax(y1)         # position of maximum intensity
    hor_slice_q = x1[ind_x1]       # Qy value of that maximum
    
    step = 0.01
    #hor_slice_q = 0.4
    # Now take horizontal slice through that maximum
    x2, y2 = g.plot_slices(data, axesLimits=realDat_axes_Feb, horiz_slice=hor_slice_q)
    x2, y2 = g.integrate_plt_slices(start = hor_slice_q - step, stop= hor_slice_q + step, data=data, axLim=realDat_axes_Feb, labelname=i, num=20, horiz_slice=True)
    x2_norm, y2_norm = graphing.normalize_by_first_peak(x2, y2, x_min = 0.085, x_max=0.137)
    
    x_max_graph.append(abs(x2_norm[np.argmax(y2_norm)])) #x value where y_max (peak) occurs

    y_s = savgol_filter(y2_norm, window_length=20, polyorder=3, mode="interp")

    # Get color from colormap
    color = cmap(i)

    # Plot with label and custom color
    plt.plot(x2_norm, y_s, label = rf"$\alpha = {(i/100+0.1):.2f}^\circ$", color=color)

# Improve legend and axis formatting
plt.legend(title="Incidence Angle", fontsize=9, ncol=2)  # 2-column legend if many datasets
plt.ylim(bottom=3e-4)
plt.xlim(left=0.055)
plt.ylabel("Normalized Intensity", fontsize=11)
plt.xlabel(r"$Q_{y}\;(1/{\rm nm})$", fontsize=11)
plt.title(rf"Horizontal Slices Along $Q_{{z}}$", fontsize=12)
plt.yscale("log")
plt.xscale("log")
plt.grid(which="both", ls="--", lw=0.5, alpha=0.6)

plt.tight_layout()

plt.figure()

num = len(x_max_graph)
angles = [i/100 + 0.1 for i in range(num)]
plt.scatter(angles, x_max_graph)
plt.show()