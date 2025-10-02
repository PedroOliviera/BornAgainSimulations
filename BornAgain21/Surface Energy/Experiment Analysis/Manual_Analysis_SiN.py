from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
import numpy as np

exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    

exp_2d_array = []
exp_axes_array = []
exp_filename = 'SiN_0p2deg.npz'
labels = ['SiN']
label = labels[0]

# First load in file data and plot 2D
exp_2d, exp_axes = g.load_npz_data(exp_filename, exp_data_directory)
exp_ax, _ = graphing.plot2D(
    realData=exp_2d,
    realDat_axes=exp_axes,
    zlim=[22, exp_2d.max()],
    title=label
)
graphing.plt.show(block=False)

#Find qz value of Yoneda band in plot - plot horizontal lineprofile
qz = float(input("Input qz value from 2D data for yoneda: "))
#qz = 0.896

graphing.plt.close()
graphing.hor_slice_comparison(hor_slice_q_array=[qz], 
                              data_npArrays=[exp_2d], 
                              data_axes_array=[exp_axes], 
                              xmin=0.06, xmax=0.125, labels=labels)
graphing.plt.xlim(right=0.6)
graphing.plt.ylim(bottom=0.000343)
graphing.plt.show(block=False)

#Find peaks in yoneda horizontal line profile and use that for vertical slices
linecuts = []
linecut = ''
while(linecut != 100):
    linecut = float(input("Input x values of maxima in yoneda band (input 100 when done)"))
    linecuts.append(linecut)
graphing.plt.close()
linecuts = [0.0942,0.165,0.245]

exp_2d_array = [exp_2d for linecut in linecuts]
exp_axes_array = [exp_axes for linecut in linecuts]
labels = [label for linecut in linecuts]
graphing.vert_slice_comparison(vert_slice_q_array=linecuts, 
                               data_npArrays=exp_2d_array,
                               data_axes_array=exp_axes_array, 
                               xmin=0.1, xmax=0.6, labels=labels)
graphing.plt.show()

exp_ax, _ = graphing.plot2D(
    realData=exp_2d,
    realDat_axes=exp_axes,
    zlim=[22, exp_2d.max()],
    title=label
)


# arrays of points to overlay on the 2D map
xs = []   # qz for plotting on x-axis
ys = []   # qy extrema for plotting on y-axis

# NEW: keep separate series for the three tracks (for line fitting)
qz_left,  qy_left  = [], []
qz_mid,   qy_mid   = [], []
qz_right, qy_right = [], []

for qz in linecuts:
    # integrate a thin band around this qz to get a 1D profile vs qy
    x_qy, I = g.integrate_plt_slices(
        start=qz - 1e-4,
        stop=qz + 1e-4,
        data=exp_2d,
        axLim=exp_axes,
        labelname="Experiment",
        num=1,
        vert_slice=True
    )

    # find minima & mid-maximum along qy (don’t open a new figure)
    left_min, mid_max, right_min = graphing.find_two_minima_and_midmax(
        x_qy, I, x_range=(0.27, 2), x_unit="1/nm", qz=qz
    )
    left_min, mid_max, right_min = [0.0], [0.0], [0.0]
    graphing.plt.show(block=False)

    # function returns (qy, log10I). we want the qy positions
    left_min[0] = float(input("Input Left Min: "))
    mid_max[0] = float(input("Input Mid Max: "))
    right_min[0] = float(input("Input Right Min: "))
    qy_positions = [left_min[0], mid_max[0], right_min[0]]

    if sum(qy == 0.0 for qy in qy_positions):
        continue

    # original overlay points
    xs.extend([qz] * len(qy_positions))
    ys.extend(qy_positions)

    # NEW: store each track separately (left, mid, right)
    
    qz_left.append(qz);   qy_left.append(qy_positions[0])
    qz_mid.append(qz);    qy_mid.append(qy_positions[1])
    qz_right.append(qz);  qy_right.append(qy_positions[2])
    graphing.plt.close()

# overlay markers on the experiment axes (unchanged)
exp_ax.scatter(xs, ys, marker="x", s=120, c="w", linewidths=2, zorder=5)

# === NEW: fit lines and plot them, then report slope & angle ===
def _fit_and_plot(ax, x_pts, y_pts, color, label, zorder=4):
    import numpy as np
    X = np.asarray(x_pts, float); Y = np.asarray(y_pts, float)
    if X.size < 2:
        print(f"Warning: not enough points for '{label}' to fit a line.")
        return
    m, b = np.polyfit(X, Y, 1)                       # slope & intercept
    angle_deg = float(np.degrees(np.arctan(m)))      # angle vs horizontal
    xfit = np.array(exp_ax.get_xlim(), dtype=float)  # span current qz-axis
    yfit = m * xfit + b
    ax.plot(xfit, yfit, color=color, linewidth=2, zorder=zorder, label=f"{label} fit")
    print(f"{label}: slope = {m:.6g}, angle = {angle_deg:.2f}°")

_fit_and_plot(exp_ax, qz_left,  qy_left,  color="tab:cyan",   label="Left minima")
_fit_and_plot(exp_ax, qz_mid,   qy_mid,   color="tab:orange", label="Mid max")
_fit_and_plot(exp_ax, qz_right, qy_right, color="tab:green",  label="Right minima")

# (optional) show legend for the fitted lines
# exp_ax.legend(loc="best")

graphing.plt.show()
