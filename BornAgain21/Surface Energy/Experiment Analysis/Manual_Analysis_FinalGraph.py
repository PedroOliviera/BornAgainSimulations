from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
import numpy as np

exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    

exp_2d_array = []
exp_axes_array = []
exp_filename = 'SiN_10deg.npz'
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
#qz = float(input("Input qz value from 2D data for yoneda: "))
qz = 0.896

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
#while(linecut != 100):
#    linecut = float(input("Input x values of maxima in yoneda band (input 100 when done)"))
#    linecuts.append(linecut)
graphing.plt.close()
#linecuts = [0.09582,0.1675,0.1925,0.255,0.344] - 35deg
#linecuts = [0.09274, 0.1597, 0.18165, 0.24375,0.32676] # Quartz - 10 deg
#linecuts = [0.09, 0.1565, 0.2363] #Sapphire - 10 deg
#linecuts = [0.0842, 0.151, 0.2324, 0.31] #Silicon - 10 deg
linecuts = [0.0947, 0.1618, 0.25]#, 0.337]

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
qz_mid,   qy_mid   = [], []

#mid_max = [0.33258, 0.46655, 0.57404, 0.71112] # Quartz first peak
#mid_max = [0.94946, 1.12704, 1.17377, 1.30307, 1.51025] # Quartz second peak

#mid_max = [0.3357, 0.44163, 0.56157] # Sapphire first peak
#mid_max = [0.89493, 1.0694, 1.22051] # Sapphire second peak

#mid_max = [0.33869, 0.44501, 0.537, 0.66544] Si first peak

mid_max = [0.317, 0.3871, 0.49566]#, 0.52629] #SiN first peak

for qz, qy in zip(linecuts, mid_max):
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
        x_qy, I, x_range=(0.1, 2), x_unit="1/nm", qz=qz, plot= False
    )
    left_min, mid_max, right_min = [0.0], [0.0], [0.0]
    graphing.plt.show(block=False)

    # function returns (qy, log10I). we want the qy positions
    print("here")
    #mid_max[0] = float(input("Input Mid Max: "))
    qy_positions = [qy]

    if sum(qy == 0.0 for qy in qy_positions):
        continue

    # original overlay points
    xs.extend([qz] * len(qy_positions))
    ys.extend(qy_positions)

    # NEW: store each track separately (left, mid, right)
    
    qz_mid.append(qz);    qy_mid.append(qy_positions[0])
    #graphing.plt.close()

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

_fit_and_plot(exp_ax, qz_mid,   qy_mid,   color="tab:orange", label="Mid max")

# (optional) show legend for the fitted lines
# exp_ax.legend(loc="best")
graphing.plt.savefig("SiN_0p1.png", dpi=300)
graphing.plt.savefig("SiN_0p1.pdf", dpi=300)
graphing.plt.show()
