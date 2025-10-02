from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
import numpy as np

exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    
exp_filenames = ['Mica_4824_2000RPM_3mgPml_0p35deg.npz','Quartz_4824_2000RPM_3mgPml_0p15deg.npz','Sapphire_4824_2000RPM_3mgPml_0p1deg.npz','SiN_0p2deg.npz']
exp_2d_array = []
exp_axes_array = []
labels = ['Mica', 'Quartz', ' Sapphire', 'SiN']
#exp_filenames = ['Sapphire_0p2deg.npz']
#labels = ['Sapphire']
linecuts1 = [0.44, 0.3, 0.31, 0.275, 0.31] #0.275 - 0.33 - 0.822 SiN 0.44 Mica
linecuts2 = [0.093, 0.091, 0.089, 0.095, 0.0829]

for linecut1, linecut2, fname in zip(linecuts1, linecuts2, exp_filenames):
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    exp_2d_array.append(exp_2d)
    exp_axes_array.append(exp_axes)

'''
graphing.hor_slice_comparison(hor_slice_q_array=linecuts1, 
                              data_npArrays=exp_2d_array, 
                              data_axes_array=exp_axes_array, 
                              xmin=0.06, xmax=0.125, labels=labels)
graphing.plt.xlim(right=0.6)
graphing.plt.ylim(bottom=0.000343)
'''
graphing.vert_slice_comparison(vert_slice_q_array=linecuts2, 
                               data_npArrays=exp_2d_array,
                               data_axes_array=exp_axes_array, 
                               xmin=0.1, xmax=0.6, labels=labels)

# draw the 2D map and keep the Experiment axes for overlay
for fname, label in zip(exp_filenames, labels):
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    exp_2d_array.append(exp_2d)
    exp_axes_array.append(exp_axes)
    exp_ax, _ = graphing.plot2D(
        realData=exp_2d,
        realDat_axes=exp_axes,
        zlim=[22, exp_2d.max()],
        title=label
    )

    # qz values (horizontal slices) to analyze
    linecuts = np.linspace(0.09, 0.40, 10)   # or: np.array([0.091, 0.1587, 0.1806])

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
            x_qy, I, x_range=(0.507, 1.6), x_unit="1/nm", qz=qz
        )

        # function returns (qy, log10I). we want the qy positions
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
