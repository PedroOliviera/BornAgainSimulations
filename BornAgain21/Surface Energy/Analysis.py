from GISAXS_Analysis import GISAXS_setup_v21 as g

save_filename = "monolayer_test.npz"
save_sim_directory = r'C:\Users\Pedro\BornAgainSimulations\data\sim-npz'

sim2D, simAxes, params = g.load_npz_data(save_filename, save_sim_directory, return_date=False, return_params=True)


exp_data_directory = r'C:\Users\Pedro\BornAgainSimulations\data\exp-npz'
    
exp_filename = '4824_3gPL_2000RPM_0p1Deg.npz'
exp2D, exp_axes = g.load_npz_data(exp_filename, exp_data_directory)

g.lineScan(exp2D, 0.01, 1, axesLimits=exp_axes, pixel_inc=10)
g.plt.show()