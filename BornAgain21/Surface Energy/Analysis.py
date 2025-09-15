from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing

def usual():
    save_filename = "monolayer_test.npz"
    save_sim_directory = r'C:\Users\Pedro\BornAgainSimulations\data\sim-npz'

    sim2D, simAxes, params = g.load_npz_data(save_filename, save_sim_directory, return_date=False, return_params=True)


    exp_data_directory = r'C:\Users\Pedro\BornAgainSimulations\data\exp-npz'
        
    exp_filename = '4824_3gPL_2000RPM_0p1Deg.npz'
    exp2D, exp_axes = g.load_npz_data(exp_filename, exp_data_directory)

    linecut1 = 0.2
    graphing.linecutsItoV(simulation_data=exp2D, 
                    L1_qz=linecut1, 
                    #L2_qy=linecut2,
                    #L5_qz=linecut5, 
                    axes_sim=exp_axes)
    graphing.plt.show()

def tip_conversion():
    exp_filename = '4824_3gPL_2000RPM_0p1Deg.tif'
    exp_data_directory = r'C:\Users\Pedro\BornAgainSimulations\data\tif'
    g.tifToNpzConversion(exp_filename,exp_data_directory, 'dec', 0.1)
tip_conversion()