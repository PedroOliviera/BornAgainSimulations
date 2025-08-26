import numpy as np
import GraphingAnalysis as graphing
import GISAXS_V6_ROI as g
from bornagain.numpyutil import Arrayf64Converter as dac
from bornagain import ba_plot as bp, deg, nm



if __name__ == '__main__':
    data_array = np.load("test_1_hemiellipsoid_distribution_15deg_3D.npy")
    graphing.yonedaPlot(data_npArrays=[data_array], data_axes=[0, 0.5, 0, 0.5], vert_slice_q=0.1)
    graphing.plot2D(simulationData=data_array, realDat_axes=[0, 0.5, 0, 0.5], zlim=[0.001,200])
    