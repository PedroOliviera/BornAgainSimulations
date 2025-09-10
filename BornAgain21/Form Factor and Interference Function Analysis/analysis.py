import numpy as np
import Graphing_Analysis as graphing
from bornagain import ba_plot as bp, deg, nm



if __name__ == '__main__':
    data_array = np.load("tests_10deg_line.npy")
    legendLabels = ['CustomSinusoidal']
    graphing.plot2D(simulationData=data_array, simData_axes=[0, 0.5, 0, 0.5])
    graphing.plt.show()
    #data_array_custom = np.load("tests_13deg_3D_custom.npy")
    #data_array_CosineRippleGauss = np.load("tests_13deg_3D_CosineRippleGauss.npy")
    #legendLabels = ['CustomSinusoidal', "CosineRippleGauss"]
    #graphing.yonedaPlot(data_npArrays=[data_array_custom, data_array_CosineRippleGauss], data_axes=[0, 0.5, 0, 0.5], vert_slice_q=0.1, xmin=0.05, xmax=0.2, labels=legendLabels)
    #graphing.plot2D_simulationComparison(simulationData=data_array_custom, simData_axes=[0, 0.5, 0, 0.5], realData=data_array_CosineRippleGauss, realDat_axes=[0, 0.5, 0, 0.5])
    