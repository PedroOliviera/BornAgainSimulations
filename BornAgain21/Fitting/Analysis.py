import GISAXS_setup_v21 as g
import Graphing_Analysis as graphing

sim2D, simAxes = g.loadSim("test.npz")

realData_npArray, realDat_axes_Feb = g.loadSim("sample35_13deg.npz")
linecut3 = 1.0
linecut4 = 0.0

graphing.plot2D(realData=realData_npArray, 
                simulationData=sim2D, 
                realDat_axes=realDat_axes_Feb, 
                simData_axes=simAxes, 
                zlim=[22,50000])
graphing.linecutsItoV(simulation_data=sim2D, 
                      experimental_data=realData_npArray, 
                      L3_qz=linecut3, 
                      L4_qy=linecut4, 
                      axes_exp=realDat_axes_Feb, 
                      axes_sim=simAxes)
graphing.plot2D(realData=realData_npArray, 
                realDat_axes=realDat_axes_Feb, 
                graphed_axes=realDat_axes_Feb,
                zlim=[22,50000])
graphing.plt.show()