import GISAXS_setup_v21 as g
import Graphing_Analysis as graphing

#sim2D, simAxes = g.loadSim("fitting_Run1_miniuit2.npz")
#sim2D, simAxes = g.loadSim("fitting_Run3_Genetic.npz")

#sim2D, simAxes, params = g.loadSim("fitting_Run4_Minuit2_test_0p062LC4_1p5LC3_.npz", return_date=False, return_params=True)
#print(params)
realData_npArray, realDat_axes_Feb = g.loadSim(r"C:\Users\Pedro\Data Transfer\Sample_35_3secIntegration\sample35_13deg.npz")

#step2
linecut3 = 1.5
linecut4 = 0.062
#step3
linecut3 = 1.5
linecut5 = 0.212
linecut2 = 1.077

#realDat_axes_Feb = simAxes

for i in range(5):
    sim2D, simAxes, params = g.loadSim("fitting_Run5_Genetic_0p062LC4_1p5LC3_" + str(i) + ".npz", return_date=False, return_params=True)
    print(params)
    print(type(params))
    graphing.plot2D(realData=realData_npArray, 
                simulationData=sim2D, 
                realDat_axes=realDat_axes_Feb, 
                simData_axes=simAxes, 
                zlim=[22,50000])
    graphing.linecutsItoV(simulation_data=sim2D, 
                      experimental_data=realData_npArray, 
                      #L2_qy=linecut2,
                      L3_qz=linecut3, 
                      L4_qy=linecut4,
                      #L5_qz=linecut5, 
                      axes_exp=realDat_axes_Feb, 
                      axes_sim=simAxes)
#graphing.plot2D(realData=realData_npArray, 
#                realDat_axes=realDat_axes_Feb, 
#                graphed_axes=realDat_axes_Feb,
#                zlim=[22,50000])

#g.graph_experiment_detectorSpace("sample35_13deg.npz",'feb',0.13)

graphing.plt.show()