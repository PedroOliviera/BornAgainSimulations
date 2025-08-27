import GISAXS_setup_v24 as g
import GraphingAnalysis as graphing
from bornagain.numpyutil import Arrayf64Converter as dac
from bornagain import ba_plot as bp
import bornagain as ba
from bornagain import nm, deg

alpha_i = 0.15

wavelength = 0.125916*nm   
sample = g.get_sampleTest()
sim = g.get_simulation_2D(sample, 'feb', alpha_i)

result = sim.simulate()
simulationData = dac.asNpArray(result.dataArray())
res2 = g.transform_axis(result, alpha_i)

bp.plot_datafield(res2)
bp.plt.show()
graphing.plot2D(simulationData=simulationData, simData_axes=bp.get_axes_limits(res2.plottableField()))