import GISAXS_setup_v24 as g
import GraphingAnalysis as graphing
from bornagain.numpyutil import Arrayf64Converter as dac
from bornagain import ba_plot as bp

sample = g.get_sampleTest()

sim = g.get_simulation_2D(sample, 'feb', 0.15)

result = sim.simulate()
simulationData = dac.asNpArray(result.dataArray())
print(bp.get_axes_limits(result))
graphing.plot2D(simulationData=simulationData, simData_axes=[-0.06409331913432209, 0.06409331913432209, -0.06409331913432209, 0.06409331913432209])