import GISAXS_setup_v24 as g
import GraphingAnalysis as graphing
from bornagain.numpyutil import Arrayf64Converter as dac
from bornagain import ba_plot as bp
import bornagain as ba
from bornagain import nm, deg
import GISAXS_V6_ROI as g2
import matplotlib.pyplot as plt
def example1():
    alpha_i = 0.15

    wavelength = 0.125916*nm   
    sample = g.get_sampleTest()
    sim = g.get_simulation_2D(sample, 'feb', alpha_i)

    result = sim.simulate()
    simulationData = dac.asNpArray(result.dataArray())
    print(type(simulationData))
    #res2 = g.transform_axis(result, alpha_i)

    #bp.plot_datafield(res2)
    #bp.plot_simres(result)
    #bp.plt.show()
    #graphing.plot2D(simulationData=simulationData, simData_axes=bp.get_axes_limits(res2.plottableField()))

def example2():
    alpha_i = 0.15

    dir = r'C:\Data'
    filename = '35_2000RPM_40mgPml_polymer_0p18.tif'
    result = g.real_data(dir, filename)
    bp.plot_simres(result)
    bp.plt.show()
    # Full detector maps (change to an ROI if you prefer)
    # Example ROI: roi = (1700, 2400, 1800, 2500)

    flip_v = True 
    qx, qy, qz, alpha_f, phi = g.uv_to_q_maps(
        alpha_i,flip_v=flip_v
    )
    plt.figure()
    plt.imshow(qy, qz)
    plt.show()

example1()
#example2()
#realdata = dac.asNpArray(result.dataArray())
#res2 = g.transform_axis(result, alpha_i)
#bp.plot_datafield(res2)
#bp.plt.show()