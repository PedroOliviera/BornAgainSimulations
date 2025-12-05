#!/usr/bin/env python3
"""
My own attempt at fitting code
"""

import bornagain as ba, numpy as np, os, matplotlib.pyplot as plt
from bornagain.numpyutil import Arrayf64Converter as dac
from bornagain import angstrom, ba_fitmonitor, R3, nm
from bornagain import ba_plot as bp, deg, nm
from matplotlib import gridspec
from bornagain import sample_tools
from matplotlib import cm
import itertools


def truncated_radius(h,d):
    """
    INPUTS
    h -> height of particle
    d -> diameter of particle
    OUTPUTS
    R -> radius of entire sphere
    """
    x = float(d/2)
    R = float((h**2 + x**2)/(2*h))
    return R

def graphSim(simulationData=None, title='Simulation', normalize=False, ax=None):
        realAxes = [-3.672692539241463, 3.672692539241463, -3.7645517111745592, 3.564962366343028]
        plt.sca(ax)  # Set current axes to the subplot passed in
        plt.title("Simulation: " + title)

        im = bp.plot_simres(simulationData, 
                            xlabel=r'$Q_{y} \;(1/{\rm nm})$', 
                            ylabel=r'$Q_{z} \;(1/{\rm nm})$', 
                            intenisty_min = 100,
                            vmin = 100,
                            zlabel=None,
                            with_cb=True,
                            cmap='jet')
        im.set_clim(vmin=100)     
        ax = im.axes  # Ensure formatting is applied to correct axes
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)

def max_particle_density(radius_nm: float, phi_max: float = 0.639) -> float:
    """
    Calculate the maximum particle number density (particles/nm^3)
    for spheres of given radius under 3D PY with packing limit.

    Parameters
    ----------
    radius_nm : float
        Particle radius in nanometers.
    phi_max : float, optional
        Maximum volume fraction. Default is 0.65 for 3D PY.

    Returns
    -------
    float
        Maximum particle density in particles/nm^3.
    """
    volume = (4/3) * np.pi * radius_nm**3  # nm^3
    density = phi_max / volume
    return density

def get_sample(radius):

    # Materials
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 1.09112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    #Roughness
    #----------------PS----------------------------------------------------
    hurst = 0.49
    corr = 84*nm
    sig = 3.2*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    roughness_PS = ba.Roughness(autocorr, ba.ErfTransient())

    offset = 7*nm
    spacing = 63*nm - offset
    num_samples = 10

    # Minimal test — adjust file path as needed
    #lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    #xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    #hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    #diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    #h_r.plt.show()   

    diam_K = [70]
    height_K = [15]
    weight_K=[1] 
    
    #form factor
    total_thickness = 214*nm

    P2VP_radius_xy = 47 #*nm
    P2VP_radius_z = P2VP_radius_xy - 25 #*nm
    p2vp_radius = P2VP_radius_xy

    Factor_xy = 0.6301396097097596 #P2VP_radius_xy / diam_K
    Factor_z =  2.615864606417858 #P2VP_radius_z / height_K

    Factor_xy = P2VP_radius_xy / diam_K[0]
    Factor_z = P2VP_radius_z / height_K[0]

    print('factor xy')
    print(Factor_xy)
    print('factor z')
    print(Factor_z)

    #density = max_particle_density(p2vp_radius)
    #density *= 0.1

    layer_PS_Top = ba.Layer(material_PS, 214.8*nm, roughness_PS)

    approximation = ba.Random3D_Dilute

    for i in range(1):
        #ff_P2VP = ba.Spheroid((diam_K[i] * Factor_xy) * nm, (height_K[i] * Factor_z) * nm)
        ff_P2VP = ba.Sphere(radius*nm)
        particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
        density = max_particle_density(radius)
        density = 4.66e-6
        layer_PS_Top.plugLiquid(density*250, particle_P2VP, approximation)

    
    # Interior
    iff = ba.InterferenceRadialParacrystal(spacing, 10000*nm)
    iff_pdf = ba.Profile1DGauss(6*nm)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(0.25)

    layout_top = ba.StructuredLayout(iff)

    offset = 7*nm
    spacing = 63*nm - offset

    for i in range(1):
        R = truncated_radius(height_K[i], diam_K[i] - offset)
        b = 2*R - height_K[i]
        ff_PS = ba.SphericalSegment(R* nm, 0.0*nm, b* nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        layout_top.addParticle(particle_PS, weight_K[i])
    layer_PS_Top.setNumberOfSlices(100)
    #----------------SiO2---------------------------------------------------
    hurst = 0.52
    corr = 10*nm
    sig = 0.2*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    roughness_SiO2 = ba.Roughness(autocorr, ba.ErfTransient())

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_vac.addStruct(1e-2, layout_top)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)
    
    # Sample
    sample = ba.Sample()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_Top)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

def get_simulation(P, sample):

    # Define specular scan:
    #axis = ba.ListScan("alpha_i (rad)", [0.106*deg,0.107*deg,0.108*deg,0.109*deg,0.11*deg,0.111*deg,0.112*deg,0.113*deg,0.114*deg,0.115*deg,0.116*deg,0.117*deg,0.118*deg,0.119*deg,0.12*deg,0.121*deg,0.122*deg,0.123*deg,0.124*deg,0.125*deg,0.126*deg,0.127*deg,0.128*deg,0.129*deg,0.13*deg,0.131*deg,0.132*deg,0.133*deg,0.134*deg,0.135*deg,0.136*deg,0.137*deg,0.138*deg,0.139*deg,0.14*deg,0.141*deg,0.142*deg,0.143*deg,0.144*deg,0.145*deg,0.146*deg,0.147*deg,0.148*deg,0.149*deg,0.15*deg,0.151*deg,0.152*deg,0.153*deg,0.154*deg,0.155*deg,0.156*deg,0.157*deg,0.158*deg,0.159*deg,0.16*deg,0.161*deg,0.162*deg,0.163*deg,0.164*deg,0.165*deg,0.166*deg,0.167*deg,0.168*deg,0.169*deg,0.17*deg,0.171*deg,0.172*deg,0.173*deg,0.174*deg,0.175*deg,0.176*deg,0.177*deg,0.178*deg,0.179*deg,0.18*deg,0.181*deg,0.182*deg,0.183*deg,0.184*deg,0.185*deg,0.186*deg,0.187*deg,0.188*deg,0.189*deg,0.19*deg,0.191*deg,0.192*deg,0.193*deg,0.194*deg,0.195*deg,0.196*deg,0.197*deg,0.198*deg,0.199*deg,0.2*deg,0.201*deg,0.202*deg,0.203*deg,0.204*deg,0.205*deg,0.206*deg,0.207*deg,0.208*deg,0.209*deg,0.21*deg,0.211*deg,0.212*deg,0.213*deg,0.214*deg,0.215*deg,0.216*deg,0.217*deg,0.218*deg,0.219*deg,0.22*deg,0.221*deg,0.222*deg,0.223*deg,0.224*deg,0.225*deg,0.226*deg,0.227*deg,0.228*deg,0.229*deg,0.23*deg,0.231*deg,0.232*deg,0.233*deg,0.234*deg,0.235*deg,0.236*deg,0.237*deg,0.238*deg,0.239*deg,0.24*deg,0.241*deg,0.242*deg,0.243*deg,0.244*deg,0.245*deg,0.246*deg,0.247*deg,0.248*deg,0.249*deg,0.25*deg,0.251*deg,0.252*deg,0.253*deg,0.254*deg,0.255*deg,0.256*deg,0.257*deg,0.258*deg,0.259*deg,0.26*deg,0.261*deg,0.262*deg,0.263*deg,0.264*deg,0.265*deg,0.266*deg,0.267*deg,0.268*deg,0.269*deg,0.27*deg,0.271*deg,0.272*deg,0.273*deg,0.274*deg,0.275*deg,0.276*deg,0.277*deg,0.278*deg,0.279*deg,0.28*deg,0.281*deg,0.282*deg,0.283*deg,0.284*deg,0.285*deg,0.286*deg,0.287*deg,0.288*deg,0.289*deg,0.29*deg,0.291*deg,0.292*deg,0.293*deg,0.294*deg,0.295*deg,0.296*deg,0.297*deg,0.298*deg,0.299*deg,0.3*deg,0.301*deg,0.302*deg,0.303*deg,0.304*deg,0.305*deg,0.306*deg,0.307*deg,0.308*deg,0.309*deg,0.31*deg,0.311*deg,0.312*deg,0.313*deg,0.314*deg,0.315*deg,0.316*deg,0.317*deg,0.318*deg,0.319*deg,0.32*deg,0.321*deg,0.322*deg,0.323*deg,0.324*deg,0.325*deg,0.326*deg,0.327*deg,0.328*deg,0.329*deg,0.33*deg,0.331*deg,0.332*deg,0.333*deg,0.334*deg,0.335*deg,0.336*deg,0.337*deg,0.338*deg,0.339*deg,0.34*deg,0.341*deg,0.342*deg,0.343*deg,0.344*deg,0.345*deg,0.346*deg,0.347*deg,0.348*deg,0.349*deg,0.35*deg,0.351*deg,0.352*deg,0.353*deg,0.354*deg,0.355*deg,0.356*deg,0.357*deg,0.358*deg,0.359*deg,0.36*deg,0.361*deg,0.362*deg,0.363*deg,0.364*deg,0.365*deg,0.366*deg,0.367*deg,0.368*deg,0.369*deg,0.37*deg,0.371*deg,0.372*deg,0.373*deg,0.374*deg,0.375*deg,0.376*deg,0.377*deg,0.378*deg,0.379*deg,0.38*deg,0.381*deg,0.382*deg,0.383*deg,0.384*deg,0.385*deg,0.386*deg,0.387*deg,0.388*deg,0.389*deg,0.39*deg,0.391*deg,0.392*deg,0.393*deg,0.394*deg,0.395*deg,0.396*deg,0.397*deg,0.398*deg,0.399*deg,0.4*deg,0.401*deg,0.402*deg,0.403*deg,0.404*deg,0.405*deg,0.406*deg,0.407*deg,0.408*deg,0.409*deg,0.41*deg,0.411*deg,0.412*deg,0.413*deg,0.414*deg,0.415*deg,0.416*deg,0.417*deg,0.418*deg,0.419*deg,0.42*deg,0.421*deg,0.422*deg,0.423*deg,0.424*deg,0.425*deg,0.426*deg,0.427*deg,0.428*deg,0.429*deg,0.43*deg,0.431*deg,0.432*deg,0.433*deg,0.434*deg,0.435*deg,0.436*deg,0.437*deg,0.438*deg,0.439*deg,0.44*deg,0.441*deg,0.442*deg,0.443*deg,0.444*deg,0.445*deg,0.446*deg,0.447*deg,0.448*deg,0.449*deg,0.45*deg,0.451*deg,0.452*deg,0.453*deg,0.454*deg,0.455*deg,0.456*deg,0.457*deg,0.458*deg,0.459*deg,0.46*deg,0.461*deg,0.462*deg,0.463*deg,0.464*deg,0.465*deg,0.466*deg,0.467*deg,0.468*deg,0.469*deg,0.47*deg,0.471*deg,0.472*deg,0.473*deg,0.474*deg,0.475*deg,0.476*deg,0.477*deg,0.478*deg,0.479*deg,0.48*deg,0.481*deg,0.482*deg,0.483*deg,0.484*deg,0.485*deg,0.486*deg,0.487*deg,0.488*deg,0.489*deg,0.49*deg,0.491*deg,0.492*deg,0.493*deg,0.494*deg,0.495*deg,0.496*deg,0.497*deg,0.498*deg,0.499*deg,0.5*deg])
    #scan = ba.AlphaScan(axis)
    n = 5000
    scan = ba.AlphaScan(n, 2*deg/n, 0.5*deg)
    
    scan.setWavelength(0.1541861561790254*nm)
    
    divergence = 1.8081312205577271e-05
    #divergence = 2.1081312205577271e-05
    #footprint_val = 0.0016305267098124154
    footprint_val = 0.5
        #footprint_val = 0.000001

    #simulating instrument
    alpha_distr = ba.DistributionGaussian(0, divergence)
    scan.setGrazingAngleDistribution(alpha_distr)
    footprint = ba.FootprintGauss(footprint_val)
    scan.setFootprint(footprint)
    simulation = ba.SpecularSimulation(scan, sample)
    simulation.options().setUseAvgMaterials(True)
    simulation.options().setIncludeSpecular(True)

    return simulation
def load_data(filename):
    datadir = os.getenv('BA_DATA_DIR', '')
    if not datadir:
        raise Exception("Environment variable BA_DATA_DIR not set")
    fname = os.path.join(datadir, filename)
    flags = ba.ImportSettings1D("2alpha (deg)", "*", "", 1, 2)
    return ba.readData1D(fname, ba.csv1D, flags)
# Define the plot function
def plot(ax, sim_result, real_data, label, plot_exp=False, color='C0', linestyle='-', marker=None):   
    y = dac.asNpArray(sim_result.dataArray()) 
    y /= y.max()
    x = dac.asNpArray(sim_result.xCenters()) 
    if plot_exp:
        ax.errorbar(dac.npArray(real_data.xCenters()),
                    dac.asNpArray(real_data.dataArray()),
                    label="Experiment",
                    markersize=1.,
                    linewidth=0.6,
                    color='r')

    ax.plot(x,y,
            label=f"{label} nm",
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
            marker=marker,
            markersize=3)

    ax.set_yscale('log')
    ax.set_xlabel("$q\;$(nm$^{-1}$)")
    ax.set_ylabel("$R$")
class PlotObserver:
    """
    Draws fit progress, for specular simulation.
    """

    def __init__(self, pause=0.0):
        self.pause = pause
        self._fig = plt.figure(figsize=(10, 7))
        self._fig.canvas.draw()

    def __call__(self, fit_objective):
        self.plot(fit_objective)
        

    def plot(self, fit_objective):
        #self._fig = plt.figure(figsize=(10, 7))
        #self._fig.canvas.draw()
        self._fig.clf()
        # retrieving data from fit suite
        exp_data = fit_objective.experimentalData().plottableField()
        sim_data = fit_objective.simulationResult().plottableField()

        # data values
        sim_values = dac.asNpArray(sim_data.dataArray())
        exp_values = dac.asNpArray(exp_data.dataArray())

        # default font properties dictionary to use
        font = { 'size': 16 }

        plt.yscale('log')
        plt.ylim((0.5*np.min(exp_values), 5*np.max(exp_values)))
        plt.plot(exp_data.axis(0).binCenters(), exp_values, 'k')
        plt.plot(sim_data.axis(0).binCenters(), sim_values, 'b')

        xlabel = bp.get_axes_labels(exp_data)[0]
        xlabel2 = bp.get_axes_labels(sim_data)[0]
        assert xlabel == xlabel2, f'Different labels: "{xlabel}" in exp vs "{xlabel2}" in sim'
        legend = ['Experiment', 'BornAgain']
        plt.legend(legend, loc='upper right', prop=font)
        plt.xlabel(xlabel, fontdict=font)
        plt.ylabel("Intensity", fontdict=font)
        self.plot_fit_parameters(fit_objective)
        plotargs = bp.parse_commandline()
        _do_show = plotargs.get('do_show', None)
        if _do_show:
            plt.pause(self.pause)
    @staticmethod
    def display_fit_parameters(fit_objective):
        """
        Displays fit parameters, chi and iteration number.
        """
        bp.plt.title('Parameters')
        bp.plt.axis('off')

        iteration_info = fit_objective.iterationInfo()

        bp.plt.text(
            0.01, 0.85, "Iterations  " +
            '{:d}'.format(iteration_info.iterationCount()))
        bp.plt.text(0.01, 0.75,
                 "Chi2       " + '{:8.4f}'.format(iteration_info.chi2()))
        for index, P in enumerate(iteration_info.parameters()):
            bp.plt.text(
                0.01, 0.55 - index*0.1,
                '{:30.30s}: {:6.3f}'.format(P.name(), P.value))
    @staticmethod
    
    def plot_fit_parameters(fit_objective):
        """
        Displays fit parameters, chi and iteration number.
        """

        iteration_info = fit_objective.iterationInfo()
        
        y = 0.5
        ax = plt.gca()

        bp.plt.text(
            0.7, y, "Iterations  " +
            '{:d}'.format(iteration_info.iterationCount()), transform=ax.transAxes)
        y += 0.05

        bp.plt.text(0.7, y,
                 "Chi2       " + '{:8.4f}'.format(iteration_info.chi2()), transform=ax.transAxes)
        y += 0.05
        
        for index, P in enumerate(iteration_info.parameters()):
            bp.plt.text(
                0.7, y, '{:30.30s}: {:6.3f}'.format(P.name(), P.value), transform=ax.transAxes)
            y += 0.05
            
    def show(self):
        plotargs = bp.parse_commandline()
        _do_show = plotargs.get('do_show', None)
        if _do_show:
            plt.show()
def fit():

    # Set environment and load data
    os.environ['BA_DATA_DIR'] = r'C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\Figures\XRR'
    filename = "40mgPml_4824_FAPbBr_oxyl_8Precursor - 4Bounce_q_norm.txt"
    data = load_data(filename)

    # Parameters and bounds:
    fixedPnB = {
        # to keep some parameters fixed, move lines here from startPnB
    }

    startPnB = {
        "SiO_r": (0.5, 0.1, 3),
        "PS_r":(2.5, 1.5, 4), 
        "PS_thickness":(289.65521612279844, 250, 300),
        "footprint_factor":(0.0016311452906756507, 0, 3),
        "divergence": (1.1134152471228338e-05, 0, 1e-4),
        "P2VP_radius": (1, 1, 30)
    }
    fixedP = {d: v[0] for d, v in fixedPnB1.items()}
    initialP = {d: v[0] for d, v in startPnB.items()}

    # Initial plot
    res = get_simulation(initialP | fixedP, sample= get_sample_parameters(P)).simulate()
    r = dac.asNpArray(res.dataArray())
    plot(r, data)
    #plt.show()
    
    # Fit:
    fit_objective = ba.FitObjective()
    fit_objective.setObjectiveMetric("chi2")
    
    fit_objective.addFitPair(
        lambda P: get_simulation(P | fixedP, sample= get_sample_parameters(P)), data, 1)
    
    #plot_observer = PlotObserver(pause=0.5)
    #fit_objective.initPlot(5, plot_observer)
    P = ba.Parameters()
    for name, p in startPnB.items():
        P.add(name, p[0], min=p[1], max=p[2])
    minimizer = ba.Minimizer()
    result = minimizer.minimize(fit_objective.evaluate, P)
    fit_objective.finalize(result)
    finalP = {r.name(): r.value for r in result.parameters()}
    # Print and plot fit outcome:

    print("Fit Result:")
    print(finalP)

    res = get_simulation(finalP | fixedP, sample= get_sample_parameters(finalP)).simulate()
    r = dac.asNpArray(res.dataArray())
    plot(r, data, finalP)

    plt.figure()
    zpoints, slds = sample_tools.materialProfile(sample= get_sample_parameters(finalP))
    plt.plot(zpoints, np.real(slds))

    plt.show()
def plotPSRadius():
    # Radii to simulate and plot
    P_P2VP_radius = [1, 5, 10, 15, 20, 25, 30]
    labels = P_P2VP_radius

    # Define line styles and markers for distinction
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', 'p']
    linestyle_cycle = itertools.cycle(linestyles)
    marker_cycle = itertools.cycle(markers)
    colors = cm.viridis(np.linspace(0, 1, len(P_P2VP_radius)))

    # Define the plot function
    def plot(ax, sim_result, real_data, label, plot_exp=False, color='C0', linestyle='-', marker=None):    
        if plot_exp:
            y = dac.asNpArray(real_data.dataArray())
            y /= y.max()
            ax.errorbar(dac.npArray(real_data.xCenters()),
                        y,
                        label="Experiment",
                        markersize=1.,
                        linewidth=0.6,
                        color='r')
        y = sim_result
        y /= y.max()
        ax.plot(dac.npArray(real_data.xCenters()), y,
                label=f"{label} nm",
                color=color,
                linewidth=0.8,
                linestyle=linestyle,
                marker=marker,
                markersize=3)

        ax.set_yscale('log')
        ax.set_xlabel("$q\;$(nm$^{-1}$)")
        ax.set_ylabel("$R$")

    # Create main figure and axis
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Run simulations and plot each
    for i, (radius, label) in enumerate(zip(P_P2VP_radius, labels)):
        sample = get_sample(radius)
        res = get_simulation(radius, sample).simulate()
        r = dac.asNpArray(res.dataArray())

        plot(ax, r, data, label,
            plot_exp=(i == 0),
            color=colors[i],
            linestyle=next(linestyle_cycle),
            marker=next(marker_cycle))

    ax.legend(title="Simulated Radius")
    plt.tight_layout()
    plt.show()
def save_data(x: np.ndarray, y: np.ndarray, directory: str, fname: str) -> str:
    """
    Save simulation data (x, y) to a text file in the specified directory.

    Parameters:
    - x: 1D numpy array of x values.
    - y: 1D numpy array of y values.
    - directory: Target directory path where the file will be saved.
    - fname: Desired filename (with or without .txt extension).

    Returns:
    - The full path to the saved file.
    """
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    # Add .txt extension if missing
    if not fname.lower().endswith('.txt'):
        fname += '.txt'

    # Construct full file path
    filepath = os.path.join(directory, fname)

    # Stack x and y into a 2D array of shape (N, 2)
    data = np.column_stack((x, y))

    # Save using NumPy's savetxt with default delimiter and header
    np.savetxt(filepath, data, delimiter='\t', header='x\ty', comments='')

    return filepath
if __name__ == '__main__':

    # Set environment and load data
    os.environ['BA_DATA_DIR'] = r'C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\Figures\XRR\sample31'
    filename = "31_FAPbBr_40gPL_16Pre_2000RPM_normalized.txt"
    data = load_data(filename)

    P_FA_particleDensity = [1, 2, 4,  8, 16, 32]

    labels = P_FA_particleDensity

    # Define line styles and markers for distinction
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', 'p']
    linestyle_cycle = itertools.cycle(linestyles)
    # Use a more visually friendly and colorblind-safe colormap
    marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', '*', 'X'])
    cmap = cm.get_cmap('tab10') if len(P_FA_particleDensity) <= 10 else cm.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(P_FA_particleDensity))]

    # Create main figure and axis
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ConversionFactor = 2.8179e-15 * 1e10
    resTotal = []
    # Run simulations and plot each
    for i, (radius, label) in enumerate(zip(P_FA_particleDensity, labels)):
        sample = get_sample(radius)
        res = get_simulation(radius, sample).simulate()
        res.setTitle(str(label))
        resTotal.append(res)

        #save data
        pfield = res.plottableField()
        x = pfield.axis(0).binCenters()
        y = dac.asNpArray(pfield.dataArray())
        save_data(x, y, 
                  directory= r'C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\XRR Simulations\PS-b-P2VP - FAPbBr',
                  fname= 'python_sim_' + str(label)
                  )

        plot(ax, res, data, label,
            plot_exp=(i == 0),
            color=colors[i],
            linestyle=next(linestyle_cycle),
            marker=next(marker_cycle))
        
        zpoints, slds = sample_tools.materialProfile(sample)
        ax2.plot(zpoints, np.real(slds))
    resTotal.append(data)

    ax.legend(title="Simulated Radius")
    plt.tight_layout()
    plt.figure()
    bp.plot_multicurve(resTotal)
    plt.show()