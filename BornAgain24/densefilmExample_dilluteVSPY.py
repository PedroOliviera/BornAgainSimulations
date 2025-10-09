#!/usr/bin/env python3
"""
Dilute film of small spheres
"""
import bornagain as ba
from bornagain import ba_plot as bp, deg, nm, R3
import height_radius_from_lineprofiles as h_r
import matplotlib.pyplot as plt
from bornagain.numpyutil import Arrayf64Converter as dac
import numpy as np
import GraphingAnalysis as graphing

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

def get_sample(approximation, p2vp_radius):

    # Materials
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP",3E-6, 2.35E-9) #2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.09112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)

    offset = 7*nm
    spacing = 63*nm - offset
    num_samples = 10

    # Minimal test — adjust file path as needed
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    h_r.plt.show()    
    
    #form factor
    total_thickness = 214*nm

    P2VP_radius_xy = 24 #*nm
    P2VP_radius_z = P2VP_radius_xy - 14 #*nm

    Factor_xy = 0.32177342  #P2VP_radius_xy / diam_K
    Factor_z =  1.53874389 #P2VP_radius_z / height_K

    print('factor xy')
    print(Factor_xy)
    print('factor z')
    print(Factor_z)

    for i in range(10):
        ff_P2VP = ba.Spheroid((diam_K[i] * Factor_xy) * nm, (height_K[i] * Factor_z) * nm)
        particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
        layer_PS_Top.plugLiquid(density * weight_K[i], particle_P2VP, approximation)

    #Roughness
    #----------------PS----------------------------------------------------
    hurst = 0.49
    corr = 84*nm
    sig = 3.2*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    roughness_PS = ba.Roughness(autocorr, ba.ErfTransient())

    #----------------SiO2---------------------------------------------------
    hurst = 0.52
    corr = 10*nm
    sig = 0.2*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    roughness_SiO2 = ba.Roughness(autocorr, ba.ErfTransient())

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, 214.8*nm, roughness_PS)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm, roughness_SiO2)
    layer_Si = ba.Layer(material_Si_Sub)
    
    
    omega_order = 9*nm
    spacing = 60*nm

    P1 = 3

    # Minimal test — adjust file path as needed
    lineprofile_dir = r"C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Characterization\AFM\2024\4-29-2024\lineProfiles_35_Big_OnePerParticle.txt"
    
    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=P1, scale=True)
    #h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)

    #########################################----SURFACE PARTICLES----################################################

    # Interference Functions
    iff = ba.InterferenceRadialParacrystal(spacing, 250*nm)
    iff_pdf = ba.Profile1DGauss(omega_order)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(1.5) #size-distribution model 

    #surface_layout = ba.StructuredLayout(iff)
    
    #surface_layout.setTotalParticleSurfaceDensity(0.0265)


    for i in range(P1):
        ff_PS = ba.SpheroidalSegment((diam_K[i]/2) * nm, height_K[i]/2 * nm, 0, height_K[i]/2 * nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        #surface_layout.addParticle(particle_PS, weight_K[i])

    #layer_vac.addStruct(surface_layout)

    # Internal Particles
    
    density = max_particle_density(p2vp_radius)
    print(density)
    
    '''
    distr = ba.DistributionGaussian(radius*nm, radius*0.1*nm)
    for parsample in distr.distributionSamples():
        ff = ba.Sphere(parsample.value)
        #ff = ba.Spheroid(radius * nm, radius/1.5 * nm)
        particle = ba.Particle(material_P2VP, ff)
        layer_PS_Top.plugLiquid(density * parsample.weight, particle, approximation)
    '''
    #p2vp_radius = 15*nm
    #PS_radius = 30*nm
    #core_ff = ba.Sphere(p2vp_radius)
    #shell_ff = ba.Sphere(PS_radius)
    #ff = ba.Spheroid(radius * nm, radius/1.5 * nm)

    #core_particle = ba.Particle(material_P2VP, core_ff)
    #shell_particle = ba.Particle(material_PS, shell_ff)
    #compound = ba.Compound()
    #compound.addComponent(shell_particle)
    #compound.addComponent(core_particle, R3(0, 0, 15*nm))
    #compound.setRadius(radius = 30.0*nm)
    #print(compound.radius())
    #layer_PS_Top.plugLiquid(density, compound, approximation)

    # Sample
    sample = ba.Sample()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_Top)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

def get_simulation(sample):
    beam = ba.Beam(1e9, 1.25916*ba.angstrom, 0.15*deg)

    n = 1000
    detector = ba.SphericalDetector(n, -0.5*deg, 0.5*deg, n, 0., 1*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation
if __name__ == '__main__':
    radi = [10,20,30,40]
    samples = [
        get_sample(ba.Random3D_PY, 25),
        get_sample(ba.Random3D_Dilute, 25)
    ]
    results = [ get_simulation(sample).simulate() for sample in samples ]
    labels = ["PY Model", "Dilute"] #[20*nm, 25*nm, 27*nm, 30*nm, 33*nm]
    for label, r in zip(labels, results):
        simulationData = dac.asNpArray(r.dataArray())
        save_filename = "test_" + str(label) + "_spheres_distribution_15deg_3D.npy"
        np.save(save_filename, simulationData)
        graphing.plot2D(simulationData=simulationData, realDat_axes=[0, 0.5, 0, 0.5], zlim=[0.01,4.7e3], title=label)
    plt.show()