#!/usr/bin/env python3
"""
Dilute film of small spheres
"""
import bornagain as ba
from bornagain import ba_plot as bp, deg, nm
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

def get_sample(approximation,radius):

    # Materials
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP",3E-6, 2.35E-9) #2.51436745E-06, 2.35391329E-09)
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

    surface_layout = ba.StructuredLayout(iff)
    
    #surface_layout.setTotalParticleSurfaceDensity(0.0265)


    for i in range(P1):
        
        ff_PS = ba.SpheroidalSegment((diam_K[i]/2) * nm, height_K[i]/2 * nm, 0, height_K[i]/2 * nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        #surface_layout.addParticle(particle_PS, weight_K[i])

    #layer_vac.addStruct(surface_layout)

    # Internal Particles
    
    density = max_particle_density(radius)
    print(density)
    
    distr = ba.DistributionGaussian(radius*nm, radius*0.1*nm)
    for parsample in distr.distributionSamples():
        ff = ba.Sphere(parsample.value)
        print(parsample.value)
        print(parsample.weight * density)
        #ff = ba.Spheroid(radius * nm, radius/1.5 * nm)
        particle = ba.Particle(material_P2VP, ff)
        layer_PS_Top.plugLiquid(density * parsample.weight, particle, approximation)
    
    #ff = ba.Sphere(radius * nm)
    
    #particle = ba.Particle(material_P2VP, ff)

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
        get_sample(ba.Random3D_PY,radi[0]),
        get_sample(ba.Random3D_PY,radi[1]),
        get_sample(ba.Random3D_PY,radi[2]),
        #get_sample(ba.Random3D_PY,30),
        get_sample(ba.Random3D_PY,radi[3]),
    ]
    results = [ get_simulation(sample).simulate() for sample in samples ]
    labels = [radi[0],radi[1],radi[2],radi[3]] #[20*nm, 25*nm, 27*nm, 30*nm, 33*nm]
    for label, r in zip(labels, results):
        simulationData = dac.asNpArray(r.dataArray())
        save_filename = "test_" + str(label) + "_spheres_distribution_15deg_3D.npy"
        np.save(save_filename, simulationData)
        graphing.plot2D(simulationData=simulationData, realDat_axes=[0, 0.5, 0, 0.5], zlim=[0.1,200], title=label)