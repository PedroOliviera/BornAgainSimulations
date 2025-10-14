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

def get_sample_withInterior(approximation):

    # Materials
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.39112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
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
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    #h_r.plt.show()    
    
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

    for i in range(2):
        #ff_P2VP = ba.Spheroid((diam_K[i] * Factor_xy) * nm, (height_K[i] * Factor_z) * nm)
        ff_P2VP = ba.Sphere(23*nm)
        particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
        density = max_particle_density(23)
        layer_PS_Top.plugLiquid(density*nm, particle_P2VP, approximation)

    
    # Interior
    iff = ba.InterferenceRadialParacrystal(spacing, 10000*nm)
    iff_pdf = ba.Profile1DGauss(6*nm)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(0.25)

    layout_top = ba.StructuredLayout(iff)

    offset = 7*nm
    spacing = 63*nm - offset

    for i in range(10):
        R = truncated_radius(height_K[i], diam_K[i] - offset)
        b = 2*R - height_K[i]
        ff_PS = ba.SphericalSegment(R* nm, 0.0*nm, b* nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        layout_top.addParticle(particle_PS, weight_K[i])

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

def get_sample(approximation, p2vp_radius):

    # Materials
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.09112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.09112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)

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
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    #h_r.plt.show()    
    
    #form factor
    total_thickness = 214*nm

    P2VP_radius_xy = 47 #*nm
    P2VP_radius_z = P2VP_radius_xy - 30 #*nm
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

    layer_PS_Top = ba.Layer(material_PS, 214.8*nm)
    for i in range(1):
        ff_P2VP = ba.Spheroid((diam_K[i] * Factor_xy) * nm, (height_K[i] * Factor_z) * nm)
        #ff_P2VP = ba.Sphere(p2vp_radius*nm)
        particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
        density = max_particle_density(diam_K[i] * Factor_xy*2 * 1000)
        layer_PS_Top.plugLiquid(density, particle_P2VP, approximation)

    #----------------SiO2---------------------------------------------------
    hurst = 0.52
    corr = 10*nm
    sig = 0.2*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    roughness_SiO2 = ba.Roughness(autocorr, ba.ErfTransient())

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)
    
    # Sample
    sample = ba.Sample()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_Top)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

def get_sample_2D():

    # Materials
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.09112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.29112645E-06, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)

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
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    #h_r.plt.show()    
    
    #form factor
    total_thickness = 214*nm

    P2VP_radius_xy = 24 #*nm
    P2VP_radius_z = P2VP_radius_xy - 14 #*nm
    p2vp_radius = P2VP_radius_xy

    Factor_xy = 0.32177342  #P2VP_radius_xy / diam_K
    Factor_z =  1.53874389 #P2VP_radius_z / height_K

    print('factor xy')
    print(Factor_xy)
    print('factor z')
    print(Factor_z)

    density_nm2=3.6e-4

    layer_thickness = 214/4 * nm
    layer_polymer1 = ba.Layer(material_PS, layer_thickness)
    layer_polymer2 = ba.Layer(material_PS, layer_thickness)
    layer_polymer3 = ba.Layer(material_PS, layer_thickness)
    layer_polymer4 = ba.Layer(material_PS, layer_thickness)

    for i in range(1):
        ff_P2VP = ba.Spheroid((diam_K[i] * Factor_xy) * nm, (height_K[i] * Factor_z) * nm)
        #ff_P2VP = ba.Sphere(p2vp_radius*nm)
        particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
        layer_polymer1.depositParticle(density_nm2, particle_P2VP)
        layer_polymer2.depositParticle(density_nm2, particle_P2VP)
        layer_polymer3.depositParticle(density_nm2, particle_P2VP)
        layer_polymer4.depositParticle(density_nm2, particle_P2VP)

    

    #----------------SiO2---------------------------------------------------
    hurst = 0.52
    corr = 10*nm
    sig = 0.2*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    roughness_SiO2 = ba.Roughness(autocorr, ba.ErfTransient())

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)
    
    # Sample
    sample = ba.Sample()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_polymer1)
    sample.addLayer(layer_polymer2)
    sample.addLayer(layer_polymer3)
    sample.addLayer(layer_polymer4)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

def get_simulation(sample):
    beam = ba.Beam(4e11, 1.25916*ba.angstrom, 0.15*deg)

    n = 1000
    detector = ba.SphericalDetector(n, 0*deg, 0.5*deg, n, 0., 0.5*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation
if __name__ == '__main__':
    radi = [10,20,30,40]
    samples = [
        #get_sample(ba.Random3D_PY, 3),
        #get_sample(ba.Random3D_Dilute, 3)
        get_sample_withInterior(ba.Random3D_Dilute),
        get_sample_withInterior(ba.Random3D_PY)
        #get_sample(ba.Random3D_PY, 25),
        #get_sample(ba.Random3D_Dilute, 25),
        #get_sample(ba.Random3D_PY, 45),
        #get_sample(ba.Random3D_Dilute, 45),
        #get_sample_2D()
    ]
    results = [ get_simulation(sample).simulate() for sample in samples ]
    labels = ["PY Model", "Dilute", "PY Model", "Dilute","PY Model", "Dilute", "2D", "AAA"] #[20*nm, 25*nm, 27*nm, 30*nm, 33*nm]
    for label, r in zip(labels, results):
        simulationData = dac.asNpArray(r.dataArray())
        save_filename = "test_" + str(label) + "_spheres_distribution_15deg_3D.npy"
        #np.save(save_filename, simulationData)
        #graphing.plot2D(simulationData=simulationData, realDat_axes=[0, 0.5, 0, 0.5], zlim=[0.02,1e3], title=label)
        bp.plt.figure()
        bp.plot_datafield(r.flat(), cmap='jet', intensity_min = 0.1, intensity_max = 1e3)
    bp.plt.show()