#!/usr/bin/env python3
"""
Basic example of a DWBA simulation of a GISAS experiment.
"""
import bornagain as ba
from bornagain import deg, nm, R3
from bornagain import ba_plot as bp, deg, nm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from scipy.ndimage import shift, rotate

wavelength = 1.25916*ba.angstrom

# detector setup as given from instrument responsible
rayonix_npx, rayonix_npy = 4096, 4096 
rayonix_pixel_size = 0.073242  # in mm
xpos_mm = 151.63
ypos_mm = 142.137
xpos_pix = 2048
ypos_pix = 2048
beam_xpos, beam_ypos = xpos_pix, ypos_pix  # in pixels
rayonix_size_x = 300
rayonix_size_y = 300

def detectorQtoMM(beamTime, SimulationQspace):
    '''Converts desired simulation Qspace to real space detector area'''
    DecFullQspace = [-2.446074535755995, 2.446074535755995, -2.4460745357559945, 2.4460745357559954] #
    FebFullQspace = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517] # the full qspace measured by detector in feb beamtime
    detector_realSpace = [300, 300, 300, 300]
    if beamTime.lower() == 'feb':
        DesiredRealSpace = SimulationQspace * ( detector_realSpace / FebFullQspace ) 
def read_profile_data(file_name):
    '''Function to read the profile data from the text file'''
    profile_numbers = []
    height_diffs = []
    diameter_diffs = []
    with open(file_name, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            data = line.split()
            profile_numbers.append(int(data[0]))
            height_diffs.append(float(data[1]))
            diameter_diffs.append(float(data[2]))
 
    return height_diffs, diameter_diffs
def normalize_histogram(data, bins=10):
    """
    Reads data from a text file, calculates a normalized histogram, and returns the bin values and normalized frequencies.
    
    Parameters:
    file_path (str): The path to the text file containing the data.
    bins (int): The number of bins for the histogram. Default is 30.
    
    Returns:
    bin_values (numpy array): The bin values.
    normalized_frequencies (numpy array): The normalized frequencies.
    """
    
    # Calculate the histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Normalize the counts
    normalized_counts = counts / counts.max()
    
    # Calculate bin values as the center of each bin
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_values, normalized_counts

def create_detector(detector_distance):
    """
    A model of the GALAXY detector
    """
    u0 = beam_xpos*rayonix_pixel_size  # in mm
    v0 = beam_ypos*rayonix_pixel_size  # in mm
    detector = ba.RectangularDetector(rayonix_npx,
                                      rayonix_npx*rayonix_pixel_size,
                                      rayonix_npy,
                                      rayonix_npy*rayonix_pixel_size)
    #rayonix_npx - number of pixels in x (4096) - ray_pixel_size: 0.073242 mm
    #rayonix_npx*rayonix_pixel_size = 300 mm
    detector.setPerpendicularToDirectBeam(detector_distance, u0, v0)
    return detector
def get_sampleSpheroids2(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 31.4*nm
    omega_order = 5*nm
    radius_PS = 31.4/2*nm
    radius_P2VP = (30.86/3)*nm #from iron oxide supplemental
    #height = P*nm 
    height = 2*nm

    #------------------Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    ff_PS = ba.HemiEllipsoid(radius_PS, radius_PS, height) #models PS as a hemisphere 
    particle_PS = ba.Particle(material_PS, ff_PS)
    layout.addParticle(particle_PS, 1)   

    #------------------Uniform PS Particles-----------------------------
    layout2 = ba.ParticleLayout()
    ff_P2VP = ba.Spheroid(radius_P2VP, radius_P2VP - 2*nm) #models PS as a hemisphere 
    particle_P2VP = ba.Particle(material_FA, ff_P2VP)
    layout2.addParticle(particle_P2VP, 1)  
    
    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    layout2.setInterference(iff)

    iff3 = ba.InterferenceRadialParacrystal(spacing - 5*nm, 400*nm)
    iff4 = ba.InterferenceRadialParacrystal(spacing - 10*nm, 400*nm)  
    iff3.setProbabilityDistribution(iff_pdf)
    iff4.setProbabilityDistribution(iff_pdf)

    layout3 = layout2
    layout4 = layout2
    layout3.setInterference(iff3)
    layout4.setInterference(iff4)
    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.1*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 1, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 130/3*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_PS, 130/3*nm)
    layer_3.addLayout(layout2)
    layer_4 = ba.Layer(material_PS, 130/3*nm)
    layer_4.addLayout(layout3)
    layer_6 = ba.Layer(material_SiO2, 2*nm)
    layer_6.addLayout(layout4)
    layer_7 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_6)
    sample.addLayer(layer_7)

    return sample
def get_sampleSpheroids(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 31.4*nm
    omega_order = 5*nm
    radius_PS = 31.4/2*nm
    radius_P2VP = (30.86/3)*nm #from iron oxide supplemental
    #height = P*nm 
    height = 2*nm

    #------------------Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    ff_PS = ba.HemiEllipsoid(radius_PS, radius_PS, height) #models PS as a hemisphere 
    particle_PS = ba.Particle(material_PS, ff_PS)
    layout.addParticle(particle_PS, 1)   

    #------------------Uniform PS Particles-----------------------------
    layout2 = ba.ParticleLayout()
    ff_P2VP = ba.Spheroid(radius_P2VP, radius_P2VP - 0.5*nm) #models PS as a hemisphere 
    particle_P2VP = ba.Particle(material_FA, ff_P2VP)
    layout2.addParticle(particle_P2VP, 1)  
    
    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    layout2.setInterference(iff)

    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.1*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 1, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 130*nm)
    layer_2.addLayout(layout)
    layer_6 = ba.Layer(material_SiO2, 2*nm)
    layer_6.addLayout(layout2)
    layer_7 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_6)
    sample.addLayer(layer_7)

    return sample
def get_sampleSpheres(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 31.4*nm
    omega_order = 5*nm
    radius_PS = 31.4/2*nm
    radius_P2VP = (30.86/3)*nm #from iron oxide supplemental
    #height = P*nm 
    height = 2*nm

    #------------------Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    ff_PS = ba.HemiEllipsoid(radius_PS, radius_PS, height) #models PS as a hemisphere 
    particle_PS = ba.Particle(material_PS, ff_PS)
    layout.addParticle(particle_PS, 1)   

    #------------------Uniform PS Particles-----------------------------
    layout2 = ba.ParticleLayout()
    ff_P2VP = ba.Sphere(10*nm) #models PS as a hemisphere 
    particle_P2VP = ba.Particle(material_FA, ff_P2VP)
    layout2.addParticle(particle_P2VP, 1)  
    
    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    #layout2.setInterference(iff)

    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.1*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 1, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 40*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_PS, 40*nm)
    layer_3.addLayout(layout2)
    layer_4 = ba.Layer(material_PS, 40*nm)
    layer_4.addLayout(layout2)
    layer_5 = ba.Layer(material_PS, 10*nm)
    layer_5.addLayout(layout2)
    layer_6 = ba.Layer(material_SiO2, 2*nm)
    layer_7 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_6)
    sample.addLayer(layer_7)

    return sample
def get_sampleHemispheres(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 40*nm
    omega_order = 7*nm
    radius_PS = (spacing/2)*nm
    radius_P2VP = 13.5*nm #from iron oxide supplemental
    #height = P*nm 
    height = 6*nm

    #------------------Non- Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    distr = ba.DistributionGaussian(radius_PS, 5*nm)
    for parsample in distr.distributionSamples():
        ff_PS = ba.HemiEllipsoid(parsample.value, parsample.value, height)
        particle_PS= ba.Particle(material_PS, ff_PS)
        layout.addParticle(particle_PS, parsample.weight)

    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    
    #layout.setInterference(iff)
    
    j = 3
    layout_j = [ba.ParticleLayout() for i in range(j)]
    for i in range(j):
        NEWspacing = spacing #+ 3 * nm * j
        distr = ba.DistributionGaussian(radius_P2VP, 1*nm)
        for parsample in distr.distributionSamples():
            ff_P2VP = ba.Sphere(parsample.value)
            particle_P2VP = ba.Particle(material_FA, ff_P2VP)
            layout_j[i].addParticle(particle_P2VP, parsample.weight)  
            iff2 = ba.InterferenceRadialParacrystal(NEWspacing, 100*nm)  
            iff2.setProbabilityDistribution(iff_pdf)
            layout_j[i].setInterference(iff2)

    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.2*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 0.75, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    print('PS radius: ' + str(radius_PS))
    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, (spacing + 3)*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_PS, (spacing + 3)*nm)
    layer_3.addLayout(layout_j[0])
    layer_4 = ba.Layer(material_PS, (spacing + 3)*nm)
    layer_4.addLayout(layout_j[1])
    layer_5 = ba.Layer(material_PS, 10*nm)
    layer_5.addLayout(layout_j[2])
    layer_6 = ba.Layer(material_PS, 3*nm)
    layer_7 = ba.Layer(material_SiO2, 2*nm)
    layer_8 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayerWithTopRoughness(layer_2,roughness_PS)
    sample.addLayerWithTopRoughness(layer_3,roughness_PS)
    sample.addLayerWithTopRoughness(layer_4,roughness_PS)
    sample.addLayerWithTopRoughness(layer_5,roughness_PS)
    sample.addLayerWithTopRoughness(layer_6,roughness_PS)
    sample.addLayerWithTopRoughness(layer_7,roughness_PS)
    sample.addLayer(layer_8)

    return sample
def get_sample4(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 31.4*nm
    omega_order = 5*nm
    radius_PS = 31.4/2*nm
    radius_P2VP = (30.86/3)*nm #from iron oxide supplemental
    #height = P*nm 
    height = 2*nm

    #------------------Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    ff_PS = ba.HemiEllipsoid(radius_PS, radius_PS, height) #models PS as a hemisphere 
    particle_PS = ba.Particle(material_PS, ff_PS)
    layout.addParticle(particle_PS, 1)   

    #------------------Distribution of P2VP Particles-----------------------------
    layout2 = ba.ParticleLayout()
    distr = ba.DistributionGaussian(radius_P2VP, 4*nm)
    for parsample in distr.distributionSamples():
        ff_P2VP = ba.Sphere(parsample.value)
        particle_P2VP= ba.Particle(material_P2VP, ff_P2VP)
        layout2.addParticle(particle_P2VP, parsample.weight)
    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    layout2.setInterference(iff)

    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.1*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 1, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 130*nm)
    layer_2.addLayout(layout)
    layer_5 = ba.Layer(material_SiO2, 2*nm)
    layer_5.addLayout(layout2)
    layer_6 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_5)
    sample.addLayer(layer_6)

    return sample
def get_sample3(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 31.4*nm
    omega_order = 5*nm
    radius_PS = 31.4/2*nm
    radius_P2VP = (30.86/3)*nm #from iron oxide supplemental
    #height = P*nm 
    height = 2*nm

    #------------------Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    distr = ba.DistributionGaussian(radius_PS, 6*nm)
    for parsample in distr.distributionSamples():
        ff_PS = ba.HemiEllipsoid(parsample.value, parsample.value, height)
        particle_PS= ba.Particle(material_PS, ff_PS)
        layout.addParticle(particle_PS, parsample.weight)  

    #------------------Distribution of P2VP Particles-----------------------------
    layout2 = ba.ParticleLayout()
    distr = ba.DistributionGaussian(radius_P2VP, 4*nm)
    for parsample in distr.distributionSamples():
        ff_P2VP = ba.Sphere(parsample.value)
        particle_P2VP= ba.Particle(material_P2VP, ff_P2VP)
        layout2.addParticle(particle_P2VP, parsample.weight)
    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    layout2.setInterference(iff)

    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.1*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 1, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 130*nm)
    layer_2.addLayout(layout)
    layer_5 = ba.Layer(material_SiO2, 2*nm)
    layer_5.addLayout(layout2)
    layer_6 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_5)
    sample.addLayer(layer_6)

    return sample
def get_sample5(P=None):
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.353858E-09)
    #material_PS = ba.RefractiveMaterial("PS", P, 2.353858E-09) 
    #material_P2VP = ba.RefractiveMaterial("P2VP", 2.7E-06, 2.58320298E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.53E-06, 2.58320298E-09)
    material_FA = ba.RefractiveMaterial("FA", 6.92973344E-06, 3.17591258E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 39*nm
    omega_order = 7*nm
    radius_PS = 39/2*nm
    radius_P2VP = P*nm #from iron oxide supplemental
    #height = P*nm 
    height = 2*nm

    #------------------Uniform PS Particles-----------------------------
    layout = ba.ParticleLayout()
    ff_PS = ba.HemiEllipsoid(radius_PS, radius_PS, height) #models PS as a hemisphere 
    particle_PS = ba.Particle(material_PS, ff_PS)
    layout.addParticle(particle_PS, 1)   

    #------------------Uniform PS Particles-----------------------------
    layout2 = ba.ParticleLayout()
    ff_P2VP = ba.Sphere(radius_P2VP) #models PS as a hemisphere 
    particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
    layout2.addParticle(particle_P2VP, 1)  
    
    #Radial Paracrystal
    # Define interference functions
    ####----------- Radial Paracrystal -----------####
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff.setProbabilityDistribution(iff_pdf)
    layout.setInterference(iff)

    iff2 = ba.InterferenceRadialParacrystal(spacing, 400*nm)  
    iff_pdf = ba.Profile1DGauss(omega_order)  
    iff2.setProbabilityDistribution(iff_pdf)
    #layout2.setInterference(iff2)

    #layout2.setTotalParticleSurfaceDensity(P)
    #roughness PS
    sigma, hurst, corrLength = 0.1*nm, 0.6, 200*nm
    roughness_Si = ba.LayerRoughness(sigma, hurst, corrLength)

    #roughness Si
    sigma, hurst, corrLength = 6*nm, 1, 200*nm
    roughness_PS = ba.LayerRoughness(sigma, hurst, corrLength)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 40*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_PS, 40*nm)
    layer_3.addLayout(layout2)
    layer_4 = ba.Layer(material_PS, 40*nm)
    layer_4.addLayout(layout2)
    layer_5 = ba.Layer(material_PS, 10*nm)
    layer_5.addLayout(layout2)
    layer_6 = ba.Layer(material_SiO2, 2*nm)
    layer_7 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    sample.addLayer(layer_4)
    sample.addLayer(layer_5)
    sample.addLayer(layer_6)
    sample.addLayer(layer_7)

    return sample
def get_simulation_2D(sample_model, detectorDistBeamtime = 'feb', angle = None, beamIntensity = 1.3e12, ROI = None):
    '''
    sample: getSample(P=stuff)
    P : Sample Parameter Variation
    Angle : incidence angle in degree without units
    detectorDistBeamtime: detector distance - can be either Dec or Feb
    '''
    if (detectorDistBeamtime == 'feb'):
        detectorDist = 2337.126
    elif (detectorDistBeamtime == 'dec'):
        detectorDist = 3052.624

    sample = sample_model

    alpha_i = angle*ba.deg

    # Beam
    beam = ba.Beam(beamIntensity, wavelength, alpha_i) 

    # Detector
    detector = create_detector(detectorDist)
       
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    background = 23
    simulation.options().setIncludeSpecular(True)
    simulation.setBackground(ba.ConstantBackground(background))
    
    #ROI to speed up simulation
    simulation.detector().addMask(ba.Rectangle(148.07181022335135, 140.59419621147006, 152.0184854930185, 177.91172274246256, False))
    #simulation.detector().setRegionOfInterest(151.41087056827945, 154.70290189426495, 210, 210)
    #simulation.detector().setRegionOfInterest(148.07181022335135,140.59419621147006, 197.0290189426496, 177.91172274246256)
    simulation.detector().setRegionOfInterest(120, 155, 180, 210)
    if ROI is not(None):
        simulation.detector().setRegionOfInterest(ROI[0], ROI[1], ROI[2], ROI[3])
    



    

    return simulation

def beamStopMask(x1 = - 0.041 , y1 = - 0.2, x2 = 0.04292, y2 = 0.5935):
    '''How mask for beamstop was calculated'''
    rayonix_size_y = 300
    rayonix_size_x = 300

    BOTLEFT_value_y = y1 
    BOTLEFT_value_x = x1

    TOPRIGHT_value_y = y2
    TOPRIGHT_value_x = x2

    axesLimits = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517]

    q_dist_y1 = BOTLEFT_value_y - axesLimits[0]
    q_dist_x1 = BOTLEFT_value_x - axesLimits[2]

    q_dist_y2 = TOPRIGHT_value_y - axesLimits[0]
    q_dist_x2 = TOPRIGHT_value_x - axesLimits[2]


    q_to_mm_ConversionFactor_y = rayonix_size_y / (axesLimits[1] - axesLimits[0]) 
    q_to_mm_ConversionFactor_x = rayonix_size_x / (axesLimits[3] - axesLimits[2]) 

    mm_BOTLEFT_y = q_dist_y1 * q_to_mm_ConversionFactor_y
    mm_BOTLEFT_x = q_dist_x1 * q_to_mm_ConversionFactor_x

    mm_TOPRIGHT_y = q_dist_y2 * q_to_mm_ConversionFactor_y
    mm_TOPRIGHT_x = q_dist_x2 * q_to_mm_ConversionFactor_x

    print(mm_BOTLEFT_x) 
    print(mm_BOTLEFT_y) 
    print(mm_TOPRIGHT_x)
    print(mm_TOPRIGHT_y)

def get_simulation_line(sample_model, detectorDistBeamtime, angle_of_incidence, center_horizontal_slice_value, center_vertical_slice_value, number_slices):
    '''
    sample: getSample()
    P : Sample Parameter Variation
    A : simulation parameter (angle, etc.)
    detectorDistBeamtime: detector distance - can be either Dec or Feb
    angle: angle of incidence
    center_horizontal_slice_value: center Qy value that is simulated
    number_slices: number of horizontal slices simulated on either side of center (total slice 1 +  2 * n, where n is this value)
    '''
    ROI_x1 = 120 #152 
    ROI_x2 = 180 #175
    ROI_y1 = 155 #150
    ROI_y2 = 210 #205

    if (detectorDistBeamtime == 'feb'):
        detectorDist = 2337.126
        axesLimits = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517]
    elif (detectorDistBeamtime == 'dec'):
        detectorDist = 3052.624
        axesLimits = [-2.446074535755995, 2.446074535755995, -2.4460745357559945, 2.4460745357559954]

    sample = sample_model

    alpha_i = angle_of_incidence*ba.deg

    # Beam
    beam = ba.Beam(8e12, wavelength, alpha_i)

    # Detector
    detector = create_detector(detectorDist)
       
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    background = 23
    simulation.options().setIncludeSpecular(True)
    simulation.setBackground(ba.ConstantBackground(background))

    q_dist_y = center_horizontal_slice_value - axesLimits[0]
    q_dist_x = center_vertical_slice_value - axesLimits[2]

    
    q_to_mm_ConversionFactor_y = rayonix_size_y / (axesLimits[1] - axesLimits[0]) 
    q_to_mm_ConversionFactor_x = rayonix_size_x / (axesLimits[3] - axesLimits[2]) 

    
    mm_center_y = q_dist_y * q_to_mm_ConversionFactor_y
    mm_center_x = q_dist_x * q_to_mm_ConversionFactor_x



    mm_y2 = mm_center_y + number_slices * rayonix_pixel_size
    mm_y1 = mm_center_y - number_slices * rayonix_pixel_size
    mm_x2 = mm_center_x + number_slices * rayonix_pixel_size
    mm_x1 = mm_center_x - number_slices * rayonix_pixel_size

    print("mm_y2: " + str(mm_y2))
    print("mm_y1: " + str(mm_y1))
    simulation.detector().setRegionOfInterest(ROI_x1, ROI_y1, ROI_x2, ROI_y2)

    simulation.detector().maskAll()


    #horizontal mask
    simulation.detector().addMask(ba.Rectangle(ROI_x1,mm_y1, ROI_x2, mm_y2), False)

    #vertical mask
    #simulation.detector().addMask(ba.Rectangle(mm_x1, ROI_y1, mm_x2, ROI_y2), False)

    simulation.detector().addMask(ba.Rectangle(148.07181022335135, 140.59419621147006, 152.0184854930185, 177.91172274246256, False))
    
    '''
    for i in range(100,200):
        simulation.detector().addMask(ba.HorizontalLine(i), False)
    '''
    
    return simulation

def get_sampleBasicStructure3_Varying_roughness(P):
    material_PS = ba.RefractiveMaterial("PS", 2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    radius_PS = P["PS_radius"]
    PS_height = P["PS_height"]
    rms_roughness = P["rms_roughness"]
    omega_order = P["sigma_order"]
    paracrystal_coherence = P["paracrystal_coherence"]
    spacing = P["spacing"]

    #------------------Gaussian Distribution of Particle Size-----------------------------
    layout = ba.ParticleLayout()
    distr = ba.DistributionGaussian(radius_PS, 5*nm)
    for parsample in distr.distributionSamples():
        ff_PS = ba.HemiEllipsoid(parsample.value,parsample.value, PS_height) # height_to_radius_ratio * parsample.value)
        particle_PS= ba.Particle(material_PS, ff_PS)
        layout.addParticle(particle_PS, parsample.weight)

    #-----------------Hexagonal Paracrystal for top surface layer ----------------------------
    lattice = ba.HexagonalLattice2D(spacing, 0)
    iff = ba.Interference2DParacrystal(lattice, 0, paracrystal_coherence, paracrystal_coherence) 
    iff.setIntegrationOverXi(True) 
    iff_pdf = ba.Profile2DGauss(omega_order, omega_order, 0) 
    iff.setProbabilityDistributions(iff_pdf, iff_pdf)
    layout.setInterference(iff)

    #-----------------Roughness----------------------------------------------------
    roughness = ba.LayerRoughness(rms_roughness, 0.7, 200*nm)

    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_PS, 215*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_SiO2, 2*nm)
    layer_4 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayerWithTopRoughness(layer_2, roughness)
    sample.addLayer(layer_3)
    sample.addLayer(layer_4)

    return sample

def get_simulation_line_Fitting(P):
    '''
    sample: getSample()
    P : Sample Parameter Variation
    A : simulation parameter (angle, etc.)
    detectorDistBeamtime: detector distance - can be either Dec or Feb
    angle: angle of incidence
    center_horizontal_slice_value: center Qy value that is simulated
    number_slices: number of horizontal slices simulated on either side of center (total slice 1 +  2 * n, where n is this value)
    '''
    ROI_x1 = 152 
    ROI_x2 = 175
    ROI_y1 = 150
    ROI_y2 = 205

    center_horizontal_slice_value = 0.21
    center_vertical_slice_value = 0.1055
    number_slices = 10
    angle = 0.1
    detectorDistBeamtime = 'feb'

    sample = get_sampleBasicStructure3_Varying_roughness(P)

    if (detectorDistBeamtime == 'feb'):
        detectorDist = 2337.126
        axesLimits = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517]
    elif (detectorDistBeamtime == 'dec'):
        detectorDist = 3052.624
        axesLimits = [-2.446074535755995, 2.446074535755995, -2.4460745357559945, 2.4460745357559954]

    beamIntensity = P["intensity"]

    alpha_i = angle*ba.deg

    # Beam
    beam = ba.Beam(beamIntensity, wavelength, alpha_i)

    # Detector
    detector = create_detector(detectorDist)
       
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    background = 0
    # simulation.setBackground(ba.ConstantBackground(background))

    q_dist_y = center_horizontal_slice_value - axesLimits[0]
    q_dist_x = center_vertical_slice_value - axesLimits[2]

    
    q_to_mm_ConversionFactor_y = rayonix_size_y / (axesLimits[1] - axesLimits[0]) 
    q_to_mm_ConversionFactor_x = rayonix_size_x / (axesLimits[3] - axesLimits[2]) 

    
    mm_center_y = q_dist_y * q_to_mm_ConversionFactor_y
    mm_center_x = q_dist_x * q_to_mm_ConversionFactor_x



    mm_y2 = mm_center_y + number_slices * rayonix_pixel_size
    mm_y1 = mm_center_y - number_slices * rayonix_pixel_size
    mm_x2 = mm_center_x + number_slices * rayonix_pixel_size
    mm_x1 = mm_center_x - number_slices * rayonix_pixel_size

    print("mm_y2: " + str(mm_y2))
    print("mm_y1: " + str(mm_y1))
    simulation.detector().setRegionOfInterest(ROI_x1, ROI_y1, ROI_x2, ROI_y2)

    simulation.detector().maskAll()


    #horizontal mask
    simulation.detector().addMask(ba.Rectangle(ROI_x1,mm_y1, ROI_x2, mm_y2), False)

    #vertical mask
    simulation.detector().addMask(ba.Rectangle(mm_x1, ROI_y1, mm_x2, ROI_y2), False)
    
    return simulation



def get_sampleTest():
    material_PS = ba.RefractiveMaterial("PS", 2.51433698E-06, 2.35385822E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 1.656e-06, 1.096e-09)
    material_FA = ba.RefractiveMaterial("FA", 3.90901641E-06, 1.79148728E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04218633E-06, 7.83926453E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.7465490081665e-06, 4.1351946628761e-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)
    
    radius_PS = 48*nm/2 

    radius_PS = 48/2*nm 
    radius_P2VP = 47/2*nm
    PS_thickness = 48*nm
    
    #Core-Shell On Surface
    ff_PS = ba.Sphere(radius_PS) #models PS hemisphere 
    particle_PS = ba.Particle(material_PS, ff_PS)

    layout = ba.ParticleLayout()
    layout.addParticle(particle_PS)

    #Hard-Disk Model
    # Define interference functions
    spacing = 70.95529824561405*nm #mean of diameters
    iff = ba.InterferenceHardDisk(spacing/2*nm,0.0002527*0.65)
    layout.setInterference(iff)
    
    # Define layers
    layer_1 = ba.Layer(material_Vacuum)
    layer_3 = ba.Layer(material_SiO2, 2*nm)
    layer_3.addLayout(layout)
    layer_4 = ba.Layer(material_Si_Sub)

     # Define sample
    sample = ba.MultiLayer()
    sample.addLayer(layer_1)
    sample.addLayer(layer_3)
    sample.addLayer(layer_4)

    return sample
def get_axes_limits(r1, units):
    """
    Returns axes range as expected by pyplot.imshow.
    :param result: SimulationResult object from a Simulation
    :param units: units to use
    :return: axes ranges as a flat list
    """
    limits = []
    for i in range(r1.rank()):
        ami, ama = r1.axisMinMax(i, units)
        assert ami < ama, f'SimulationResult has invalid axis {i}, extending from {ami} to {ama}'
        limits.append(ami)
        limits.append(ama)
    return limits
def real_data(filename, directory):
    """
    Loads experimental data and returns numpy array.
    """
    filepath = os.path.join(directory, filename)
    return ba.readData2D(filepath).npArray()
def find_nearest(array, value):
    """
    Finds the closest value to a specified target in a NumPy array
    """
    idx = (np.abs(array - value)).argmin()
    return idx
def plot_slices(arrayData, axesLimits, horiz_slice = None, vert_slice = None, desiredRange = None, normalize = False):
    """
    arrayData should be in the form data.npArray()
    horizontal slice and vertical slice should be q-space values
    If both horizontal and vertical slices are inputed only result will be vertical slice.
    """
    ysize, xsize = np.shape(arrayData)
    if (vert_slice != None):
        slicename = 'vert'
        xarray = np.linspace(axesLimits[0], axesLimits[1], xsize)
        xindex = find_nearest(xarray, vert_slice)

        global xindex_mm 
        xindex_mm = xindex * rayonix_pixel_size

        if(desiredRange != None):
            y = arrayData[:,xindex]
            y = np.flip(y)
            x = np.linspace(axesLimits[2], axesLimits[3], y.size)
            xindex_low = find_nearest(x, desiredRange[0])
            xindex_high = find_nearest(x, desiredRange[1])
            x = x[xindex_low:xindex_high]
            y = y[xindex_low:xindex_high]
            if(normalize):
                y = normalize_array(y)
        
        else:
            y = arrayData[:,xindex]
            y = np.flip(y)
            if(normalize):
                y = normalize_array(y)
            x = np.linspace(axesLimits[2], axesLimits[3], y.size)
        
        return x , y
        
    elif (horiz_slice != None):
        yarray = np.linspace(axesLimits[2], axesLimits[3], ysize)
        yindex = find_nearest(yarray, horiz_slice)
        global yindex_mm 
        yindex_mm = yindex * rayonix_pixel_size
        yindex = ysize - yindex #CHECK THIS

        if(desiredRange != None):
            y = arrayData[yindex,:]
            x = np.linspace(axesLimits[0], axesLimits[1], y.size)
            xindex_low = find_nearest(x, desiredRange[0])
            xindex_high = find_nearest(x, desiredRange[1])
            x = x[xindex_low:xindex_high]
            y = y[xindex_low:xindex_high]
            if(normalize):
                y = hori_normalize_array(y)

        else:
            y = arrayData[yindex,:]
            if(normalize):
                y = hori_normalize_array(y)
            x = np.linspace(axesLimits[0], axesLimits[1], y.size)
            
        return x , y
def PlotAllTif(directory):
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    for tif_file in tif_files:
        realData_npArray = real_data(tif_file, directory)
        plt.figure()
        hor_slice_q = 0.23
        plot_slices(realData_npArray, realDat_axes_Dec, horiz_slice = hor_slice_q)
        plt.title(tif_file[:25])
        plt.xlim(-1, 1)
    plt.show()
def domainSpacing(Slice_along_Qy, QyArray):
    """
    Only works for real data
    """
    PositiveMax = np.argmax(Slice_along_Qy[2034:]) 
    NegativeMax = np.argmax(Slice_along_Qy[:2034]) 
    Spacing = (QyArray[PositiveMax + 2034] - QyArray[NegativeMax]) / 2

    print("Domain spacing in qspace (1/nm):")
    print(Spacing)
    print("Domain spacing in real (nm):")
    print(2*np.pi/Spacing)
def normalize_array(arr):
    """
    Normalize a numpy array to a range of 0 to 1.
    
    Parameters:
    arr (numpy array): The array to be normalized.
    
    Returns:
    numpy array: The normalized array.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max)
    return normalized_arr
def hori_normalize_array(arr):
    """
    Normalizes a NumPy array such that the average of the first 20
    and the last 20 elements becomes 0, and the highest value becomes 1.

    Parameters:
        arr (numpy.ndarray): Input 1D array.

    Returns:
        numpy.ndarray: The normalized array.
    """
    if len(arr) < 20:
        raise ValueError("Array must have at least 20 elements.")
    
    # Calculate the average of the first 20 and last 20 elements
    avg_first_20 = np.mean(arr[:20])
    avg_last_20 = np.mean(arr[-20:])
    avg_offset = (avg_first_20 + avg_last_20) / 2
    
    # Subtract the offset to make the average zero
    adjusted_arr = arr - avg_offset
    
    # Find the maximum value after adjustment
    max_val = np.max(adjusted_arr)
    if max_val == 0:
        raise ValueError("Maximum value of the adjusted array is zero, cannot normalize.")
    
    # Normalize to make the maximum value 1
    normalized_arr = adjusted_arr / max_val
    
    return normalized_arr
def center_img(img):
    pixelX_cen = 2075.96734114306
    pixelY_cen = 2048 + 2048 - 1915.635837361077
    angle = -1

    h, w = img.shape
    current_center_x, current_center_y = w // 2, h // 2
    
    shift_x = current_center_x - pixelX_cen  
    shift_y = current_center_y - pixelY_cen 

    # Use scipy.ndimage.shift to shift the image
    centered_image = shift(img, shift=(shift_y, shift_x))
    rotated_image = rotate(centered_image, angle, reshape = False)
    return rotated_image
def center_img2(img):
    pixelX_cen = xpos_mm / rayonix_pixel_size
    pixelY_cen = 2048 + 2048 - ypos_mm / rayonix_pixel_size
    angle = 0.4

    h, w = img.shape
    current_center_x, current_center_y = w // 2, h // 2
    
    shift_x = current_center_x - pixelX_cen  
    shift_y = current_center_y - pixelY_cen 

    # Use scipy.ndimage.shift to shift the image
    centered_image = shift(img, shift=(shift_y, shift_x))
    rotated_image = rotate(centered_image, angle, reshape = False)
    return rotated_image
def integrate_plt_slices(start, stop, data, axLim, labelname = '', num=0, horiz_slice = None, vert_slice = None, normalize = False, desiredRange = None):
    x_sum = 0 
    y_sum = 0
    inc = abs(stop - start) / num
    if (horiz_slice != None):
        for i in range(num):
            x, y = plot_slices(data, axesLimits=axLim, horiz_slice = start + inc * i, normalize = normalize, desiredRange=desiredRange)
            x_sum += x 
            y_sum += y 
        x_sum = x_sum / num 
        y_sum = y_sum / num
    elif (vert_slice != None):
        for i in range(num):
            x, y = plot_slices(data, axesLimits=axLim, vert_slice = start + inc * i, normalize = normalize, desiredRange=desiredRange)
            x_sum += x 
            y_sum += y 
        x_sum = x_sum / num
        y_sum = y_sum / num
    return x_sum, y_sum
def lineScan(data, hslice_bot, hslice_top, axesLimits, pixel_inc = 1):
    '''
    axesLimits is typically either realDat_axes_Feb or readDat_axes_Dec
    '''    
    plt.title('Horizontal Slices Along ' + r'$Q_{y}$')

    
    increment = ((abs(axesLimits[2]) + abs(axesLimits[3]))/rayonix_npy ) * pixel_inc
    num = (hslice_top - hslice_bot) / increment
    # Setup the normalization and the colormap
    normalize = mcolors.Normalize(vmin=0, vmax=num)
    colormap = cm.jet

    hslice_top = hslice_bot + increment * num
    normalize_scale = mcolors.Normalize(vmin=hslice_bot, vmax=hslice_top)
    print('number of integrated pixels')
    for i in range(int(num)):
        hslice = hslice_bot + increment * i
        # Assuming plot_slices is a function returning (x, y) arrays
        x, y = plot_slices(data, axesLimits=axesLimits, horiz_slice=hslice,  normalize=False)
        plt.plot(x, y, label=r'$Q_{z}: $' + str(hslice), color=colormap(normalize(i)))

    # Setup the colorbar
    scalarmappable = cm.ScalarMappable(norm=normalize_scale, cmap=colormap)
    scalarmappable.set_array((hslice_bot, hslice_top + increment, increment))

    # Explicitly specify the current axes for the colorbar
    ax = plt.gca()  # Get the current axes
    cbar = plt.colorbar(scalarmappable, ax=ax)

    # Add labels, scales, and show the plot
    plt.ylabel("Intensity")
    plt.xlabel(r'$Q_{y} \;(1/{\rm nm})$')
    plt.yscale('log')
    plt.xscale('log')
    # Set the label on top
    cbar.set_label(r'$Q_{z}:$', loc='top', labelpad = -50, rotation=0) 

    # Adjust the label position slightly higher
    label = cbar.ax.xaxis.label
    label.set_position((0.5, 1.2))
    plt.plot()