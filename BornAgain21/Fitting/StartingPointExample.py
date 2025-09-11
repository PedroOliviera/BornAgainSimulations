import GISAXS_setup_v21 as g
import Graphing_Analysis as graphing
from bornagain import ba_plot as bp, deg, angstrom, nm
import bornagain as ba
from bornagain import nm, deg
import matplotlib.pyplot as plt
import height_radius_from_lineprofiles as h_r
import numpy as np
import cmath
from scipy.special import j0  # Bessel J0
from scipy.integrate import quad
import math

def get_sample_step2(P):
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    num_samples = 3
    PS_sig = P["sigma"] 
    PS_zeta = P["zeta"]

    #----------------Roughness---------------------------------------------
    #----------------PS----------------------------------------------------
    hurst = 0.6
    corr = PS_zeta
    sig = PS_sig
    roughness_PS = ba.LayerRoughness(sig, hurst, corr)

    #----------------SiO2---------------------------------------------------
    hurst = 0.52
    corr = 10*nm
    sig = 0.2*nm
    roughness_SiO2 = ba.LayerRoughness(sig, hurst, corr)

    # Define layers
    thickness = 214.8*nm

    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, thickness)
    #layer_PS_Top.addLayout(surface_layout)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)


    return sample

def get_sample_step3(P):

    omega_order = P["omega_order"]
    surface_density = P["surface_density"]
    dampening_length = P["dampening_length"]
    PS_delta = P["PS_delta"]

    material_PS = ba.RefractiveMaterial("PS", PS_delta, 2.46904652E-09)
    #material_P2VP = ba.RefractiveMaterial("P2VP", 3E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 63*nm
    num_samples = 10

    # Minimal test — adjust file path as needed
    lineprofile_dir =  r"C:\Users\Pedro\Data Transfer\Lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    
    
    #########################################----SURFACE PARTICLES----################################################
    surface_layout = ba.ParticleLayout()
    for i in range(num_samples):
        ff_PS = ba.CosineRippleGauss((diam_K[i]) * nm, (diam_K[i]) * nm, height_K[i] * nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])
    
    # Radial Interference Functions
    iff = ba.InterferenceRadialParacrystal(spacing, dampening_length)
    iff_pdf = ba.Profile1DGauss(omega_order)
    
    iff.setProbabilityDistribution(iff_pdf)
    
    iff.setKappa(1.5) #size-distribution model 

    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(surface_density)

    #----------------Roughness---------------------------------------------
    #----------------PS----------------------------------------------------
    hurst = 0.6
    corr = 331*nm
    sig = 1.098e0
    roughness_PS = ba.LayerRoughness(sig, hurst, corr)
    #hurst = 0.7
    #corr = 200*nm
    #sig = 6.795*nm
    #roughness_PS = ba.LayerRoughness(sig, hurst, corr)

    #----------------SiO2---------------------------------------------------
    hurst = 0.52
    corr = 10*nm
    sig = 0.2*nm
    roughness_SiO2 = ba.LayerRoughness(sig, hurst, corr)

    # Define layers
    thickness = 214.8*nm

    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, thickness)
    layer_PS_Top.addLayout(surface_layout)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)

    return sample

def main():

    region = [150, 155, 212, 212]
    #region = [0,0,300,300]
    number_of_samples = 3
    i = 0
    sim2D, simAxes, params = g.loadSim("step3_genetic_run1" + str(i) + ".npz", return_date=False, return_params=True)

    '''
    simulation_line = g.get_simulation_line_step2(sample_model=get_sample(number_of_samples), 
                                            detectorDistBeamtime='feb', 
                                            angle_of_incidence= 0.13, 
                                            center_horizontal_slice_values=horizontal_slices, 
                                            center_vertical_slice_values= vertical_slices, 
                                            vertical_bounds= [0.777, 3.19],
                                            horizontal_bounds=[0,4],
                                            number_slices=1,beamIntensity=2.689e+13)
    '''
    #'''
    simulation_2D = g.get_simulation_2D(sample_model=get_sample_step3(params), 
                                        detectorDistBeamtime= 'feb',
                                         angle = 0.13, 
                                         beamIntensity = 2.689e+13,
                                         ROI=region,
                                         oneThread=False)
    #'''
    result = simulation_2D.simulate()
    simul_dat = result.extracted_field()
    final_array = simul_dat.npArray()
    save_filename = "step3_genetic_run1" + str(i) + "_FullSim.npz"
    simulationDataAxes = g.get_axes_limits(result, ba.Coords_QSPACE)

    g.saveSim(save_filename, final_array, simulationDataAxes)
    print("DONE")

main()