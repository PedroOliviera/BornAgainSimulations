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
from bornagain import R3

def get_sample_spheres(P):

    omega_order = P["omega_order"]
    surface_density = P["surface_density"]
    dampening_length = P["dampening_length"]
    PS_delta = P["PS_delta"]

    r_P2VP = 20*nm
    h_P2VP = 7*nm

    material_PS = ba.RefractiveMaterial("PS", PS_delta, 2.46904652E-09)
    #material_P2VP = ba.RefractiveMaterial("P2VP", 3E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 63*nm

    # Define layer thickness
    thickness = 214.8*nm
    mid_layer_thickness = spacing
    num_layers = int(thickness/mid_layer_thickness)
    remainder_thickness = thickness - num_layers * mid_layer_thickness
    top_layer_thickness, bot_layer_thickness = remainder_thickness / 2 , remainder_thickness / 2

    #########################################----BURIED PARTICLES----################################################
    buried_layout = ba.ParticleLayout()
    ff_P2VP = ba.Sphere(r_P2VP) 
    particle = ba.Particle(material_P2VP, ff_P2VP)
    particle.translate(0,0,-mid_layer_thickness)
    buried_layout.addParticle(particle, 1)
    #iff.setPositionVariance(2*nm)
    buried_layout.setInterference(iff)
    buried_layout.setTotalParticleSurfaceDensity(0.0003)

    #----------------Roughness---------------------------------------------
    #----------------PS----------------------------------------------------
    hurst = 0.49
    corr = 84*nm
    sig = 3.2*nm
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
    
    #Add Layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, top_layer_thickness)
    layer_PS_Top.addLayout(surface_layout)
    layer_PS_Mid = ba.Layer(material_PS, mid_layer_thickness)
    layer_PS_Mid.addLayout(buried_layout)
    layer_PS_Bot = ba.Layer(material_PS, bot_layer_thickness)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    for i in range(num_layers):
        sample.addLayer(layer_PS_Mid)
    sample.addLayer(layer_PS_Bot)
    sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)
    
    return sample

def get_sample_spheroids(P):

    omega_order = P["omega_order"]
    surface_density = P["surface_density"]
    dampening_length = P["dampening_length"]
    PS_delta = P["PS_delta"]

    r_P2VP = 20*nm
    h_P2VP = 7*nm

    material_PS = ba.RefractiveMaterial("PS", PS_delta, 2.46904652E-09)
    #material_P2VP = ba.RefractiveMaterial("P2VP", 3E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 63*nm

    # Define layer thickness
    thickness = 214.8*nm
    mid_layer_thickness = spacing
    num_layers = int(thickness/mid_layer_thickness)
    remainder_thickness = thickness - num_layers * mid_layer_thickness
    top_layer_thickness, bot_layer_thickness = remainder_thickness / 2 , remainder_thickness / 2

    #########################################----BURIED PARTICLES----################################################
    buried_layout = ba.ParticleLayout()
    ff_P2VP = ba.Spheroid(r_P2VP, h_P2VP) 
    particle = ba.Particle(material_P2VP, ff_P2VP)
    particle.translate(0,0,-mid_layer_thickness)
    buried_layout.addParticle(particle, 1)
    #iff.setPositionVariance(2*nm)
    buried_layout.setInterference(iff)
    buried_layout.setTotalParticleSurfaceDensity(0.0003)

    #----------------Roughness---------------------------------------------
    #----------------PS----------------------------------------------------
    hurst = 0.49
    corr = 84*nm
    sig = 3.2*nm
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
    
    #Add Layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, top_layer_thickness)
    layer_PS_Top.addLayout(surface_layout)
    layer_PS_Mid = ba.Layer(material_PS, mid_layer_thickness)
    layer_PS_Mid.addLayout(buried_layout)
    layer_PS_Bot = ba.Layer(material_PS, bot_layer_thickness)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    for i in range(num_layers):
        sample.addLayer(layer_PS_Mid)
    sample.addLayer(layer_PS_Bot)
    sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)
    
    return sample

def get_sample_spheresAndSpheroidsWithInterference(P):
    omega_order = P["omega_order"]
    surface_density = P["surface_density"]
    dampening_length = P["dampening_length"]
    PS_delta = P["PS_delta"]

    r_P2VP = 20*nm
    h_P2VP = 7*nm

    material_PS = ba.RefractiveMaterial("PS", PS_delta, 2.46904652E-09)
    #material_P2VP = ba.RefractiveMaterial("P2VP", 3E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 63*nm

    # Define layer thickness
    thickness = 214.8*nm
    mid_layer_thickness = spacing
    num_layers = int(thickness/mid_layer_thickness)
    remainder_thickness = thickness - num_layers * mid_layer_thickness
    top_layer_thickness, bot_layer_thickness = remainder_thickness / 2 , remainder_thickness / 2

    #########################################----BURIED PARTICLES----################################################
    buried_layout = ba.ParticleLayout()
    ff_P2VP = ba.Spheroid(r_P2VP, h_P2VP) 
    particle = ba.Particle(material_P2VP, ff_P2VP)
    particle.translate(0,0,-mid_layer_thickness)
    buried_layout.addParticle(particle, 1)
    #iff.setPositionVariance(2*nm)
    buried_layout.setInterference(iff)
    buried_layout.setTotalParticleSurfaceDensity(0.0003)

    #----------------Roughness---------------------------------------------
    #----------------PS----------------------------------------------------
    hurst = 0.49
    corr = 84*nm
    sig = 3.2*nm
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
    
    #Add Layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, top_layer_thickness)
    layer_PS_Top.addLayout(surface_layout)
    layer_PS_Mid = ba.Layer(material_PS, mid_layer_thickness)
    layer_PS_Mid.addLayout(buried_layout)
    layer_PS_Bot = ba.Layer(material_PS, bot_layer_thickness)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    for i in range(num_layers):
        sample.addLayer(layer_PS_Mid)
    sample.addLayer(layer_PS_Bot)
    sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)
    
    return sample

def get_sample_spheresAndSpheroids(P):
    omega_order = P["omega_order"]
    surface_density = P["surface_density"]
    dampening_length = P["dampening_length"]
    PS_delta = P["PS_delta"]

    r_P2VP = 20*nm
    h_P2VP = 7*nm

    material_PS = ba.RefractiveMaterial("PS", PS_delta, 2.46904652E-09)
    #material_P2VP = ba.RefractiveMaterial("P2VP", 3E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    spacing = 63*nm

    # Define layer thickness
    thickness = 214.8*nm
    mid_layer_thickness = spacing
    num_layers = int(thickness/mid_layer_thickness)
    remainder_thickness = thickness - num_layers * mid_layer_thickness
    top_layer_thickness, bot_layer_thickness = remainder_thickness / 2 , remainder_thickness / 2

    #########################################----BURIED PARTICLES----################################################
    buried_layout = ba.ParticleLayout()
    ff_P2VP = ba.Spheroid(r_P2VP, h_P2VP) 
    particle = ba.Particle(material_P2VP, ff_P2VP)
    particle.translate(0,0,-mid_layer_thickness)
    buried_layout.addParticle(particle, 1)
    #iff.setPositionVariance(2*nm)
    buried_layout.setInterference(iff)
    buried_layout.setTotalParticleSurfaceDensity(0.0003)

    #----------------Roughness---------------------------------------------
    #----------------PS----------------------------------------------------
    hurst = 0.49
    corr = 84*nm
    sig = 3.2*nm
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
    
    #Add Layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, top_layer_thickness)
    layer_PS_Top.addLayout(surface_layout)
    layer_PS_Mid = ba.Layer(material_PS, mid_layer_thickness)
    layer_PS_Mid.addLayout(buried_layout)
    layer_PS_Bot = ba.Layer(material_PS, bot_layer_thickness)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    for i in range(num_layers):
        sample.addLayer(layer_PS_Mid)
    sample.addLayer(layer_PS_Bot)
    sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)
    
    return sample

def main():
    #For Feb Data
    realDat_axes_Feb = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517]

    directory1 = r'C:\Users\Pedro\Data Transfer\Sample_35_3secIntegration'
    filename1 = 'N3.tif'
    realData_npArray = g.real_data(filename1, directory1)
    realData_npArray = g.center_img(realData_npArray)

    region = [150, 155, 212, 212]
    number_of_samples = 10
    radius_P2VP = [18,19,20,21,22,23,24]
    #simulation_2D = g.get_simulation_2D(sample_model=get_sample(number_of_samples), detectorDistBeamtime= 'feb', angle = 0.13, beamIntensity = 8e12,ROI=region,oneThread=False)
    for radius in radius_P2VP:
        simulation_line = g.get_simulation_line(sample_model=get_sample(number_of_samples,r_P2VP=radius), detectorDistBeamtime='feb', angle_of_incidence= 0.13, center_horizontal_slice_value=0.22, center_vertical_slice_value= 0, number_slices=5,oneThread=False)
        #simulation_2D = g.get_simulation_2D(sample_model=get_sample(number_of_samples,r_P2VP=radius), detectorDistBeamtime= 'feb', angle = 0.13, beamIntensity = 8e12,ROI=region,oneThread=False)
        result = simulation_line.simulate()
        simul_dat = result.extracted_field()
        final_array = simul_dat.npArray()
        save_filename = "tests_13deg_line_CosineRippleGauss_P2VPradius_2D_"+ str(radius) +"nm.npy"
        np.save(save_filename, final_array)

    simulationDataAxes = g.get_axes_limits(result, ba.Coords_QSPACE)
    legendLabels = ["P2VP radius: " + str(radius) for radius in radius_P2VP]
    data2D = []
    for radius in radius_P2VP:
        data2D.append(np.load("tests_13deg_line_CosineRippleGauss_P2VPradius_2D_"+ str(radius) +"nm.npy"))

    #graphing.plot2D(simulationData=data2D, simData_axes=simulationDataAxes, realData=realData_npArray, realDat_axes=realDat_axes_Feb, zlim=[22,70000000])
    vert_slice_q = 0.2
    graphing.yonedaPlot(vert_slice_q, data2D, data_axes=simulationDataAxes, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb, xmin = 0.08, xmax = 0.2, labels=legendLabels) #, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb)
    plt.show()
def testProfiles():
    # Minimal test — adjust file path as needed
    lineprofile_dir = r"C:\Users\Pedro\Data Transfer\Lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"
    lineprofile_dir = r"C:\Users\Pedro\Data Transfer\Lineprofiles\lineProfiles_34_Big_OnePerParticle_2.txt"
    x_cols, y_cols = h_r.load_lineprofiles(lineprofile_dir)
    figs, pages = h_r.plot_profiles_in_pages(
    x_cols, y_cols,
    total_plots=15,
    frac = 0.1,
    per_fig=3,
    select="random",
    seed=123
    )
    plt.show()

def testK_medoidsSummary():
    # Minimal test — adjust file path as needed
    num_samples = 10
    lineprofile_dir = r"C:\Users\Pedro\Data Transfer\Lineprofiles\lineProfiles_34_Big_OnePerParticle_2.txt"

    Factor = 3.1
    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    h_r.visualize_kmedoids(dmin_nm/Factor, hsub_nm, diam_K/Factor, height_K, labels, weight_rep=weight_K)

def GraphingOnly():
    data2D = np.load("tests_10deg_line.npy")

    graphing.plot2D(simulationData=data2D, simData_axes=simulationDataAxes, realData=realData_npArray, realDat_axes=realDat_axes_Feb, zlim=[22,70000000])
    vert_slice_q = 0.2
    graphing.yonedaPlot(vert_slice_q, [data2D], data_axes=simulationDataAxes, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb) #, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb)


main()