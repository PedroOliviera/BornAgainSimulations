import GISAXS_setup_v21 as g
import Graphing_Analysis as graphing
from bornagain import ba_plot as bp
import bornagain as ba
from bornagain import nm, deg
import matplotlib.pyplot as plt
import height_radius_from_lineprofiles as h_r
import numpy as np

def get_sample(num_samples):
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    omega_order = 5*nm
    spacing = 67*nm

    # Minimal test — adjust file path as needed
    lineprofile_dir = r"C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Characterization\AFM\2025\06-26-2025\lineProfiles_35_Big_OnePerParticle.txt"

    Factor = 1
    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.2)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    h_r.visualize_kmedoids(dmin_nm/Factor, hsub_nm, diam_K/Factor, height_K, labels, weight_rep=weight_K)
    
    #########################################----SURFACE PARTICLES----################################################
    surface_layout = ba.ParticleLayout()
    for i in range(num_samples):
        ff_PS = ba.HemiEllipsoid((diam_K[i]/Factor) * nm, (diam_K[i]/Factor) * nm, height_K[i] * nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])
    
    # Interference Functions
    iff = ba.InterferenceRadialParacrystal(spacing, 400*nm)
    iff_pdf = ba.Profile1DGauss(omega_order)
    
    iff.setProbabilityDistribution(iff_pdf)
    
    iff.setKappa(1.5) #size-distribution model 

    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(0.00005825) #PLAY WITH THIS
    

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

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, 214.8*nm)
    layer_PS_Top.addLayout(surface_layout)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_Top)
    sample.addLayer(layer_SiO2)
    #sample.addLayerWithTopRoughness(layer_PS_Top, roughness_PS)
    #sample.addLayerWithTopRoughness(layer_SiO2, roughness_SiO2)
    sample.addLayer(layer_Si)


    return sample

def main():
    #For Feb Data
    realDat_axes_Feb = [-3.1895200744655168, 3.1895200744655168, -3.1895200744655163, 3.189520074465517]

    directory1 = r'C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\GISAXS Analysis\Data\GISAS\35'
    filename1 = '35_2000RPM_40mgPml_polymer_0.13.tif'
    realData_npArray = g.real_data(filename1, directory1)
    realData_npArray = g.center_img2(realData_npArray)

    region = [150, 155, 212, 212]
    number_of_samples = 10#50
    simulation_2D = g.get_simulation_2D(sample_model=get_sample(number_of_samples), detectorDistBeamtime= 'feb', angle = 0.13, beamIntensity = 8e12,ROI=region)
    #simulation_line = g.get_simulation_line(sample_model=get_sample(number_of_samples), detectorDistBeamtime='feb', angle_of_incidence= 0.13, center_horizontal_slice_value=0.22, center_vertical_slice_value= 0, number_slices=5)
    #simulation_2D = g2.get_simulation_2D(sample_model=g2.get_sampleTest(), detectorDistBeamtime= 'feb', angle = 0.1, beamIntensity = 8e12, ROI= region)
    result = simulation_2D.simulate()
    simul_dat = result.extracted_field()
    final_array = simul_dat.npArray()
    save_filename = "tests_10deg_line.npy"
    np.save(save_filename, final_array)

    simulationDataAxes = g.get_axes_limits(result, ba.Coords_QSPACE)
    data2D = np.load("tests_10deg_line.npy")

    graphing.plot2D(simulationData=data2D, simData_axes=simulationDataAxes, realData=realData_npArray, realDat_axes=realDat_axes_Feb, zlim=[22,70000000])
    vert_slice_q = 0.1
    graphing.yonedaPlot(vert_slice_q, [data2D], data_axes=simulationDataAxes, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb) #, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb)

def testProfiles():
    # Minimal test — adjust file path as needed
    lineprofile_dir = r"C:\New Sims\New Sims\lineProfiles_34_Big_OnePerParticle_2.txt"
    x_cols, y_cols = h_r.load_lineprofiles(lineprofile_dir)
    figs, pages = h_r.plot_profiles_in_pages(
    x_cols, y_cols,
    total_plots=15,
    frac = 0.8,
    per_fig=3,
    select="random",
    seed=123
    )
    plt.show()
main()