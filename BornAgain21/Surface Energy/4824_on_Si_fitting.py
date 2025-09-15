from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
from GISAXS_Analysis.materials import get_materials
from GISAXS_Analysis import height_radius_from_lineprofiles as h_r
import bornagain as ba
from bornagain import nm, deg
import numpy as np
import os

def get_interference_radial(surface_layout, spacing, coherence_length, omega_order):
    iff = ba.InterferenceRadialParacrystal(spacing, coherence_length)
    iff_pdf = ba.Profile1DGauss(omega_order)
    iff.setProbabilityDistribution(iff_pdf)
    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(0.0003) #PLAY WITH THIS 0.0265
    #iff.setKappa(1.5) #size-distribution model

def get_sample():
    materials = get_materials()
    material_PS=materials['PS']
    material_P2VP=materials['P2VP']
    material_Si_Sub=materials['Si_Sub']
    material_SiO2=materials['SiO2']
    material_Vacuum=materials['Vacuum']

    #Particle size with dispersion
    PS_h = 16.21 #mu_h
    PS_d = 47.63 #mu_d
    sigma_h = 1.14
    sigma_d = 4
    N = 1000
    rho = 0.518
    num_samples = 100
    height, diameter = h_r.sample_height_diameter(PS_h, PS_d, sigma_h, sigma_d, rho, N)
    PS_diameters, PS_heights, weight_Ks, labels = h_r.summarize_pairs_kmedoids(diameter, height, K=num_samples, scale=True)
    h_r.visualize_kmedoids(diameter, height, PS_diameters, PS_heights, labels, weight_rep=weight_Ks)
    graphing.plt.show()
    surface_layout = ba.ParticleLayout()
    for PS_diameter, PS_height, weight_K in zip(PS_diameters, PS_heights, weight_Ks):
        ff_PS = ba.HemiEllipsoid(PS_diameter/2 * nm, PS_diameter/2 * nm, PS_height * nm)
        #ff_PS = ba.Sphere(PS_diameter*nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K)

    # Radial Interference Functions
    spacing = 58*nm
    coherence_length = 1000*nm
    omega_order = 4*nm

    # 2D Paracrystal hexagonal inteference function
    lattice = ba.BasicLattice2D(spacing, spacing, 120*deg, 0*deg) 
    iff = ba.Interference2DParacrystal(lattice, 0, coherence_length, coherence_length)
    iff.setIntegrationOverXi(True)
    iff_pdf = ba.Profile2DGauss(omega_order, omega_order, 0)
    iff.setProbabilityDistributions(iff_pdf, iff_pdf)
    surface_layout.setInterference(iff)

    #Add Layers
    PS_brush_thickness = 2*nm

    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_brush = ba.Layer(material_PS, PS_brush_thickness)
    layer_PS_brush.addLayout(surface_layout)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

     # Define sample 
    sample = ba.MultiLayer()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_brush)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

def main():         
    
    exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    
    exp_filename = '4824_3gPL_2000RPM_0p1Deg.npz'
    exp2D, exp_axes = g.load_npz_data(exp_filename, exp_data_directory)
    region = [150, 150, 200, 300]
    simulation_line = g.get_simulation_line(sample_model=get_sample(), 
                                            detectorDistBeamtime='dec', 
                                            angle_of_incidence= 0.1, 
                                            center_horizontal_slice_values=[0.2], 
                                            center_vertical_slice_values= [0],
                                            number_slices=10,beamIntensity=1e11)

    #simulation_2D = g.get_simulation_2D(sample_model=get_sample(), detectorDistBeamtime= 'dec', angle = 0.1, beamIntensity = 1e11,ROI=region)
    result = simulation_line.simulate()
    simul_dat = result.extracted_field()
    final_array = simul_dat.npArray()
    save_filename = "monolayer_test.npz"
    simulationDataAxes = g.get_axes_limits(result, ba.Coords_QSPACE)
    save_sim_directory = r'C:\BornAgainSimulations\data\sim-npz'
    
    g.save_npz_data(os.path.join(save_sim_directory, save_filename), final_array, simulationDataAxes)
    sim2D, simAxes, params = g.load_npz_data(save_filename, save_sim_directory, return_date=False, return_params=True)

    linecut1 = 0.2
    linecut2 = 0.1

    graphing.plot2D(realData=exp2D, 
                simulationData=sim2D, 
                realDat_axes=exp_axes, 
                simData_axes=simAxes, 
                zlim=[22,50000])
    graphing.linecutsItoV(simulation_data=sim2D, 
                      experimental_data=exp2D, 
                      #L2_qy=linecut2,
                      L1_qz=linecut1, 
                      L2_qy=linecut2,
                      #L5_qz=linecut5, 
                      axes_exp=exp_axes, 
                      axes_sim=simAxes)
    graphing.plt.show()
def test():
    exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    exp_filename = '4824_3gPL_2000RPM_0p1Deg.npz'
    g.graph_experiment_detectorSpace(exp_filename, exp_data_directory, detectorDistBeamtime = 'dec', angle = 0.1)
    graphing.plt.show()
main()