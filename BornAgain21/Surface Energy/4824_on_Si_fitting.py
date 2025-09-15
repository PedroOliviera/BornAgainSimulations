from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
from GISAXS_Analysis.materials import get_materials
from GISAXS_Analysis import height_radius_from_lineprofiles as h_r
import bornagain as ba
from bornagain import nm, deg
import numpy as np
import os
from scipy.special import j0  # Bessel J0
from scipy.integrate import quad
import math
from bornagain import ba_plot as bp
class CustomFormFactor(ba.IFormFactor):
    """
    A custom defined form factor.
    The particle is a prism of height H,
    with a base in form of a Greek cross ("plus" sign) with side length L.
    """
    def __init__(self, R, H, *, epsabs=1e-9, epsrel=1e-7, limit=200, qz_eps=1e-12):
        ba.IFormFactor.__init__(self)
        self.R = float(R)
        self.H = float(H)
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit
        self.qz_eps = qz_eps
        self._two_pi = 2.0 * math.pi
        self._pi_over_R = math.pi / self.R

    def clone(self):
        """
        IMPORTANT NOTE:
        The clone method needs to call transferToCPP() on the cloned object
        to transfer the ownership of the clone to the cpp code
        """
        cloned_ff = CustomFormFactor(self.R, self.H)
        cloned_ff.transferToCPP()
        return cloned_ff
    
    def _z_of_r(self, r: float) -> float:
        return 0.5 * self.H * (1.0 + math.cos(self._pi_over_R * r))

    def formfactor(self, q):
        # --- real inputs only ---
        qx, qy, qz = q.x().real, q.y().real, q.z().real
        qpar = math.hypot(qx, qy)
        R = self.R

        # qz -> 0 branch: F(Q⊥,0) = 2π ∫ r J0(Q⊥ r) z(r) dr  (purely real)
        if abs(qz) < self.qz_eps:
            def g_r(r: float) -> float:
                return r * j0(qpar * r) * self._z_of_r(r)
            I = quad(g_r, 0.0, R, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)[0]
            return self._two_pi * I

        # general real-qz case:
        # Let A = ∫ r J0(Q⊥ r) [cos(qz z(r)) - 1] dr
        #     B = ∫ r J0(Q⊥ r)  sin(qz z(r))        dr
        # Then F = (2π/(i qz)) (A + i B)  =>  Re(F)= 2π * B/qz,  Im(F)= -2π * A/qz
        def g_A(r: float) -> float:
            zr = self._z_of_r(r)
            return r * j0(qpar * r) * (math.cos(qz * zr) - 1.0)

        def g_B(r: float) -> float:
            zr = self._z_of_r(r)
            return r * j0(qpar * r) * math.sin(qz * zr)

        A = quad(g_A, 0.0, R, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)[0]
        B = quad(g_B, 0.0, R, epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit)[0]

        ReF = (self._two_pi / qz) * B
        ImF = -(self._two_pi / qz) * A
        return complex(ReF, ImF)
    
    def radialExtension(self):
        # axisymmetric cap of base radius R ⇒ radial extent is R
        return self.R

    def spanZ(self, rotation):
        return ba.Span(0, self.H)

def get_interference_radial(surface_layout, spacing, coherence_length, omega_order):
    iff = ba.InterferenceRadialParacrystal(spacing, coherence_length)
    iff_pdf = ba.Profile1DGauss(omega_order)
    iff.setProbabilityDistribution(iff_pdf)
    #iff.setKappa(1) #size-distribution model
    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(0.0003) #PLAY WITH THIS 0.0265
    
def get_interference_2D_para(surface_layout, spacing, coherence_length, omega_order):
    lattice = ba.BasicLattice2D(spacing, spacing, 120*deg, 0*deg) 
    iff = ba.Interference2DParacrystal(lattice, 0, coherence_length, coherence_length)
    #iff.setIntegrationOverXi(True)
    iff_pdf = ba.Profile2DGauss(omega_order, omega_order, 0)
    iff.setProbabilityDistributions(iff_pdf, iff_pdf)
    #iff.setPositionVariance(5*nm)
    surface_layout.setInterference(iff)

def get_interference_hardDisk(surface_layout,radius, density):
    iff = ba.InterferenceHardDisk(radius, density)
    surface_layout.setInterference(iff)

def get_sample():
    materials = get_materials()
    material_PS=materials['PS']
    material_P2VP=materials['P2VP']
    material_Si_Sub=materials['Si_Sub']
    material_SiO2=materials['SiO2']
    material_Vacuum=materials['Vacuum']

    #Particle size with dispersion
    PS_h = 15 #mu_h
    PS_d = 58#58 #mu_d
    sigma_h = 1.14
    sigma_d = 4
    N = 1000
    rho = 0.518
    num_samples = 1
    height, diameter = h_r.sample_height_diameter(PS_h, PS_d, sigma_h, sigma_d, rho, N)
    PS_diameters, PS_heights, weight_Ks, labels = h_r.summarize_pairs_kmedoids(diameter, height, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(diameter, height, PS_diameters, PS_heights, labels, weight_rep=weight_Ks)
    #graphing.plt.show()
    #surface_layout1 = ba.ParticleLayout()
    surface_layout2 = ba.ParticleLayout()
    for PS_diameter, PS_height, weight_K in zip(PS_diameters, PS_heights, weight_Ks):
        #ff_PS = ba.HemiEllipsoid(PS_diameter/2 * nm, PS_diameter/2 * nm, PS_height * nm)
        #ff_PS = CustomFormFactor(PS_diameter/2*nm, PS_height*nm)
        #ff_PS = ba.Sphere(PS_diameter*nm)
        ff_PS = ba.Pyramid4(PS_diameter * nm, PS_height * nm, 37*deg)
        particle_PS= ba.Particle(material_PS, ff_PS)
        #surface_layout1.addParticle(particle_PS, weight_K)
        surface_layout2.addParticle(particle_PS, weight_K)

    # Interference Function
    #spacing1 = 52*nm
    spacing2 = 104*nm
    coherence_length = 1000*nm
    omega_order = 12*nm
    #get_interference_2D_para(surface_layout2, spacing2, coherence_length, omega_order)
    #get_interference_radial(surface_layout1, spacing1, coherence_length, omega_order)
    get_interference_radial(surface_layout2, spacing2, coherence_length, omega_order)
    #Add Layers
    PS_brush_thickness = 2*nm

    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_brush = ba.Layer(material_PS, PS_brush_thickness)
    #layer_PS_brush.addLayout(surface_layout1)
    layer_PS_brush.addLayout(surface_layout2)
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
    
    exp_data_directory = r'C:\Users\Pedro\BornAgainSimulations\data\exp-npz'
    
    exp_filename = '4824_3gPL_2000RPM_0p1Deg.npz'
    exp2D, exp_axes = g.load_npz_data(exp_filename, exp_data_directory)
    region = [150, 150, 200, 300]
    #simulation_line = g.get_simulation_line(sample_model=get_sample(), 
    #                                        detectorDistBeamtime='dec', 
    #                                        angle_of_incidence= 0.1, 
    #                                        center_horizontal_slice_values=None, 
    #                                        center_vertical_slice_values= [0.125],
    #                                        number_slices=10,beamIntensity=1e13)

    simulation_2D = g.get_simulation_2D(sample_model=get_sample(), detectorDistBeamtime= 'dec', angle = 0.1, beamIntensity = 1e13,ROI=region)
    result = simulation_2D.simulate()
    simul_dat = result.extracted_field()
    final_array = simul_dat.npArray()
    
    save_filename = "monolayer_test2.npz"
    simulationDataAxes = g.get_axes_limits(result, ba.Coords_QSPACE)
    save_sim_directory = r'C:\Users\Pedro\BornAgainSimulations\data\sim-npz'
    
    g.save_npz_data(os.path.join(save_sim_directory, save_filename), final_array, simulationDataAxes)
    sim2D, simAxes, params = g.load_npz_data(save_filename, save_sim_directory, return_date=False, return_params=True)

    #linecut1 = 0.2
    linecut2 = 0.125

    graphing.plot2D(realData=exp2D, 
                simulationData=sim2D, 
                realDat_axes=exp_axes, 
                simData_axes=simAxes, 
                zlim=[22,50000])
    graphing.linecutsItoV(simulation_data=sim2D, 
                      experimental_data=exp2D, 
                      #L2_qy=linecut2,
                      #L1_qz=linecut1, 
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