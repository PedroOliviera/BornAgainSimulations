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

def get_sample(num_samples, r_P2VP):
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    omega_order = 4*nm #9nm
    spacing = 54*nm
    
    # Minimal test — adjust file path as needed
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    Factor = 2
    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(dmin_nm/Factor, hsub_nm, diam_K/Factor, height_K, labels, weight_rep=weight_K)
    
    #########################################----SURFACE PARTICLES----################################################
    surface_layout = ba.ParticleLayout()
    for i in range(num_samples):
        ff_PS = ba.CosineRippleGauss((diam_K[i]) * nm, (diam_K[i]) * nm, height_K[i] * nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])
    
    # Radial Interference Functions
    iff = ba.InterferenceRadialParacrystal(spacing, 450*nm)
    iff_pdf = ba.Profile1DGauss(omega_order)
    
    iff.setProbabilityDistribution(iff_pdf)
    
    #iff.setKappa(1.5) #size-distribution model 

    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(0.0003) #PLAY WITH THIS 0.0265

    #########################################----BURIED PARTICLES----################################################
    buried_layout = ba.ParticleLayout()
    polyDispersity_P2VP = r_P2VP*0.1
    distr = ba.DistributionGaussian(r_P2VP*nm, polyDispersity_P2VP*nm)
    #for P2VP_radius in distr.distributionSamples():
    #    ff_P2VP = ba.Sphere(P2VP_radius.value)
    #    particle = ba.Particle(material_P2VP, ff_P2VP)
    #    buried_layout.addParticle(particle, P2VP_radius.weight)
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

    # Define layers
    thickness = 214.8*nm
    mid_layer_thickness = spacing
    num_layers = int(thickness/spacing)
    remainder_thickness = thickness - num_layers * mid_layer_thickness
    top_layer_thickness, bot_layer_thickness = remainder_thickness / 2 , remainder_thickness / 2

    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, top_layer_thickness)
    #layer_PS_Top.addLayout(surface_layout)
    layer_PS_Mid = ba.Layer(material_PS, mid_layer_thickness)
    #layer_PS_Mid.addLayout(buried_layout)
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

    directory1 = r'C:\BornAgainSimulations\data\tif'
    filename1 = '35_2000RPM_40mgPml_polymer_0p10.tif'
    realData_npArray = g.real_data(filename1, directory1)
    realData_npArray = g.center_img(realData_npArray)

    region = [150, 155, 212, 212]
    number_of_samples = 1
    radius_P2VP = [24]
    #simulation_2D = g.get_simulation_2D(sample_model=get_sample(number_of_samples), detectorDistBeamtime= 'feb', angle = 0.13, beamIntensity = 8e12,ROI=region,oneThread=False)
    for radius in radius_P2VP:
        #simulation_line = g.get_simulation_line(sample_model=get_sample(number_of_samples,r_P2VP=radius), detectorDistBeamtime='feb', angle_of_incidence= 0.13, center_horizontal_slice_value=0.22, center_vertical_slice_value= 0, number_slices=5,oneThread=False)
        simulation_2D = g.get_simulation_2D(sample_model=get_sample(number_of_samples,r_P2VP=radius), detectorDistBeamtime= 'feb', angle = 0.1, beamIntensity = 8e12,ROI=region,oneThread=False)
        result = simulation_2D.simulate()
        simul_dat = result.extracted_field()
        final_array = simul_dat.npArray()
        save_filename = "tests_13deg_line_CosineRippleGauss_P2VPradius_2D_"+ str(radius) +"nm.npy"
        np.save(save_filename, final_array)

    simulationDataAxes = g.get_axes_limits(result, ba.Coords_QSPACE)
    legendLabels = ["P2VP radius: " + str(radius) for radius in radius_P2VP]
    data2D = []
    for radius in radius_P2VP:
        data2D.append(np.load("tests_13deg_line_CosineRippleGauss_P2VPradius_2D_"+ str(radius) +"nm.npy"))

    graphing.plot2D(simulationData=data2D[0], simData_axes=simulationDataAxes, realData=realData_npArray, realDat_axes=realDat_axes_Feb, zlim=[22,70000000])
    vert_slice_q = 0.2
    #graphing.yonedaPlot(vert_slice_q, data2D, data_axes=simulationDataAxes, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb, xmin = 0.08, xmax = 0.2, labels=legendLabels) #, data2_npArray=realData_npArray, data_axes2=realDat_axes_Feb)
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