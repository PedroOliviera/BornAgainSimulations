from GISAXS_Analysis import GISAXS_setup_v23 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
import bornagain as ba
from bornagain import deg, nm
from bornagain.numpyutil import Arrayf64Converter as dac
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from GISAXS_Analysis import height_radius_from_lineprofiles as h_r
import matplotlib as mpl
# ---------- Important functions ----------

def plot_horizontal_slice_simple(
    alpha_cut_deg,
    exp_arr, exp_extent,          # [phi_left, phi_right, alpha_bottom, alpha_top]
    sim_arr, sim_extent,          # [phi_left, phi_right, alpha_bottom, alpha_top]
    exp_origin="upper",           # how you PLOTTED exp_arr: "upper" (default) or "lower"
    sim_origin="lower",           # how you PLOTTED sim_arr: "lower" (typical)
):
    """
    Horizontal slice (constant α) -> plot intensity vs φ on a shared axis.
    Arrays are shaped (n_alpha, n_phi). Handles origin mismatch.
    """
    # Unpack extents
    phi_eL, phi_eR, a_eB, a_eT = map(float, exp_extent)
    phi_sL, phi_sR, a_sB, a_sT = map(float, sim_extent)

    n_alpha_e, n_phi_e = exp_arr.shape
    n_alpha_s, n_phi_s = sim_arr.shape

    # 1D φ axes (left->right always increases)
    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)

    # 1D α axes depend on how the image was PLOTTED (origin)
    if exp_origin.lower() == "lower":
        alpha_e = np.linspace(a_eB, a_eT, n_alpha_e)   # row 0 -> α_bottom
    else:
        alpha_e = np.linspace(a_eT, a_eB, n_alpha_e)   # row 0 -> α_top

    if sim_origin.lower() == "lower":
        alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)
    else:
        alpha_s = np.linspace(a_sT, a_sB, n_alpha_s)

    # Nearest α row on each grid
    row_e = int(np.argmin(np.abs(alpha_e - alpha_cut_deg)))
    row_s = int(np.argmin(np.abs(alpha_s - alpha_cut_deg)))

    # Extract slices (vs φ on their native grids)
    y_exp = exp_arr[row_e, :]
    y_sim = sim_arr[row_s, :]

    # Interpolate EXP slice onto SIM φ grid so curves share x-axis
    y_exp_on_sim = np.interp(phi_s, phi_e, y_exp, left=np.nan, right=np.nan)

    # Plot
    plt.figure(figsize=(6,4))
    plt.semilogy(phi_s, y_exp_on_sim, label=fr"Exp @ $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.semilogy(phi_s, y_sim,        label=fr"Sim @ $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.xlabel(r"$\varphi_f$ (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(fr"Horizontal slice at $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.legend()
    plt.tight_layout()

def plot_vertical_slice_simple(
    phi_cut_deg,
    exp_arr, exp_extent,          # [phi_left, phi_right, alpha_bottom, alpha_top]
    sim_arr, sim_extent,          # [phi_left, phi_right, alpha_bottom, alpha_top]
    exp_origin="upper",           # how you PLOTTED exp_arr: "upper" (default) or "lower"
    sim_origin="lower",           # how you PLOTTED sim_arr: "lower" (typical)
):
    """
    Vertical slice (constant φ) -> plot intensity vs α on a shared axis.
    Arrays are shaped (n_alpha, n_phi). Handles origin mismatch and ensures
    α axes are ascending for interpolation.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Unpack extents
    phi_eL, phi_eR, a_eB, a_eT = map(float, exp_extent)
    phi_sL, phi_sR, a_sB, a_sT = map(float, sim_extent)

    n_alpha_e, n_phi_e = exp_arr.shape
    n_alpha_s, n_phi_s = sim_arr.shape

    # φ axes (left→right always increases)
    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)

    # α axes depend on how the image was PLOTTED (origin)
    if exp_origin.lower() == "lower":
        alpha_e = np.linspace(a_eB, a_eT, n_alpha_e)   # row 0 -> α_bottom
    else:
        alpha_e = np.linspace(a_eT, a_eB, n_alpha_e)   # row 0 -> α_top

    if sim_origin.lower() == "lower":
        alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)
    else:
        alpha_s = np.linspace(a_sT, a_sB, n_alpha_s)

    # Nearest φ column on each grid
    col_e = int(np.argmin(np.abs(phi_e - phi_cut_deg)))
    col_s = int(np.argmin(np.abs(phi_s - phi_cut_deg)))

    # Extract slices (vs α on their native grids)
    y_exp = exp_arr[:, col_e]
    y_sim = sim_arr[:, col_s]

    # Ensure α axes are ascending for interpolation & plotting
    def _asc(x, y):
        return (x, y) if x[0] <= x[-1] else (x[::-1], y[::-1])

    alpha_e_plot, y_exp_plot = _asc(alpha_e, y_exp)
    alpha_s_plot, y_sim_plot = _asc(alpha_s, y_sim)

    # Interpolate EXP slice onto SIM α grid so curves share x-axis
    y_exp_on_sim = np.interp(alpha_s_plot, alpha_e_plot, y_exp_plot, left=np.nan, right=np.nan)

    # Plot
    plt.figure(figsize=(6,4))
    plt.semilogy(alpha_s_plot, y_exp_on_sim, label=fr"Exp @ $\varphi_f$={phi_cut_deg:.2f}°")
    plt.semilogy(alpha_s_plot, y_sim_plot,   label=fr"Sim @ $\varphi_f$={phi_cut_deg:.2f}°")
    plt.xlabel(r"$\alpha_f$ (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(fr"Vertical slice at $\varphi_f$={phi_cut_deg:.2f}°")
    plt.legend()
    plt.tight_layout()

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

def sample_radial_paracrystal_truncated(omega_nm=6,#6,
                              damping_length_nm=10000, density_nm2=3.6e-4): #3.6e-4
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09)
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
    surface_layout = ba.ParticleLayout(); 
    for i in range(num_samples):
        R = truncated_radius(height_K[i], diam_K[i] - offset)
        b = 2*R - height_K[i]
        ff_PS = ba.SphericalSegment(R* nm, 0.0*nm, b* nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])
        print(height_K[i])
        print(diam_K[i])
        print(R)
        print(b)


    #interference function
    iff = ba.InterferenceRadialParacrystal(spacing*nm, damping_length_nm*nm)
    iff_pdf = ba.Profile1DGauss(omega_nm*nm)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(0.25)
    #Particle Layout
    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(density_nm2)
    
    #Layers
    top = ba.Layer(ba.Vacuum())
    top.addLayout(surface_layout)
    polymer = ba.Layer(material_PS, 214*nm)
    sub = ba.Layer(m_substrate)
    
    #Sample
    s = ba.Sample()
    s.addLayer(top)
    s.addLayer(polymer)
    s.addLayer(sub)
    return s

def sample_radial_paracrystal_truncated_with_roughness(height, diameter, brush_thickness): #3.6e-4
    
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09)
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08)
    material_SiO2 = ba.RefractiveMaterial("Si Sub", 4.74631315E-06, 4.16025294E-08)

    omega_nm=7
    damping_length_nm=0
    domain_size = 20000
    variance = 0

    spacing = 85

    height = height 

    diameter = diameter #np.mean([67.5, 66.7, 63.8, 66.6, 66.9, 66, 62.6, 67.6, 65.3])


    # Minimal test — adjust file path as needed
    #lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    #xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    #hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    #diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    #h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    #h_r.plt.show()

    
    
    #form factor
    surface_layout = ba.ParticleLayout(); 
    R = truncated_radius(height, diameter)
    b = 2*R - height
    ff_PS = ba.SphericalSegment(R* nm, 0.0*nm, b* nm)
    particle_PS= ba.Particle(material_PS, ff_PS)
    surface_layout.addParticle(particle_PS, 1)

    print("Radius")
    print(R)
    print("Removed b:")
    print(b)


    #interference function

    lattice = ba.BasicLattice2D(spacing * nm, spacing* nm, 120*deg, 0*deg)

    iff = ba.Interference2DParacrystal(lattice, damping_length_nm*nm, domain_size*nm, domain_size*nm)
    iff.setIntegrationOverXi(True)
    iff_pdf = ba.Profile2DGauss(omega_nm*nm, omega_nm*nm, 0*deg)
    iff.setProbabilityDistributions(iff_pdf, iff_pdf)
    if variance != 0:
        iff.setPositionVariance(variance*nm)
    #Particle Layout
    surface_layout.setInterference(iff)

    #roughness
    sig = 0.2*nm
    hurst = 0.7
    corr = 25*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    transient = ba.ErfTransient()
    roughness_sub = ba.Roughness(autocorr, transient)
    
    #Layers
    top = ba.Layer(ba.Vacuum())
    top.addLayout(surface_layout)
    polymer = ba.Layer(material_PS, brush_thickness*nm)
    SiO2 = ba.Layer(material_SiO2, 2*nm, roughness_sub)
    sub = ba.Layer(m_substrate)
    
    #Sample
    s = ba.Sample()
    s.addLayer(top)
    s.addLayer(polymer)
    s.addLayer(SiO2)
    s.addLayer(sub)
    return s

def plot2D(exp_arr, I_sim, extent_angles, zmax, zmin):

    if zmin == 0.0 and zmax == 0.0:
        norm = mpl.colors.Normalize(0, 1)
    else:
        norm = mpl.colors.LogNorm(zmin, zmax)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5), constrained_layout=True)

    im1 = ax1.imshow(exp_arr, extent=exp_axes,
                    aspect="auto", norm=norm, cmap='jet')
    ax1.set_title("Experiment (resampled to φ/α)")
    ax1.set_xlabel(r"$\varphi_f$ (deg)"); ax1.set_ylabel(r"$\alpha_f$ (deg)")
    fig.colorbar(im1, ax=ax1, label="Intensity (a.u.)")

    im2 = ax2.imshow(I_sim, origin= 'lower', extent=extent_angles,
                    aspect="auto", norm=norm, cmap='jet')
    ax2.set_title("Simulation (radial paracrystal, 20 nm spheres)")
    ax2.set_xlabel(r"$\varphi_f$ (deg)"); ax2.set_ylabel(r"$\alpha_f$ (deg)")
    fig.colorbar(im2, ax=ax2, label="Intensity (a.u.)")

    ax1.set_ylim(extent_angles[2],extent_angles[3])
    ax2.set_ylim(extent_angles[2],extent_angles[3])
    ax1.set_xlim(extent_angles[0],extent_angles[1])
    ax2.set_xlim(extent_angles[0],extent_angles[1])

def func_simulation(sample):
    # ---------- SIMULATE ----------
    sim = g.get_simulation_2D(sample_model=sample,
                            detectorDistBeamtime=beamtime,
                            angle=alpha_i_deg,
                            beamIntensity=4e11,
                            ROI_deg=ROI_deg,
                            divergence=False,
                            resolution=False,
                            oneThread=False)

    print('starting simulation')
    df = sim.simulate()
    print('finished simulation')

    # BA23-official way to get NumPy arrays from Datafield:
    I_flat = dac.asNpArray(df.dataArray())     # 1D intensities, length N = n_alpha * n_phi
    phi    = dac.npArray(df.xCenters())        # x-axis centers (φ), length n_phi
   
    n_phi = int(phi.size)
    N     = int(I_flat.size)
    if n_phi == 0 or N == 0 or (N % n_phi != 0):
        raise RuntimeError(f"Cannot infer 2D shape from BA23 Datafield: N={N}, n_phi={n_phi}")

    n_alpha = N // n_phi
    I_sim   = I_flat.reshape(n_alpha, n_phi).copy()   # rows: α, cols: φ
    

    phi_min, a_min, phi_max, a_max = ROI_deg
    extent_angles = [phi_min, phi_max, a_min, a_max]
    
    return I_sim, extent_angles

# ---------- USER INPUTS ----------
exp_dir      = r"C:\BornAgainSimulations\data\exp-npz"
exp_npz_file = "Si_0p2deg.npz"     # saved with Q axes: [qy_min,qy_max,qz_min,qz_max]
alpha_i_deg  = 0.2
beamtime     = "dec"
ROI_deg      = (0, 0, 0.8, 1.75)          # (phi_min, alpha_min, phi_max, alpha_max)

# ---------- LOAD EXPERIMENT (Q axes) ----------
exp_arr, _ = g.load_npz_data(exp_npz_file, exp_dir)
exp_axes = g.extent_phi_alpha_from_image(exp_arr, 'dec', alpha_i_deg=alpha_i_deg)

# ---------- SAMPLE (BA23-compliant) ----------





diameter_array = [45, 50, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 80, 85]
height_array = [5, 10, 12.5, 15, 17.5, 20, 25, 30]
brush_thickness = 2
for diameter in diameter_array:
    for height in height_array:
        save_label = '_diameter_' + str(diameter) + '_height_' + str(height) 

        sample = sample_radial_paracrystal_truncated_with_roughness(height=height, diameter=diameter, brush_thickness=brush_thickness)
        I_sim_returned, extent_angles_returned = func_simulation(sample)

        plot2D(exp_arr, I_sim_returned, extent_angles_returned, zmax = 3.7e4, zmin = 25)
        plt.savefig('Monolayer_paracrystal_sim1_0p2deg' + save_label + '.png', dpi = 300)
        plt.savefig('Monolayer_paracrystal_sim1_0p2deg' + save_label + '.pdf', dpi = 300)
        plot_horizontal_slice_simple(alpha_cut_deg=0.25, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim_returned, 
                                    sim_extent=extent_angles_returned)
        
        plt.savefig('Monolayer_paracrystal_sim1_0p2deg_horizontal_lineprofile' + save_label + '.png', dpi = 300)
        plt.savefig('Monolayer_paracrystal_sim1_0p2deg_horizontal_lineprofile' + save_label + '.pdf', dpi = 300)
        plot_vertical_slice_simple(phi_cut_deg=0.0967, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim_returned, 
                                sim_extent=extent_angles_returned, exp_origin="upper", sim_origin="lower")
        
        plt.savefig('Monolayer_paracrystal_sim1_0p2deg_vertical_lineprofile' + save_label + '.png', dpi = 300)
        plt.savefig('Monolayer_paracrystal_sim1_0p2deg_vertical_lineprofile' + save_label + '.pdf', dpi = 300)
        
        g.save_npz_data('Monolayer_paracrystal_sim1_0p2deg' + save_label + '.npz',I_sim_returned, extent_angles_returned)
        plt.close()

t_brush_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
diameter = 65
height = 15
for t_brush in t_brush_array:

    save_label = '_brush_thickness_' + str(t_brush) 

    sample = sample_radial_paracrystal_truncated_with_roughness(height=height, diameter=diameter, brush_thickness=t_brush)
    I_sim_returned, extent_angles_returned = func_simulation(sample)

    plot2D(exp_arr, I_sim_returned, extent_angles_returned, zmax = 3.7e4, zmin = 25)
    plt.savefig('Monolayer_paracrystal_sim1_0p2deg' + save_label + '.png', dpi = 300)
    plt.savefig('Monolayer_paracrystal_sim1_0p2deg' + save_label + '.pdf', dpi = 300)
    plot_horizontal_slice_simple(alpha_cut_deg=0.25, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim_returned, 
                                sim_extent=extent_angles_returned)
    
    plt.savefig('Monolayer_paracrystal_sim1_0p2deg_horizontal_lineprofile' + save_label + '.png', dpi = 300)
    plt.savefig('Monolayer_paracrystal_sim1_0p2deg_horizontal_lineprofile' + save_label + '.pdf', dpi = 300)
    plot_vertical_slice_simple(phi_cut_deg=0.0967, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim_returned, 
                            sim_extent=extent_angles_returned, exp_origin="upper", sim_origin="lower")
    
    plt.savefig('Monolayer_paracrystal_sim1_0p2deg_vertical_lineprofile' + save_label + '.png', dpi = 300)
    plt.savefig('Monolayer_paracrystal_sim1_0p2deg_vertical_lineprofile' + save_label + '.pdf', dpi = 300)
    
    g.save_npz_data('Monolayer_paracrystal_sim1_0p2deg' + save_label + '.npz',I_sim_returned, extent_angles_returned)
    plt.close()
