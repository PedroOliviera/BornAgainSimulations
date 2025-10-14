from GISAXS_Analysis import GISAXS_setup_v23 as g
from GISAXS_Analysis import Graphing_Analysis as graphing
import bornagain as ba
from bornagain import deg, nm, R3
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


# ---------- USER INPUTS ----------
exp_dir      = r"C:\BornAgainSimulations\data\exp-npz"
exp_npz_file = "32_10deg.npz"     # saved with Q axes: [qy_min,qy_max,qz_min,qz_max]
alpha_i_deg  = 0.1
beamtime     = "feb"
ROI_deg      = (0, 0, 0.5, 1.75)          # (phi_min, alpha_min, phi_max, alpha_max)

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

def sample_radial_paracrystal_CosineRippleGauss(omega_nm=10,#6,
                              damping_length_nm=10000, density_nm2=1e-4): #3.6e-4
    material_PS  = ba.RefractiveMaterial("PS",     3.5e-6, 2.3e-9)
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
        ff_PS = ba.CosineRippleGauss(diam_K[i], diam_K[i], height_K[i])
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])
        print(height_K[i])
        print(diam_K[i])


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

def sample_radial_paracrystal_hemiellipsoid(omega_nm=10,#6,
                              damping_length_nm=10000, density_nm2=3.6e-4): #3.6e-4
    material_PS  = ba.RefractiveMaterial("PS",     3.5e-6, 2.3e-9)
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
        ff_PS = ba.HemiEllipsoid(diam_K[i]/2, diam_K[i]/2, height_K[i])
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])
        print(height_K[i])
        print(diam_K[i])


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



def sample_radial_paracrystal_truncated(omega_nm=10,#6,
                              damping_length_nm=450, density_nm2=3.6e-4): #3.6e-4
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09)
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)

    offset_diameter =  - 15*nm
    offset_height = 1*nm
    spacing = 63*nm + offset_diameter
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
        height = height_K[i] + offset_height
        R = truncated_radius(height, diam_K[i] + offset_diameter)
        b = 2*R - height
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
    iff.setKappa(0.65)
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

def sample_radial_paracrystal_truncated_perovskite(omega_nm=10,#6,
                              damping_length_nm=450, density_nm2=3.6e-4): #3.6e-4
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09)
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)
    material_FA = ba.RefractiveMaterial("FA", 6.0e-6, 7.8e-8)

    offset_diameter =  - 15*nm
    offset_height = 2*nm
    spacing = 63*nm + offset_diameter
    num_samples = 10
    perovskite_radius = 3*nm
    # Minimal test — adjust file path as needed
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    h_r.visualize_kmedoids(dmin_nm, hsub_nm, diam_K, height_K, labels, weight_rep=weight_K)
    h_r.plt.show()

    
    
    #form factor

    perov_np = ba.Particle(material_FA, ba.Sphere(perovskite_radius))
    
    surface_layout = ba.ParticleLayout(); 
    for i in range(num_samples):
        height = height_K[i] + offset_height
        R = truncated_radius(height, diam_K[i] + offset_diameter)
        b = 2*R - height
        ff_PS = ba.SphericalSegment(R* nm, 0.0*nm, b* nm)
        particle_PS= ba.Particle(material_PS, ff_PS)

        np_depth = height
        composition = ba.Compound()
        composition.addComponent(particle_PS)
        composition.addComponent(perov_np, R3(0,0, -perovskite_radius))

        surface_layout.addParticle(composition, weight_K[i])
        print(height_K[i])
        print(diam_K[i])
        print(R)
        print(b)


    #interference function
    iff = ba.InterferenceRadialParacrystal(spacing*nm, damping_length_nm*nm)
    iff_pdf = ba.Profile1DGauss(omega_nm*nm)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(0.65)
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

def sample_radial_paracrystal_truncated_with_roughness(omega_nm=6,#6,
                              damping_length_nm=400, density_nm2=3.6e-4): #3.6e-4
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
    iff.setKappa(0.35)
    #Particle Layout
    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(density_nm2)

    #roughness
    sig = 3*nm
    hurst = 0.7
    corr = 25*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    transient = ba.ErfTransient()
    roughness = ba.Roughness(autocorr, transient)
    
    #Layers
    top = ba.Layer(ba.Vacuum())
    top.addLayout(surface_layout)
    polymer = ba.Layer(material_PS, 214*nm, roughness)
    sub = ba.Layer(m_substrate)
    
    #Sample
    s = ba.Sample()
    s.addLayer(top)
    s.addLayer(polymer)
    s.addLayer(sub)
    return s

# ---------- SAMPLE (BA23-compliant) ----------
sample = sample_radial_paracrystal_truncated()

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
I_sim   = I_flat.reshape(n_alpha, n_phi)   # rows: α, cols: φ

phi_min, a_min, phi_max, a_max = ROI_deg
extent_angles = [phi_min, phi_max, a_min, a_max]

# ---------- LOAD EXPERIMENT (Q axes) ----------
exp_arr, _ = g.load_npz_data(exp_npz_file, exp_dir)
exp_axes = g.extent_phi_alpha_from_image(exp_arr, 'feb', alpha_i_deg=alpha_i_deg)



zmax = 3.7e4
zmin = 25

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

plt.savefig('cosineRippleGauss.png')
plot_horizontal_slice_simple(alpha_cut_deg=0.1, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim,sim_extent=extent_angles)

plt.show()