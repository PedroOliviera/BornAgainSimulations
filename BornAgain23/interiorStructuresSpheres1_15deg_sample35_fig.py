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


def reflect_and_stitch_horizontal(a, b, reflect='first'):
    """
    Reflect one of the arrays about a vertical axis (left-right flip)
    and stitch them side by side (horizontally).

    Parameters
    ----------
    a, b : np.ndarray
        Input 2D arrays.
    reflect : {'first', 'second'}, default='second'
        Which array to reflect before stitching.
        'first'  → flip a left-right
        'second' → flip b left-right

    Returns
    -------
    combined : np.ndarray
        Combined 2D array: [reflected | other]
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    if a.shape[0] != b.shape[0]:
        raise ValueError("Arrays must have the same height (number of rows).")

    if reflect == 'first':
        a_ref = np.fliplr(a)
        combined = np.hstack((a_ref, b))
    elif reflect == 'second':
        b_ref = np.fliplr(b)
        combined = np.hstack((a, b_ref))
    else:
        raise ValueError("reflect must be 'first' or 'second'.")

    return combined

def slices_for_window(shape, data_extent, window_extent, origin='upper'):
    ny, nx = map(int, shape)
    x0, x1, y0, y1 = map(float, data_extent)
    X0, X1, Y0, Y1 = map(float, window_extent)

    x_edges = np.linspace(min(x0, x1), max(x0, x1), nx + 1)
    y_edges = np.linspace(min(y0, y1), max(y0, y1), ny + 1)

    ix0 = max(0, np.searchsorted(x_edges, min(X0, X1), side='right') - 1)
    ix1 = min(nx, np.searchsorted(x_edges, max(X0, X1), side='left'))

    iy0 = max(0, np.searchsorted(y_edges, min(Y0, Y1), side='right') - 1)
    iy1 = min(ny, np.searchsorted(y_edges, max(Y0, Y1), side='left'))

    if origin == 'upper':
        # Flip vertical order for images where y decreases downward
        iy0, iy1 = ny - iy1, ny - iy0

    return slice(iy0, iy1), slice(ix0, ix1)


def resize_nearest(img, out_shape):
    """Nearest-neighbor resize (pure NumPy), preserves aspect with integer mapping of centers."""
    in_h, in_w = img.shape
    out_h, out_w = out_shape
    iy = np.round((np.arange(out_h) + 0.5) * in_h / out_h - 0.5).astype(int)
    ix = np.round((np.arange(out_w) + 0.5) * in_w / out_w - 0.5).astype(int)
    iy = np.clip(iy, 0, in_h - 1)
    ix = np.clip(ix, 0, in_w - 1)
    return img[iy][:, ix]

def plot_horizontal_slice_simple(
    alpha_cut_deg,
    exp_arr,
    exp_extent,
    sim_arr,
    sim_extent,
    exp_origin="upper",
    sim_origin="lower",
    save_fname=None,
):
    # --- build coordinate grids ---
    phi_eL, phi_eR, a_eB, a_eT = map(float, exp_extent)
    phi_sL, phi_sR, a_sB, a_sT = map(float, sim_extent)

    n_alpha_e, n_phi_e = exp_arr.shape
    n_alpha_s, n_phi_s = sim_arr.shape

    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)

    # experimental alpha grid
    if exp_origin.lower() == "lower":
        alpha_e = np.linspace(a_eB, a_eT, n_alpha_e)
    else:
        alpha_e = np.linspace(a_eT, a_eB, n_alpha_e)

    # simulation alpha grid
    if sim_origin.lower() == "lower":
        alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)
    else:
        alpha_s = np.linspace(a_sT, a_sB, n_alpha_s)

    # --- extract horizontal slices at alpha_cut_deg ---
    row_e = int(np.argmin(np.abs(alpha_e - alpha_cut_deg)))
    row_s = int(np.argmin(np.abs(alpha_s - alpha_cut_deg)))

    y_exp = exp_arr[row_e, :]   # exp vs phi_e
    y_sim = sim_arr[row_s, :]   # sim vs phi_s

    # Make phi axes increasing for nicer saving/plotting
    if phi_e[0] > phi_e[-1]:
        phi_e_plot = np.flip(phi_e)
        y_exp_plot = np.flip(y_exp)
    else:
        phi_e_plot = phi_e
        y_exp_plot = y_exp

    if phi_s[0] > phi_s[-1]:
        phi_s_plot = np.flip(phi_s)
        y_sim_plot = np.flip(y_sim)
    else:
        phi_s_plot = phi_s
        y_sim_plot = y_sim

    # --- save BOTH datasets in the same text file, on native phi axes ---
    if save_fname:
        fname = str(save_fname)
        if not fname.endswith(".txt"):
            fname += ".txt"

        len_e = phi_e_plot.size
        len_s = phi_s_plot.size
        N = max(len_e, len_s)

        # pad with NaNs so we can have one rectangular array
        phi_e_col = np.full(N, np.nan)
        y_exp_col = np.full(N, np.nan)
        phi_s_col = np.full(N, np.nan)
        y_sim_col = np.full(N, np.nan)

        phi_e_col[:len_e] = phi_e_plot
        y_exp_col[:len_e] = y_exp_plot
        phi_s_col[:len_s] = phi_s_plot
        y_sim_col[:len_s] = y_sim_plot

        data = np.column_stack((phi_e_col, y_exp_col,
                                phi_s_col, y_sim_col))

        np.savetxt(
            fname,
            data,
            fmt="%.6e",
            header="# phi_exp(deg)  I_exp  phi_sim(deg)  I_sim  "
                   f"(horizontal slice at alpha_f={alpha_cut_deg:.3f} deg)",
        )

    # --- plotting on their respective axes ---
    plt.figure(figsize=(6, 4))

    plt.semilogy(
        phi_e_plot,
        y_exp_plot,
        label=fr"Exp @ $\alpha_f$={alpha_cut_deg:.2f}°",
        marker="o",
        markersize=2,
        linestyle="",
    )

    plt.semilogy(
        phi_s_plot,
        y_sim_plot,
        label=fr"Sim @ $\alpha_f$={alpha_cut_deg:.2f}°",
    )

    plt.xlabel(r"$\varphi_f$ (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(fr"Horizontal slice at $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.xlim(0, 1)  # keep your original limit; change if needed
    plt.ylim(20, 1e5)
    plt.legend()
    plt.tight_layout()


def plot_vertical_slice_simple(
    phi_cut_deg,
    exp_arr,
    exp_extent,
    sim_arr,
    sim_extent,
    exp_origin="upper",
    sim_origin="lower",
    save_fname=None,
):
    # --- build coordinate grids ---
    phi_eL, phi_eR, a_eB, a_eT = map(float, exp_extent)
    phi_sL, phi_sR, a_sB, a_sT = map(float, sim_extent)

    n_alpha_e, n_phi_e = exp_arr.shape
    n_alpha_s, n_phi_s = sim_arr.shape

    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)

    # experimental alpha grid
    if exp_origin.lower() == "lower":
        alpha_e = np.linspace(a_eB, a_eT, n_alpha_e)
    else:
        alpha_e = np.linspace(a_eT, a_eB, n_alpha_e)

    # simulation alpha grid
    if sim_origin.lower() == "lower":
        alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)
    else:
        alpha_s = np.linspace(a_sT, a_sB, n_alpha_s)

    # --- extract vertical slices at phi_cut_deg ---
    col_e = int(np.argmin(np.abs(phi_e - phi_cut_deg)))
    col_s = int(np.argmin(np.abs(phi_s - phi_cut_deg)))

    y_exp = exp_arr[:, col_e]   # exp vs alpha_e
    y_sim = sim_arr[:, col_s]   # sim vs alpha_s

    # Make both alpha axes increasing for nicer plotting/saving
    if alpha_e[0] > alpha_e[-1]:
        alpha_e_plot = np.flip(alpha_e)
        y_exp_plot   = np.flip(y_exp)
    else:
        alpha_e_plot = alpha_e
        y_exp_plot   = y_exp

    if alpha_s[0] > alpha_s[-1]:
        alpha_s_plot = np.flip(alpha_s)
        y_sim_plot   = np.flip(y_sim)
    else:
        alpha_s_plot = alpha_s
        y_sim_plot   = y_sim

    # --- save BOTH datasets in the same text file, on native axes ---
    if save_fname:
        fname = str(save_fname)
        if not fname.endswith(".txt"):
            fname += ".txt"

        len_e = alpha_e_plot.size
        len_s = alpha_s_plot.size
        N = max(len_e, len_s)

        # pad with NaNs so we can have one rectangular array
        alpha_e_col = np.full(N, np.nan)
        y_exp_col   = np.full(N, np.nan)
        alpha_s_col = np.full(N, np.nan)
        y_sim_col   = np.full(N, np.nan)

        alpha_e_col[:len_e] = alpha_e_plot
        y_exp_col[:len_e]   = y_exp_plot
        alpha_s_col[:len_s] = alpha_s_plot
        y_sim_col[:len_s]   = y_sim_plot

        data = np.column_stack((alpha_e_col, y_exp_col,
                                alpha_s_col, y_sim_col))

        np.savetxt(
            fname,
            data,
            fmt="%.6e",
            header="# alpha_f_exp  I_exp  alpha_f_sim  I_sim  "
                   f"(vertical slice at phi_f={phi_cut_deg:.3f} deg)",
        )

    # --- plotting on their respective axes ---
    plt.figure(figsize=(6, 4))

    plt.semilogy(
        alpha_e_plot,
        y_exp_plot,
        label=fr"Exp @ $\varphi_f$={phi_cut_deg:.2f}°",
        marker="o",
        markersize=2,
        linestyle="",
    )

    plt.semilogy(
        alpha_s_plot,
        y_sim_plot,
        label=fr"Sim @ $\varphi_f$={phi_cut_deg:.2f}°",
    )

    plt.xlim(0, 1.75)
    plt.ylim(50, 5e4)
    plt.xlabel(r"$\alpha_f$ (deg)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(fr"Vertical slice at $\varphi_f$={phi_cut_deg:.2f}°")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
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

def sample_radial_paracrystal_CosineRippleGauss(omega_nm=10,#6,
                              damping_length_nm=10000, density_nm2=0.01): #3.6e-4
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

def sample_radial_paracrystal_hemiellipsoid(omega_nm=6,#6,
                              damping_length_nm=10000, density_nm2=0.01): #3.6e-4
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

def sample_radial_paracrystal_truncated(omega_nm=0,#6,
                              damping_length_nm=0, density_nm2=0): #3.6e-4
    material_PS  = ba.RefractiveMaterial("PS",     2.537E-06, 2.182E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", 2.537E-06 + 0.25e-6, 0.017767581 * 1e-6 ) # 2.49112645E-06, 2.58315258E-09 
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.07E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.76E-06, 4.16025294E-08)
    material_FA  = ba.RefractiveMaterial("FA",     6E-06, 2.353858E-09) 

    offset = 7*nm
    spacing = 63*nm - offset
    num_samples = 10

    # Minimal test — adjust file path as needed
    lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

    xc, yc = h_r.load_lineprofiles(lineprofile_dir)
    hsub_nm, dmin_nm = h_r.extract_hsub_and_dmin(xc, yc, frac=0.0)

    diam_K, height_K, weight_K, labels = h_r.summarize_pairs_kmedoids(dmin_nm, hsub_nm, K=num_samples, scale=True)
    
    #form factor
    surface_layout = ba.ParticleLayout(); 
    for i in range(num_samples):
        R = truncated_radius(height_K[i], diam_K[i] - offset)
        b = 2*R - height_K[i]
        ff_PS = ba.SphericalSegment(R* nm, 0.0*nm, b* nm)
        particle_PS= ba.Particle(material_PS, ff_PS)
        surface_layout.addParticle(particle_PS, weight_K[i])

    #interference function
    omega_surface = 6
    damping_length_surface = 10000
    density_nm2_surface = 0.75e-4
    spacing_surface = 56*nm
    iff_surf = ba.InterferenceRadialParacrystal(spacing_surface*nm, damping_length_surface*nm)
    iff_pdf_surf = ba.Profile1DGauss(omega_surface*nm)
    iff_surf.setProbabilityDistribution(iff_pdf_surf)
    iff_surf.setKappa(0.35)

    #Particle Layout
    surface_layout.setInterference(iff_surf)
    surface_layout.setTotalParticleSurfaceDensity(density_nm2_surface)

    total_thickness = 214
    num_layers = 4
    layer_thickness = total_thickness/num_layers
    

    P2VP_radius_xy = 48/2 
    P2VP_radius_z = P2VP_radius_xy * 0.75
    std_dev = 2 

    interior_layout = ba.ParticleLayout()
    distr_radius = ba.DistributionGaussian(P2VP_radius_xy * nm, std_dev * nm)

    for xy_radius in distr_radius.distributionSamples():
        
        ff_P2VP = ba.Spheroid(xy_radius.value, (P2VP_radius_z/P2VP_radius_xy) * xy_radius.value) 
        #ff_P2VP = ba.Sphere(P2VP_radius_xy)
        particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
        vertical_shift = layer_thickness/2 - P2VP_radius_z
        particle_P2VP_position = R3(0*nm, 0*nm, vertical_shift)
        particle_P2VP.translate(particle_P2VP_position)
        interior_layout.addParticle(particle_P2VP, xy_radius.weight)

    interior_layout.setTotalParticleSurfaceDensity(density_nm2)

    omega_int = 10
    spacing_int = 50
    density_nm2_int = 4.6e-4
    damping_length_int = 10000*nm
    #interference function
    iff_int = ba.InterferenceRadialParacrystal(spacing_int*nm, damping_length_int*nm)
    iff_pdf_int = ba.Profile1DGauss(omega_int*nm)
    iff_int.setProbabilityDistribution(iff_pdf_int)
    iff_int.setKappa(0.35)
    #Particle Layout
    interior_layout.setInterference(iff_int)
    interior_layout.setTotalParticleSurfaceDensity(density_nm2_int)

    #Polymer Roughness 
    hurst = 0.49
    corr = 84*nm
    sig = 1*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    transient = ba.ErfTransient()
    roughness_poly = ba.Roughness(autocorr, transient)

    #Si Roughness 
    hurst = 0.49
    corr = 84*nm
    sig = 1*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    transient = ba.ErfTransient()
    roughness = ba.Roughness(autocorr, transient)

    #SiO2 Roughness 
    hurst = 0.49
    corr = 84*nm
    sig = 1*nm
    autocorr = ba.SelfAffineFractalModel(sig, hurst, corr)
    transient = ba.ErfTransient()
    roughness = ba.Roughness(autocorr, transient)

    #Layers
    top = ba.Layer(ba.Vacuum())
    top.addLayout(surface_layout)
    polymer1 = ba.Layer(material_PS, layer_thickness, roughness_poly)
    polymer2 = ba.Layer(material_PS, layer_thickness)
    polymer2.addLayout(interior_layout)
    polymer3 = ba.Layer(material_PS, layer_thickness)
    polymer3.addLayout(interior_layout)
    polymer4 = ba.Layer(material_PS, layer_thickness)
    polymer4.addLayout(interior_layout)
    SiO2 = ba.Layer(material_SiO2, 2*nm)
    SiO2.addLayout(interior_layout)
    Si = ba.Layer(material_Si_Sub)

    #Sample
    s = ba.Sample()
    s.addLayer(top)
    s.addLayer(polymer1)
    s.addLayer(polymer2)
    s.addLayer(polymer3)
    s.addLayer(polymer4)
    s.addLayer(SiO2)
    s.addLayer(Si)
    return s


# ---------- USER INPUTS ----------
exp_dir      = r"C:\BornAgainSimulations\data\exp-npz\feb"
exp_npz_file = "35_15deg.npz"     # saved with Q axes: [qy_min,qy_max,qz_min,qz_max]
alpha_i_deg  = 0.15
beamtime     = "feb"
ROI_deg      = (0, 0, 1, 1.75)           # (phi_min, alpha_min, phi_max, alpha_max)

# ---------- SAMPLE (BA23-compliant) ----------
sample = sample_radial_paracrystal_truncated()

# ---------- SIMULATE ----------
sim = g.get_simulation_2D(sample_model=sample, detectorDistBeamtime=beamtime, angle=alpha_i_deg, beamIntensity=2.8e12, ROI_deg=ROI_deg, divergence=False, resolution=False, oneThread=False)

alpha_horizontal_lincut = 0.1452
phi_vertical_lincut = 0.119
#sim = g.get_simulation_line(sample, 'feb', angle=alpha_i_deg, center_horizontal_slice_values=[alpha_horizontal_lincut], center_vertical_slice_values=[phi_vertical_lincut], beamIntensity=20e11, number_slices=3, ROI_deg=ROI_deg)

print('starting simulation')
#sim.options().setUseAvgMaterials(True)
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

plot_horizontal_slice_simple(alpha_cut_deg=alpha_horizontal_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim,sim_extent=extent_angles, save_fname='horizontal_S35_fitted_15deg')
plot_vertical_slice_simple(phi_cut_deg=phi_vertical_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim,sim_extent=extent_angles, save_fname='vertical_S35_fitted_105eg')

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



# 1) Crop EXP to the sim window (extent_angles)
ys_e, xs_e = slices_for_window(exp_arr.shape, exp_axes, extent_angles)
exp_crop = exp_arr[ys_e, xs_e]
exp_crop = np.flipud(exp_crop)  # flip vertically

# 2) Resample EXP crop to match sim shape exactly
exp_arr_subtract = resize_nearest(exp_crop, I_sim.shape)
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

merged_2d = reflect_and_stitch_horizontal(I_sim, exp_arr_subtract)

x0, x1, y0, y1 = extent_angles
new_extent = (-x1, x1, y0, y1)

im1 = ax1.imshow(merged_2d, origin='lower', extent=new_extent,
                 aspect='auto', norm=norm, cmap='jet')
ax1.set_xlabel(r"$\varphi_f$ (°)")
ax1.set_ylabel(r"$\alpha_f$ (°)")

cbar = fig.colorbar(im1, ax=ax1, label="Intensity (a.u.)")
cbar.set_label("Intensity (a.u.)", fontsize=20)
cbar.ax.tick_params(labelsize=20)

ax1.xaxis.label.set_fontsize(22)
ax1.yaxis.label.set_fontsize(22)
ax1.tick_params(axis='both', labelsize=20)

for spine in ax1.spines.values():
    spine.set_linewidth(2.5)   # thicker axes lines

ax1.tick_params(width=2, length=8)  # width = line thickness, length = tick size
ax1.set_yticks(np.arange(0, 1.80, 0.25))
ax1.set_xticks(np.arange(-0.5, 0.55, 0.25))

plt.savefig(r"C:\BornAgainSimulations\data\sim-npz\GISAXS_S35_interior_0p15deg.png", dpi=500)
plt.savefig(r"C:\BornAgainSimulations\data\sim-npz\GISAXS_S35_interior_0p15deg.pdf", dpi=500)

# No tight_layout() when using constrained layout
plt.show()