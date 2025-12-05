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
import lmfit

# Global Cut Positions
alpha_horizontal_lincut = 0.126
phi_vertical_lincut = 0.13315

# ---------- Helper Functions (Plotting & array manipulation) ----------

def reflect_and_stitch_horizontal(a, b, reflect='first'):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Arrays must have the same height.")
    if reflect == 'first':
        combined = np.hstack((np.fliplr(a), b))
    elif reflect == 'second':
        combined = np.hstack((a, np.fliplr(b)))
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
        iy0, iy1 = ny - iy1, ny - iy0
    return slice(iy0, iy1), slice(ix0, ix1)

def resize_nearest(img, out_shape):
    in_h, in_w = img.shape
    out_h, out_w = out_shape
    iy = np.clip(np.round((np.arange(out_h) + 0.5) * in_h / out_h - 0.5).astype(int), 0, in_h - 1)
    ix = np.clip(np.round((np.arange(out_w) + 0.5) * in_w / out_w - 0.5).astype(int), 0, in_w - 1)
    return img[iy][:, ix]

def plot_horizontal_slice_simple(alpha_cut_deg, exp_arr, exp_extent, sim_arr, sim_extent, exp_origin="upper", sim_origin="lower"):
    phi_eL, phi_eR, a_eB, a_eT = map(float, exp_extent)
    phi_sL, phi_sR, a_sB, a_sT = map(float, sim_extent)
    n_alpha_e, n_phi_e = exp_arr.shape
    n_alpha_s, n_phi_s = sim_arr.shape
    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)
    if exp_origin.lower() == "lower":
        alpha_e = np.linspace(a_eB, a_eT, n_alpha_e)
    else:
        alpha_e = np.linspace(a_eT, a_eB, n_alpha_e)
    if sim_origin.lower() == "lower":
        alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)
    else:
        alpha_s = np.linspace(a_sT, a_sB, n_alpha_s)
    row_e = int(np.argmin(np.abs(alpha_e - alpha_cut_deg)))
    row_s = int(np.argmin(np.abs(alpha_s - alpha_cut_deg)))
    y_exp_on_sim = np.interp(phi_s, phi_e, exp_arr[row_e, :], left=np.nan, right=np.nan)
    
    plt.figure(figsize=(6,4))
    plt.semilogy(phi_s, y_exp_on_sim, label=fr"Exp @ $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.semilogy(phi_s, sim_arr[row_s, :], label=fr"Sim @ $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.xlabel(r"$\varphi_f$ (deg)"); plt.ylabel("Intensity (a.u.)")
    plt.title(fr"Horizontal slice at $\alpha_f$={alpha_cut_deg:.2f}°")
    plt.xlim(0,1)
    plt.legend(); plt.tight_layout()

def plot_vertical_slice_simple(phi_cut_deg, exp_arr, exp_extent, sim_arr, sim_extent, exp_origin="upper", sim_origin="lower"):
    phi_eL, phi_eR, a_eB, a_eT = map(float, exp_extent)
    phi_sL, phi_sR, a_sB, a_sT = map(float, sim_extent)
    n_alpha_e, n_phi_e = exp_arr.shape
    n_alpha_s, n_phi_s = sim_arr.shape
    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)
    if exp_origin.lower() == "lower":
        alpha_e = np.linspace(a_eB, a_eT, n_alpha_e)
    else:
        alpha_e = np.linspace(a_eT, a_eB, n_alpha_e)
    if sim_origin.lower() == "lower":
        alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)
    else:
        alpha_s = np.linspace(a_sT, a_sB, n_alpha_s)
    
    col_e = int(np.argmin(np.abs(phi_e - phi_cut_deg)))
    col_s = int(np.argmin(np.abs(phi_s - phi_cut_deg)))
    y_exp = exp_arr[:, col_e]
    y_sim = sim_arr[:, col_s]
    
    if alpha_e[0] > alpha_e[-1]:
        alpha_e_sorted = np.flip(alpha_e)
        y_exp_sorted = np.flip(y_exp)
    else:
        alpha_e_sorted = alpha_e
        y_exp_sorted = y_exp
        
    y_exp_on_sim = np.interp(alpha_s, alpha_e_sorted, y_exp_sorted, left=np.nan, right=np.nan)
    
    plt.figure(figsize=(6,4))
    plt.semilogy(alpha_e, y_exp, label=fr"Exp @ $\varphi_f$={phi_cut_deg:.2f}°", marker='o', markersize=2, linestyle='')
    plt.semilogy(alpha_s, y_sim, label=fr"Sim @ $\varphi_f$={phi_cut_deg:.2f}°")
    plt.xlim(0, 1.75); plt.ylim(50,5e4)
    plt.xlabel(r"$\alpha_f$ (deg)"); plt.ylabel("Intensity (a.u.)")
    plt.title(fr"Vertical slice at $\varphi_f$={phi_cut_deg:.2f}°")
    plt.legend(); plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()

def truncated_radius(h, d):
    return float((h**2 + (d/2)**2)/(2*h))

def extract_experimental_slices(exp_arr, exp_axes, sim_extent, sim_shape):
    """
    Interpolates the full experimental image onto the simulation's 
    horizontal and vertical slice coordinates.
    """
    # 1. Setup Axes
    # Experiment
    phi_eL, phi_eR, a_eB, a_eT = exp_axes
    n_alpha_e, n_phi_e = exp_arr.shape
    phi_e = np.linspace(phi_eL, phi_eR, n_phi_e)
    # Handle descending alpha axis (common in images)
    alpha_e = np.linspace(a_eT, a_eB, n_alpha_e) 
    
    # Simulation
    phi_sL, phi_sR, a_sB, a_sT = sim_extent
    n_alpha_s, n_phi_s = sim_shape
    phi_s = np.linspace(phi_sL, phi_sR, n_phi_s)
    alpha_s = np.linspace(a_sB, a_sT, n_alpha_s)

    # 2. Horizontal Slice (Interpolate Exp to Sim's Phi Axis)
    # Find row in Exp closest to cut
    row_e = int(np.argmin(np.abs(alpha_e - alpha_horizontal_lincut)))
    exp_slice_h_raw = exp_arr[row_e, :]
    exp_slice_h = np.interp(phi_s, phi_e, exp_slice_h_raw)

    # 3. Vertical Slice (Interpolate Exp to Sim's Alpha Axis)
    # Find col in Exp closest to cut
    col_e = int(np.argmin(np.abs(phi_e - phi_vertical_lincut)))
    exp_slice_v_raw = exp_arr[:, col_e]
    
    # Handle alpha sort order for interpolation
    if alpha_e[0] > alpha_e[-1]:
        exp_slice_v = np.interp(alpha_s, np.flip(alpha_e), np.flip(exp_slice_v_raw))
    else:
        exp_slice_v = np.interp(alpha_s, alpha_e, exp_slice_v_raw)
        
    return exp_slice_h, exp_slice_v

def get_lognormal_params(mean, std_dev):
    """
    Converts physical Mean and StdDev to BornAgain's Median and Scale Param.
    """
    # Variance of the log-data (sigma^2)
    var_term = np.log(1 + (std_dev / mean)**2)
    # Scale Param (sigma of the logs)
    scale_param = np.sqrt(var_term)
    # Median (geometric mean)
    median = mean / np.exp(var_term / 2)
    return median, scale_param
# ---------- Sample Definitions ----------

def sample_radial_paracrystal_truncated(omega_nm, PS_radius_xy, PS_radius_z, std_dev_radius, std_dev_height, spacing, kappa):
    material_PS  = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.353858E-09)
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)

    surface_layout = ba.ParticleLayout()
    # --- 1. Define your Statistical Parameters ---
    mu_R_phys = PS_radius_xy * nm
    sig_R_phys = std_dev_radius * nm

    mu_H_phys = PS_radius_z * nm
    sig_H_phys = std_dev_height * nm

    rho = 0.7  # Correlation of the LOG values

    # A. Convert Physical Stats to Log-Normal Parameters (Median & Scale/Sigma)
    med_R, scale_R = get_lognormal_params(mu_R_phys, sig_R_phys)
    med_H, scale_H = get_lognormal_params(mu_H_phys, sig_H_phys)

    # B. Get the "Mean of the Logs" (mu) for the correlation math
    # In lognormal math, ln(Median) = mu (the mean of the underlying normal)
    mu_log_R = np.log(med_R)
    mu_log_H = np.log(med_H)

    # --- 2. Outer Loop: Iterate over Radius ---
    distr_radius = ba.DistributionLogNormal(med_R, scale_R)

    for r_sample in distr_radius.distributionSamples():
        radius_val = r_sample.value
        weight_r = r_sample.weight
        
        # --- 3. Inner Loop: Iterate over Height (Conditional) ---
        
        # transform current radius sample to log-space
        log_radius_val = np.log(radius_val)

        # Calculate Conditional Mean in LOG-SPACE
        # (How far is log(R) from log(Median_R), scaled by correlation)
        cond_mu_log_H = mu_log_H + rho * (scale_H / scale_R) * (log_radius_val - mu_log_R)
        
        # Calculate Conditional Sigma in LOG-SPACE (Standard deviation of the logs)
        cond_scale_H = scale_H * np.sqrt(1 - rho**2)

        # Convert the Log-Space Mean back to a Median for BornAgain
        # Note: The 'scale' (sigma) remains the same between log-space and BornAgain param
        cond_median_H = np.exp(cond_mu_log_H)
        
        # Safety check for nearly perfect correlation
        if cond_scale_H < 1e-12:
            cond_scale_H = 1e-12
        
        # Create the conditional LogNormal distribution
        distr_height = ba.DistributionLogNormal(cond_median_H, cond_scale_H)

        for h_sample in distr_height.distributionSamples():
            height_val = h_sample.value
            weight_h = h_sample.weight
            
            # Combined weight
            total_weight = weight_r * weight_h

            # --- 4. Create Particle ---
            ff_PS = ba.Spheroid(radius_val, height_val)
            particle_PS = ba.Particle(material_PS, ff_PS)
            
            surface_layout.addParticle(particle_PS, total_weight)

    #interference function
    dampening_length = 1000
    iff = ba.InterferenceRadialParacrystal(spacing*nm, dampening_length*nm)
    iff_pdf = ba.Profile1DGauss(omega_nm*nm)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(kappa)
    #Particle Layout
    surface_layout.setInterference(iff)
    surface_layout.setTotalParticleSurfaceDensity(3.6e-5)
   
    #Layers
    top = ba.Layer(ba.Vacuum())
    top.addLayout(surface_layout)
    polymer = ba.Layer(material_PS, 288*nm)
    sub = ba.Layer(m_substrate)
   
    #Sample
    s = ba.Sample()
    s.addLayer(top)
    s.addLayer(polymer)
    s.addLayer(sub)
    return s

# ---------- Simulation Wrapper ----------

def run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=4e11, background = 23):
    sim = g.get_simulation_line(
        sample, beamtime, angle=alpha_i_deg, 
        center_horizontal_slice_values=[alpha_horizontal_lincut], 
        center_vertical_slice_values=[phi_vertical_lincut], 
        beamIntensity=intensity, number_slices=3, ROI_deg=ROI_deg, bounds_alpha=[0,1.75], bounds_phi=[0.0675, 1],
        background = background
    )
    # Force threading options if needed here
    # sim.options().setNumberOfThreads(1) 
    
    df = sim.simulate()
    
    # FORCE MEMORY COPY to prevent debugger crashes
    I_flat = dac.asNpArray(df.dataArray()).copy()
    phi    = dac.npArray(df.xCenters()).copy()
    
    n_phi = int(phi.size)
    N = int(I_flat.size)
    if n_phi == 0: raise RuntimeError("n_phi is 0")
    n_alpha = N // n_phi
    I_sim = I_flat.reshape(n_alpha, n_phi)
    
    phi_min, a_min, phi_max, a_max = ROI_deg
    extent_angles = [phi_min, phi_max, a_min, a_max]
    return I_sim, extent_angles

# ---------- FIT LOGIC (SIMPLIFIED) ----------

def solve_residuals(params, exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg):
    """
    Objective function for lmfit.
    """
    # 1. Extract Parameters
    omega = params['omega_nm'].value
    PS_radius_xy = params['PS_radius_xy'].value
    PS_radius_z = params['PS_radius_z'].value
    std_dev_radius = params['std_dev_radius'].value
    std_dev_height = params['std_dev_height'].value
    spacing = params['spacing'].value
    kappa = params['kappa'].value
    intensity = params['intensity'].value
    background = params['background'].value

    # 2. Build & Simulate
    sample = sample_radial_paracrystal_truncated(
        omega_nm = params['omega_nm'].value, 
        spacing  = params['spacing'].value,
        kappa = params['kappa'].value,
        PS_radius_xy = params['PS_radius_xy'].value,
        PS_radius_z = params['PS_radius_z'].value,
        std_dev_radius = params['std_dev_radius'].value,
        std_dev_height = params['std_dev_height'].value
        )
    I_sim, sim_extent = run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=intensity, background = background)
    
    # 3. Extract Simulation Slices
    # Determine axes from I_sim shape and extent
    n_alpha_s, n_phi_s = I_sim.shape
    phi_s = np.linspace(sim_extent[0], sim_extent[1], n_phi_s)
    alpha_s = np.linspace(sim_extent[2], sim_extent[3], n_alpha_s)
    
    # Find indices
    row_s = int(np.argmin(np.abs(alpha_s - alpha_horizontal_lincut)))
    col_s = int(np.argmin(np.abs(phi_s - phi_vertical_lincut)))
    
    sim_slice_h = I_sim[row_s, :]
    sim_slice_v = I_sim[:, col_s]
    
    # 4. Extract Experimental Slices (using Helper)
    exp_slice_h, exp_slice_v = extract_experimental_slices(exp_arr, exp_axes, sim_extent, I_sim.shape)

    # 5. Calculate Residuals (Log Space, avoiding zeros)
    valid_h = (sim_slice_h > 1e-9) & (exp_slice_h > 1e-9)
    valid_v = (sim_slice_v > 1e-9) & (exp_slice_v > 1e-9)
    
    res_h = np.log10(sim_slice_h[valid_h]) - np.log10(exp_slice_h[valid_h])
    res_v = np.log10(sim_slice_v[valid_v]) - np.log10(exp_slice_v[valid_v])
    
    residuals = np.concatenate((res_h, res_v))
    
    # Logging
    chi2 = np.sum(residuals**2)
    print(f"Iter: omega={omega:.2f}, PS_radius_xy={PS_radius_xy:.1f}, PS_radius_z={PS_radius_z:.2e}, std_dev_radius={std_dev_radius:.2e}")
    print(f"std_dev_height={std_dev_height:.2e}, spacing={spacing:.1e}, kappa={kappa:.1e}, int={intensity:.1e}, background={background:.1e}, chi2={chi2:.2f}")
    
    return residuals

def run_lmfit_optimization(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg):
    params = lmfit.Parameters()

    params.add('omega_nm', value=4.02841517, min=1.0, max=8)
    params.add('PS_radius_xy', value=37, min=20, max=45)
    params.add('PS_radius_z', value=7, min=5, max=12)
    params.add('std_dev_radius', value=4.3, min=1, max=8)
    params.add('std_dev_height', value=1.5, min=0.5, max=3)
    params.add('spacing', value=48, min=40, max=60)
    params.add('kappa', value = 0.65, min = 0.0, max = 1)
    params.add('intensity', value=4e11, min=1e10, max=10e12)
    params.add('background', value = 28, min = 20, max=35)

    # --- STEP 1: GLOBAL SEARCH (Coarse Fit) ---
    print("--- Step 1: Global Search (Differential Evolution) ---")
    # This explores the whole parameter space defined by min/max
    global_result = lmfit.minimize(
        solve_residuals, 
        params, 
        method='differential_evolution', 
        args=(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg),
        nan_policy='omit',
        # Strategy: limit population and generations to save time
        options={'maxiter': 10, 'popsize': 5, 'disp': True} # 10 / 5
    )
    
    print("\nGlobal Search Complete. Best Chi2:", global_result.chisqr)
    print(lmfit.fit_report(global_result))

    # --- STEP 2: LOCAL POLISH (Fine Fit) ---
    print("--- Step 2: Local Polish (Levenberg-Marquardt) ---")
    # We use the parameters found in Step 1 as the starting point for Step 2
    local_result = lmfit.minimize(
        solve_residuals, 
        global_result.params, # <--- Use params from Step 1
        method='leastsq',     # <--- Default local optimizer
        args=(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg),
        nan_policy='omit',
        max_nfev=20         # Limit local steps 100
    )

    print("\n--- Final Fit Report ---")
    print(lmfit.fit_report(local_result))
    
    return local_result.params

# ===================================================================
#                          MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    print('start')
    # 1. SETUP
    exp_dir      = r"C:\BornAgainSimulations\data\exp-npz\feb"
    exp_npz_file = "32_10deg.npz"
    beamtime     = "feb"
    alpha_i_deg  = 0.10
    ROI_deg      = (0, 0, 1, 1.75)
    
    # 2. LOAD DATA
    exp_arr, _ = g.load_npz_data(exp_npz_file, exp_dir)
    exp_axes = g.extent_phi_alpha_from_image(exp_arr, 'feb', alpha_i_deg=alpha_i_deg)

    # 3. RUN FITTING (Uncomment to fit)
    best_params = run_lmfit_optimization(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg)

    # 4. FINAL SIMULATION Use best_params here
    sample = sample_radial_paracrystal_truncated(
        omega_nm=best_params['omega_nm'].value, 
        spacing = best_params['spacing'].value,
        kappa = best_params['kappa'].value,
        PS_radius_xy=best_params['PS_radius_xy'].value,
        PS_radius_z=best_params['PS_radius_z'].value,
        std_dev_radius=best_params['std_dev_radius'].value,
        std_dev_height=best_params['std_dev_height'].value
        )
    '''
    sample = sample_radial_paracrystal_truncated(
        omega_nm=3, 
        spacing = 40,
        kappa = 0.54927224
        )
        intensity=1.3526e12
    '''
    
    I_sim, extent_angles = run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=best_params['intensity'].value, background=best_params['background'].value)
    #I_sim, extent_angles = run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=4e11, background = 30)

    # 5. VISUALIZE
    plot_horizontal_slice_simple(alpha_cut_deg=alpha_horizontal_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim, sim_extent=extent_angles)
    plot_vertical_slice_simple(phi_cut_deg=phi_vertical_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim, sim_extent=extent_angles)
    plt.show()
    
    # 6. 2D STITCH PLOT
    ys_e, xs_e = slices_for_window(exp_arr.shape, exp_axes, extent_angles)
    exp_crop = np.flipud(exp_arr[ys_e, xs_e])
    exp_arr_subtract = resize_nearest(exp_crop, I_sim.shape)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    merged_2d = reflect_and_stitch_horizontal(I_sim, exp_arr_subtract)
    x0, x1, y0, y1 = extent_angles
    im1 = ax1.imshow(merged_2d, origin='lower', extent=(-x1, x1, y0, y1), aspect='auto', norm=LogNorm(25, 3.7e4), cmap='jet')
    ax1.set_xlabel(r"$\varphi_f$ (°)"); ax1.set_ylabel(r"$\alpha_f$ (°)")
    fig.colorbar(im1, ax=ax1, label="Intensity (a.u.)")
    plt.show()