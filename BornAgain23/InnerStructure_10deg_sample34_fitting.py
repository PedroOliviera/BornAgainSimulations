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
import scipy.ndimage

# Global Cut Positions
alpha_horizontal_lincut = 0.12
phi_vertical_lincut = 0.1441 #0.15
global ctr 
ctr=0
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

def sample_radial_paracrystal_truncated(omega_nm_int, P2VP_radius_xy, P2VP_radius_z, P2VP_std_dev_radius, P2VP_std_dev_height, P2VP_spacing, kappa_int, delta):
    material_PS  = ba.RefractiveMaterial("PS",     2.537E-06, 2.182E-09) 
    material_P2VP  = ba.RefractiveMaterial("P2VP", delta*1e-6, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09 
    #material_PS  = ba.RefractiveMaterial("PS",     2.33E-06, 2.182E-09) 
    #material_P2VP  = ba.RefractiveMaterial("P2VP", 2.55*1e-6, 2.58315258E-09 ) # 2.49112645E-06, 2.58315258E-09 
    #material_P2VP  = ba.RefractiveMaterial("P2VP", 3*1e-6, 2.58315258E-09 )
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.07E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.76E-06, 4.16025294E-08)
    material_FA  = ba.RefractiveMaterial("FA", 4.5E-06, 1.6E-07) #4.8

    Perovskite_Fraction = 0.5

    #FITTED PARAMETERS FROM 0.1DEG FIT

    omega_nm_top = 7.7886 
    spacing_top = 47.8097132
    kappa_top = 7.1920e-05
    PS_radius_xy = 16.89
    PS_radius_z = 9.48915916 # 6.48915916
    PS_std_dev_radius = 3 #0.1#2.34180742 
    PS_std_dev_height = 5# 9.1#3.000000008 

    #Total thickness from XRR
    total_thickness = 303.7 * nm
    num_layers = 4
    layer_thickness = total_thickness/num_layers

    ##################################################---------Surface FF---------##################################################################
    surface_layout = ba.ParticleLayout()
    # --- 1. Define your Statistical Parameters ---
    mu_R_phys = PS_radius_xy * nm
    sig_R_phys = PS_std_dev_radius * nm

    mu_H_phys = PS_radius_z * nm
    sig_H_phys = PS_std_dev_height * nm

    rho = 0.7  # Correlation of the LOG values from Sample 35 AFM

    # A. Convert Physical Stats to Log-Normal Parameters (Median & Scale/Sigma)
    med_R, scale_R = get_lognormal_params(mu_R_phys, sig_R_phys)
    med_H, scale_H = get_lognormal_params(mu_H_phys, sig_H_phys)

    # B. Get the "Mean of the Logs" (mu) for the correlation math
    # In lognormal math, ln(Median) = mu (the mean of the underlying normal)
    mu_log_R = np.log(med_R)
    mu_log_H = np.log(med_H)

    # --- 2. Outer Loop: Iterate over Radius ---
    distr_radius = ba.DistributionLogNormal(med_R, scale_R, 10, 2)

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
        distr_height = ba.DistributionLogNormal(cond_median_H, cond_scale_H, 10, 2)

        for h_sample in distr_height.distributionSamples():
            height_val = h_sample.value
            weight_h = h_sample.weight
            
            # Combined weight
            total_weight = weight_r * weight_h

            # --- 4. Create Particle ---
            ff_PS = ba.Spheroid(radius_val, height_val/2)
            particle_PS = ba.Particle(material_PS, ff_PS)
            
            surface_layout.addParticle(particle_PS, total_weight*(1 - Perovskite_Fraction))

    ##############################################-----------Interior FF-----------##############################################################
    
    interior_layout = ba.ParticleLayout()
    # --- 1. Define your Statistical Parameters ---

    mu_R_phys = P2VP_radius_xy * nm
    sig_R_phys = P2VP_std_dev_radius * nm

    mu_H_phys = P2VP_radius_z * nm
    sig_H_phys = P2VP_std_dev_height * nm

    rho = 0.7  # Correlation of the LOG values from Sample 35 AFM

    # A. Convert Physical Stats to Log-Normal Parameters (Median & Scale/Sigma)
    med_R, scale_R = get_lognormal_params(mu_R_phys, sig_R_phys)
    med_H, scale_H = get_lognormal_params(mu_H_phys, sig_H_phys)

    # B. Get the "Mean of the Logs" (mu) for the correlation math
    # In lognormal math, ln(Median) = mu (the mean of the underlying normal)
    mu_log_R = np.log(med_R)
    mu_log_H = np.log(med_H)

    # --- 2. Outer Loop: Iterate over Radius ---
    distr_radius = ba.DistributionLogNormal(med_R, scale_R, 10, 2)

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
        distr_height = ba.DistributionLogNormal(cond_median_H, cond_scale_H, 10, 2)

        for h_sample in distr_height.distributionSamples():
            height_val = h_sample.value
            weight_h = h_sample.weight
            
            # Combined weight
            total_weight = weight_r * weight_h

            # --- 4. Create Particle ---
            ff_P2VP = ba.Spheroid(radius_val, height_val/2)
            particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
            
            interior_layout.addParticle(particle_P2VP, total_weight*(1 - Perovskite_Fraction)*2)
    
    ##############################################-----------Interior FF-----------##############################################################
    '''
    interior_layout = ba.ParticleLayout()
    # --- 1. Define your Statistical Parameters ---

    mu_R_phys = P2VP_radius_xy * nm
    sig_R_phys = P2VP_std_dev_radius * nm

    mu_H_phys = P2VP_radius_z * nm
    sig_H_phys = P2VP_std_dev_height * nm

    rho = 0.7  # Correlation of the LOG values from Sample 35 AFM

    # A. Convert Physical Stats to Log-Normal Parameters (Median & Scale/Sigma)
    med_R, scale_R = get_lognormal_params(mu_R_phys, sig_R_phys)
    med_H, scale_H = get_lognormal_params(mu_H_phys, sig_H_phys)

    # B. Get the "Mean of the Logs" (mu) for the correlation math
    # In lognormal math, ln(Median) = mu (the mean of the underlying normal)
    mu_log_R = np.log(med_R)
    mu_log_H = np.log(med_H)

    # --- 2. Outer Loop: Iterate over Radius ---
    distr_radius = ba.DistributionLogNormal(med_R, scale_R, 10, 2)

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
        distr_height = ba.DistributionLogNormal(cond_median_H, cond_scale_H, 10, 2)

        for h_sample in distr_height.distributionSamples():
            height_val = h_sample.value
            weight_h = h_sample.weight
            
            # Combined weight
            total_weight = weight_r * weight_h

            # --- 4. Create Particle ---
            ff_P2VP = ba.Spheroid(radius_val, height_val/2)
            FA_radius = 7*nm
            translate_Z = height_val/2 - FA_radius
            ff_FA = ba.Sphere(FA_radius/P2VP_radius_z * height_val)
            particle_FA = ba.Particle(material_FA, ff_FA)
            particle_FA_position = R3(0*nm, 0*nm, translate_Z)
            particle_FA.translate(particle_FA_position)
            particle_P2VP = ba.Particle(material_P2VP, ff_P2VP)
            coreshell = ba.CoreAndShell(particle_FA, particle_P2VP)

            
            interior_layout.addParticle(coreshell, total_weight* 1)
    
    
    '''
    #interference function interiour layer
    damping_length_nm = 1000
    density_nm2_int = 3.6e-4
    iff_int = ba.InterferenceRadialParacrystal(P2VP_spacing*nm, damping_length_nm*nm)
    iff_pdf_int = ba.Profile1DGauss(omega_nm_int*nm)
    iff_int.setProbabilityDistribution(iff_pdf_int)
    iff_int.setKappa(kappa_int)
    #Particle Layout
    interior_layout.setInterference(iff_int)
    interior_layout.setTotalParticleSurfaceDensity(density_nm2_int)

    #interference function surface
    dampening_length = 1000
    density_nm2_top = 3.6e-4
    iff_top = ba.InterferenceRadialParacrystal(spacing_top*nm, dampening_length*nm)
    iff_pdf_top = ba.Profile1DGauss(omega_nm_top*nm)
    iff_top.setProbabilityDistribution(iff_pdf_top)
    iff_top.setKappa(kappa_top)
    #Particle Layout
    surface_layout.setInterference(iff_int)
    surface_layout.setTotalParticleSurfaceDensity(density_nm2_top)
   
    #Define roughness
    transient = ba.ErfTransient()
    autocorr_PS = ba.SelfAffineFractalModel(2.01*nm, 0.7, 200*nm)
    autocorr_SiO2 = ba.SelfAffineFractalModel(0.1*nm, 0.7, 25*nm)

    roughness_PS = ba.Roughness(autocorr_PS, transient)
    roughness_SiO2 = ba.Roughness(autocorr_SiO2, transient)

    #occlusions
    void_layout = ba.ParticleLayout()
    ff_voids = ba.Sphere(7*nm)
    particle_voids = ba.Particle(ba.Vacuum(), ff_voids)
    void_layout.addParticle(particle_voids, 1)
    void_layout.setTotalParticleSurfaceDensity(1e-3)
    #Layers
    top = ba.Layer(ba.Vacuum())
    top.addLayout(surface_layout)
    polymer1 = ba.Layer(material_PS, layer_thickness, roughness_PS)
    polymer2 = ba.Layer(material_PS, layer_thickness)
    #polymer2.addLayout(interior_layout)
    polymer3 = ba.Layer(material_PS, layer_thickness)
    polymer3.addLayout(interior_layout)
    polymer4 = ba.Layer(material_PS, layer_thickness)
    #polymer4.addLayout(interior_layout)
    polymer5 = ba.Layer(material_PS, layer_thickness)
    SiO2 = ba.Layer(material_SiO2, 2.621*nm, roughness_SiO2)
    #SiO2.addLayout(interior_layout)
    Si = ba.Layer(material_Si_Sub)

    #Sample
    s = ba.Sample()
    s.addLayer(top)
    s.addLayer(polymer1)
    s.addLayer(polymer2)
    s.addLayer(polymer3)
    s.addLayer(polymer4)
    s.addLayer(polymer5)
    s.addLayer(SiO2)
    s.addLayer(Si)
    return s

# ---------- Simulation Wrapper ----------

def run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=4e11, background = 23, sim_2D = False):
    if sim_2D is True:
        sim = g.get_simulation_2D(
        sample, beamtime, angle=alpha_i_deg, 
        beamIntensity=intensity, ROI_deg=ROI_deg, 
        #bounds_alpha=[0,1.75], bounds_phi=[0.0675, 1.75], center_horizontal_slice_values=[alpha_horizontal_lincut], center_vertical_slice_values=[phi_vertical_lincut], 
        background = background
        )
    else:
        sim = g.get_simulation_line(
            sample, beamtime, angle=alpha_i_deg, 
            beamIntensity=intensity, ROI_deg=ROI_deg, 
            bounds_alpha=[0,1.75], bounds_phi=[0.0675, 1.75], center_horizontal_slice_values=[alpha_horizontal_lincut], center_vertical_slice_values=[phi_vertical_lincut], 
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
    omega_nm_int = params['omega_nm_int'].value
    P2VP_radius_xy = params['P2VP_radius_xy'].value
    P2VP_radius_z = params['P2VP_radius_z'].value
    P2VP_std_dev_radius = params['P2VP_std_dev_radius'].value
    P2VP_std_dev_height = params['P2VP_std_dev_height'].value
    P2VP_spacing = params['P2VP_spacing'].value
    kappa_int = params['kappa_int'].value
    delta = params['delta'].value
    intensity = params['intensity'].value
    background = params['background'].value


    # 2. Build & Simulate
    sample = sample_radial_paracrystal_truncated(
        omega_nm_int = omega_nm_int,
        P2VP_radius_xy = P2VP_radius_xy,
        P2VP_radius_z = P2VP_radius_z ,
        P2VP_std_dev_radius = P2VP_std_dev_radius,
        P2VP_std_dev_height = P2VP_std_dev_height,
        P2VP_spacing = P2VP_spacing,
        kappa_int = kappa_int,
        delta = delta,
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

    #w_h = 1.0 / len(res_h)
    #w_v = 1.0 / len(res_v)

    #res_h_scaled = np.sqrt(w_h) * res_h
    #res_v_scaled = np.sqrt(w_v) * res_v
    #residuals = np.concatenate((res_h_scaled, res_v_scaled))
    residuals = res_h

    # Logging
    chi2 = np.sum(residuals**2)
    global ctr
    print(f"Iter {ctr} : omega={omega_nm_int:.2f}, P2VP_radius_xy={P2VP_radius_xy:.1f}, P2VP_radius_z={P2VP_radius_z:.2e}, P2VP spacing={P2VP_spacing:.2e}, chi2={chi2:.2f}")
    ctr += 1
    return residuals

def run_lmfit_optimization(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg):
    params = lmfit.Parameters()

    params.add('omega_nm_int', value=10, min=3, max=14)
    params.add('P2VP_radius_xy', value=18.5, min=15, max=30)
    params.add('P2VP_radius_z', value=10.172595, min=5, max=15)
    params.add('P2VP_std_dev_radius', value=1, min=0.5, max=8)
    params.add('P2VP_std_dev_height', value=1, min=0.5, max=5)
    params.add('P2VP_spacing', value=40.61897, min=36, max=45)
    params.add('kappa_int', value=0.35, vary = False)
    params.add('delta', value = 4.537, vary=False) 
    params.add('background', value = 27.0209844, vary=False)
    params.add('intensity', value=4e11, min=1e10, max=10e12)

    # --- STEP 1: GLOBAL SEARCH (Coarse Fit) ---
    print("--- Step 1: Global Search (Differential Evolution) ---")
    # This explores the whole parameter space defined by min/max
    global_result = lmfit.minimize(
        solve_residuals, 
        params, 
        method='differential_evolution', 
        args=(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg),
        nan_policy='omit',
        disp = False,
        max_nfev=100 
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
    exp_npz_file = "4_10deg.npz"
    beamtime     = "feb"
    alpha_i_deg  = 0.1
    ROI_deg      = (0, 0, 0.6, 1.75)
    
    # 2. LOAD DATA
    exp_arr, _ = g.load_npz_data(exp_npz_file, exp_dir)
    exp_axes = g.extent_phi_alpha_from_image(exp_arr, 'feb', alpha_i_deg=alpha_i_deg)


    # 3. RUN FITTING (Uncomment to fit)
    #best_params = run_lmfit_optimization(exp_arr, exp_axes, beamtime, alpha_i_deg, ROI_deg)

    
    # 4. FINAL SIMULATION Use best_params here
    #sample = sample_radial_paracrystal_truncated(
    #    omega_nm_int=best_params['omega_nm_int'].value, 
    #    P2VP_spacing = best_params['P2VP_spacing'].value,
    #    kappa_int = best_params['kappa_int'].value,
    #    P2VP_radius_xy=best_params['P2VP_radius_xy'].value,
    #    P2VP_radius_z=best_params['P2VP_radius_z'].value,
    #    P2VP_std_dev_radius=best_params['P2VP_std_dev_radius'].value,
    #    P2VP_std_dev_height=best_params['P2VP_std_dev_height'].value,
    #    delta=best_params['delta'].value
    #    )
    
    sample = sample_radial_paracrystal_truncated(
        omega_nm_int=10, 
        P2VP_spacing = 40,
        kappa_int = 0.35,
        #P2VP_radius_xy=18.5,
        P2VP_radius_xy=31/2,
        #P2VP_radius_z=22,#10.172595,
        P2VP_radius_z=18,
        P2VP_std_dev_radius=1.5,#2.5-1.75
        P2VP_std_dev_height=3, #3-2
        delta=3.25#.537
        )
    
    #I_sim, extent_angles = run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=best_params['intensity'].value, background=best_params['background'].value)
    I_sim, extent_angles = run_simulation(sample, beamtime, alpha_i_deg, ROI_deg, intensity=50e10, background = 18, sim_2D = True)

    # 5. VISUALIZE
    
    #I_sim = scipy.ndimage.gaussian_filter(I_sim, sigma=[2.0, 1.5])

    #plot_horizontal_slice_simple(alpha_cut_deg=alpha_horizontal_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim, sim_extent=extent_angles)
    #plot_vertical_slice_simple(phi_cut_deg=phi_vertical_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim, sim_extent=extent_angles)

    #plot_horizontal_slice_simple(alpha_cut_deg=alpha_horizontal_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim, sim_extent=extent_angles, save_fname='horizontal_S4_fitted_10deg')
    #plot_vertical_slice_simple(phi_cut_deg=phi_vertical_lincut, exp_arr=exp_arr, exp_extent=exp_axes, sim_arr=I_sim, sim_extent=extent_angles, save_fname='vertical_S4_fitted_10deg')
    
    # 6. 2D STITCH PLOT
    #I_sim, extent_angles = run_simulation(sample, beamtime, alpha_i_deg, (0, 0, 1, 1.75), intensity=best_params['intensity'].value, background=best_params['background'].value, sim_2D = False)

    ys_e, xs_e = slices_for_window(exp_arr.shape, exp_axes, extent_angles)
    exp_crop = np.flipud(exp_arr[ys_e, xs_e])
    exp_arr_subtract = resize_nearest(exp_crop, I_sim.shape)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    merged_2d = reflect_and_stitch_horizontal(I_sim, exp_arr_subtract)
    x0, x1, y0, y1 = extent_angles
    im1 = ax1.imshow(merged_2d, origin='lower', extent=(-x1, x1, y0, y1), aspect='auto', norm=LogNorm(25, 3.7e4), cmap='jet')
    ax1.set_xlabel(r"$\varphi_f$ (°)"); ax1.set_ylabel(r"$\alpha_f$ (°)")
    ax1.xaxis.label.set_fontsize(22)
    ax1.yaxis.label.set_fontsize(22)
    ax1.tick_params(axis='both', labelsize=20)
    cbar = fig.colorbar(im1, ax=ax1, label="Intensity (a.u.)")
    cbar.set_label("Intensity (a.u.)", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    for spine in ax1.spines.values():
        spine.set_linewidth(2.5)   # thicker axes lines
    ax1.tick_params(width=2, length=8)  # width = line thickness, length = tick size
    ax1.set_yticks(np.arange(0, 1.80, 0.25))
    ax1.set_xticks(np.arange(-0.5, 0.55, 0.25))

    plt.show()

    #I_sim, extent_angles = run_simulation(sample, beamtime, alpha_i_deg, (0, 0, 1, 1.75), intensity=best_params['intensity'].value, background=best_params['background'].value, sim_2D = True)

    ys_e, xs_e = slices_for_window(exp_arr.shape, exp_axes, extent_angles)
    exp_crop = np.flipud(exp_arr[ys_e, xs_e])
    exp_arr_subtract = resize_nearest(exp_crop, I_sim.shape)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    merged_2d = reflect_and_stitch_horizontal(I_sim, exp_arr_subtract)
    x0, x1, y0, y1 = extent_angles
    im1 = ax1.imshow(merged_2d, origin='lower', extent=(-x1, x1, y0, y1), aspect='auto', norm=LogNorm(25, 3.7e4), cmap='jet')
    ax1.set_xlabel(r"$\varphi_f$ (°)"); ax1.set_ylabel(r"$\alpha_f$ (°)")
    ax1.xaxis.label.set_fontsize(22)
    ax1.yaxis.label.set_fontsize(22)
    ax1.tick_params(axis='both', labelsize=20)
    cbar = fig.colorbar(im1, ax=ax1, label="Intensity (a.u.)")
    cbar.set_label("Intensity (a.u.)", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    for spine in ax1.spines.values():
        spine.set_linewidth(2.5)   # thicker axes lines
    ax1.tick_params(width=2, length=8)  # width = line thickness, length = tick size
    ax1.set_yticks(np.arange(0, 1.80, 0.25))
    ax1.set_xticks(np.arange(-0.5, 0.55, 0.25))
    plt.savefig(r"C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\Paper\Figures\GISAXS Fits\GISAXS_surface_0p1deg_sample4_2.png", dpi=500)
    plt.savefig(r"C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\Paper\Figures\GISAXS Fits\GISAXS_surface_0p1deg_sample4_2.pdf", dpi=500)
    plt.show()