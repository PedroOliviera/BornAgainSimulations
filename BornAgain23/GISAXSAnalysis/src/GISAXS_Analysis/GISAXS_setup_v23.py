# === BornAgain 23 (angles-first) GISAXS utilities ============================
# Uses SphericalDetector and works in (phi_f, alpha_f) throughout.
# - ROIs and masks are in degrees.
# - Line-slice helpers take phi/alpha (deg), not q.
# - Experimental data NPZ stores ANGLE axes (phi_min, phi_max, alpha_min, alpha_max).

import bornagain as ba
from bornagain import deg, nm
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from scipy.ndimage import shift, rotate
import json
import datetime as dt
from scipy.interpolate import griddata

# ---------------------- Instrument / geometry constants ----------------------
wavelength = 1.25916 * ba.angstrom   # keep BornAgain length units (nm internally)
# Rayonix MX300 geometry (used only to compute FOV angles for SphericalDetector)
rayonix_npx, rayonix_npy = 4096, 4096
rayonix_pixel_size = 0.073242  # mm per pixel
rayonix_size_x = rayonix_npx * rayonix_pixel_size  # ~300 mm
rayonix_size_y = rayonix_npy * rayonix_pixel_size  # ~300 mm

# Beam center on detector (in pixels). Keep your calibrated values:
xpos_pix = 2048
ypos_pix = 2048
beam_xpos, beam_ypos = xpos_pix, ypos_pix  # pixels

# Two common detector distances (mm) from your script:
DIST_FEB = 2337.126
DIST_DEC = 3052.624

# IMPORTANT: in your file you already have `wavelength = 1.25916*ba.angstrom`
import numpy as np

# Globals you said exist (units shown for clarity)
# wavelength (Å) is not needed just to compute angle extents
rayonix_pixel_size = 0.073242  # mm / pixel

def extent_phi_alpha_from_image(img2d: np.ndarray, beamtime, alpha_i_deg: float):
    """
    Compute imshow extent [phi_left, phi_right, alpha_bottom, alpha_top] in degrees
    from a detector image (2D pixel intensities).

    Assumptions:
      - Direct beam is at the exact image center (row & col).
      - Image is horizontally centered (no horizontal shift needed for the horizon).
      - Rows increase downward.
      - Detector is flat and perpendicular to the beam at distance L_mm.

    Returns:
      (phi_left_deg, phi_right_deg, alpha_bottom_deg, alpha_top_deg)
    """

    if beamtime == 'feb':
        L_mm = DIST_FEB
    if beamtime == 'dec':
        L_mm = DIST_DEC
        
    if img2d.ndim != 2:
        raise ValueError("img2d must be a 2-D array")

    nrows, ncols = img2d.shape
    s_mm = float(rayonix_pixel_size)

    # --- 1) Locate direct beam (center pixel) and horizon pixel ---
    i_db = (nrows - 1) / 2.0          # direct-beam row (center)
    j_db = (ncols - 1) / 2.0          # direct-beam col (center)

    # Horizon is above the direct beam by Δi = (L/s)*tan(alpha_i)
    delta_i = (L_mm / s_mm) * np.tan(np.deg2rad(alpha_i_deg))
    i_hor = i_db - delta_i            # horizon row index
    j_hor = j_db                      # horizon column index (no horizontal shift)

    # --- 2) Horizontal angle φ at left/right image edges ---
    # Detector-plane horizontal offsets (mm) at left/right edges
    y_left_mm  = (0        - j_hor) * s_mm
    y_right_mm = ((ncols-1) - j_hor) * s_mm

    # φ = arctan(y / L)
    phi_left  = np.degrees(np.arctan2(y_left_mm,  L_mm))
    phi_right = np.degrees(np.arctan2(y_right_mm, L_mm))
    # ensure phi_left < phi_right
    if phi_left > phi_right:
        phi_left, phi_right = phi_right, phi_left

    # --- 3) Vertical angle α at the top/bottom image edges ---
    # α depends on column via cos(φ), so take the four corners and bound
    def phi_at_col(j):
        return np.arctan2((j - j_hor) * s_mm, L_mm)  # radians

    # Corners: (row, col) = (0, 0), (0, ncols-1), (nrows-1, 0), (nrows-1, ncols-1)
    corners = [
        (0, 0),
        (0, ncols-1),
        (nrows-1, 0),
        (nrows-1, ncols-1),
    ]
    alphas = []
    for i, j in corners:
        phi_r = phi_at_col(j)
        # Use z_up = (i_hor - i)*s so α>0 above horizon, α<0 below horizon
        z_up_mm = (i_hor - i) * s_mm
        alpha_r = np.arctan2(z_up_mm * np.cos(phi_r), L_mm)
        alphas.append(np.degrees(alpha_r))

    alpha_bottom = float(np.min(alphas))
    alpha_top    = float(np.max(alphas))

    return float(phi_left), float(phi_right), alpha_bottom, alpha_top


# ----------------------------- Utility helpers ------------------------------
def get_timestamp(prefer_internet=True) -> dt.datetime:
    tz = dt.datetime.now().astimezone().tzinfo
    return dt.datetime.now(tz=tz)

def _jsonify_params(d):
    """Make dict JSON-safe (handles numpy scalars/arrays)."""
    def py(v):
        if isinstance(v, np.generic):    return v.item()
        if isinstance(v, np.ndarray):    return v.tolist()
        if isinstance(v, (list, tuple)): return [py(x) for x in v]
        if isinstance(v, dict):          return {str(k): py(v2) for k, v2 in v.items()}
        return v
    return {str(k): py(v) for k, v in d.items()}

def save_npz_data(save_name: str, data2d: np.ndarray, axes_limits, params: dict | None = None):
    """
    Save a 2D array and its ANGLE axes limits: [phi_min, phi_max, alpha_min, alpha_max] in degrees.
    """
    date = get_timestamp().isoformat()
    axes = np.asarray(axes_limits, dtype=np.float64)
    payload = {"sim": data2d, "axes_limits": axes, "saved_at": date}
    if params is not None:
        payload["params_json"] = json.dumps(_jsonify_params(params), separators=(",", ":"))
    np.savez(save_name, **payload)

def load_npz_data(filename: str, directory: str, return_date: bool = False, return_params: bool = False):
    """
    Load data saved by save_npz_data().
    Returns:
      (array2d, ANGLE_axes) [+ saved_at] [+ params]
    ANGLE_axes = [phi_min_deg, phi_max_deg, alpha_min_deg, alpha_max_deg]
    """
    with np.load(os.path.join(directory, filename), allow_pickle=False) as f:
        sim  = f["sim"]
        axes = f["axes_limits"]
        out = [sim, axes]
        if return_date:
            saved_at_arr = f.get("saved_at")
            out.append(saved_at_arr.item() if saved_at_arr is not None else None)
        if return_params:
            pj = f.get("params_json")
            out.append(json.loads(str(pj)) if pj is not None else None)
    return tuple(out)

def real_data(filename, directory):
    """Loads experimental data (2D) as a numpy array."""
    filepath = os.path.join(directory, filename)
    return ba.readData2D(filepath).npArray()

# ----------------------- Angle-field-of-view calculations --------------------
def _beam_center_mm():
    """Beam center in mm (u0, v0) from pixel center and pixel size."""
    u0 = beam_xpos * rayonix_pixel_size
    v0 = beam_ypos * rayonix_pixel_size
    return u0, v0

def _rectangular_mm_extents():
    """Full detector mm extents along x (horizontal) and y (vertical)."""
    return (0.0, rayonix_size_x, 0.0, rayonix_size_y)

def _rect_to_angle_extents(distance_mm: float):
    """
    Compute angular extents (deg) [phi_min, phi_max, alpha_min, alpha_max]
    that correspond to the physical detector at a given distance and beam center.
    φ = atan2(x - u0, D), α = atan2(y - v0, D)
    """
    u0, v0 = _beam_center_mm()
    x_lo, x_hi, y_lo, y_hi = _rectangular_mm_extents()

    phi_min = np.degrees(np.arctan2(x_lo - u0, distance_mm))
    phi_max = np.degrees(np.arctan2(x_hi - u0, distance_mm))
    alp_min = np.degrees(np.arctan2(y_lo - v0, distance_mm))
    alp_max = np.degrees(np.arctan2(y_hi - v0, distance_mm))
    # Ensure ordering
    phi_min, phi_max = min(phi_min, phi_max), max(phi_min, phi_max)
    alp_min, alp_max = min(alp_min, alp_max), max(alp_min, alp_max)
    return [phi_min, phi_max, alp_min, alp_max]

def _angular_pixel_sizes(distance_mm: float):
    """Angular size (deg) per pixel along φ and α."""
    phi_min, phi_max, alp_min, alp_max = _rect_to_angle_extents(distance_mm)
    dphi = (phi_max - phi_min) / rayonix_npx
    dalp = (alp_max - alp_min) / rayonix_npy
    return dphi, dalp

# ------------------------- Detector (Spherical, BA23) ------------------------
def create_detector(distance_mm: float, add_resolution: bool):
    """
    BA23+: spherical detector with angle axes (φ_f, α_f).
    Field of view derived from your 300x300 mm geometry at 'distance_mm' and beam center.
    """
    phi_min, phi_max, alp_min, alp_max = _rect_to_angle_extents(distance_mm)

    detector = ba.SphericalDetector(
        rayonix_npx, phi_min*deg, phi_max*deg,
        rayonix_npy, alp_min*deg, alp_max*deg
    )

    if add_resolution:
        # Angular blur (deg); same effective blur as before (you can tune)
        sigma_deg = 0.000293
        detector.setResolutionFunction(
            ba.ResolutionFunction2DGaussian(sigma_deg*deg, sigma_deg*deg)
        )
    return detector

# --------------------------- Sample (unchanged) ------------------------------
def get_sampleTest():
    material_PS     = ba.RefractiveMaterial("PS",     2.51433698E-06, 2.35385822E-09)
    material_P2VP   = ba.RefractiveMaterial("P2VP",   1.656e-06,      1.096e-09)
    material_FA     = ba.RefractiveMaterial("FA",     3.90901641E-06, 1.79148728E-07)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04218633E-06, 7.83926453E-08)
    material_SiO2   = ba.RefractiveMaterial("SiO2",   4.7465490081665e-06, 4.1351946628761e-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    radius_PS = (48/2) * nm
    ff_PS = ba.Sphere(radius_PS)
    particle_PS = ba.Particle(material_PS, ff_PS)

    layout = ba.ParticleLayout()
    layout.addParticle(particle_PS)

    layer_1 = ba.Layer(material_Vacuum)
    layer_3 = ba.Layer(material_SiO2, 2*nm); layer_3.addLayout(layout)
    layer_4 = ba.Layer(material_Si_Sub)

    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_3)
    sample.addLayer(layer_4)
    return sample

# ----------------------- Simulation constructors (angles) --------------------
def _select_distance(tag: str) -> float:
    tag = (tag or "").lower()
    if tag == "dec": return DIST_DEC
    return DIST_FEB  # default 'feb'

def get_simulation_2D(sample_model,
                      detectorDistBeamtime='feb',
                      angle=None,                    # incidence α_i in degrees
                      beamIntensity=8e12,
                      ROI_deg=None,                 # [phi1, alpha1, phi2, alpha2] in degrees
                      divergence=False,
                      resolution=False,
                      oneThread=False,
                      beamstop_deg=None             # optional beamstop mask [phi1, alpha1, phi2, alpha2] deg
                      ):
    """
    2D simulation working entirely in ANGLES.
    - ROI, masks and (optional) beamstop are in degrees.
    """
    distance = _select_distance(detectorDistBeamtime)
    alpha_i_deg = float(angle)

    beam = ba.Beam(beamIntensity, wavelength, alpha_i_deg*deg)
    
    detector = create_detector(distance, resolution)
    #detector.setResolutionFunction(ba.ResolutionFunction2DGaussian(0.000075, 0.000075))

    sigma_phi = 0.0000085 * deg   # Tighter horizontal beam (slitted)
    sigma_alpha = 0.004 * deg # Tighter vertical beam (slitted)

    detector.setResolutionFunction(ba.ResolutionFunction2DGaussian(sigma_phi, sigma_alpha))
    sim = ba.ScatteringSimulation(beam, sample_model, detector)
    sim.options().setIncludeSpecular(False)
    sim.setBackground(ba.ConstantBackground(23))
    sim.options().setUseAvgMaterials(True)
    if divergence:
        sim.addParameterDistribution(ba.ParameterDistribution.BeamInclinationAngle,
                                     ba.DistributionGaussian(alpha_i_deg*deg, 0.016*deg, 7, 2))
        sim.addParameterDistribution(ba.ParameterDistribution.BeamAzimuthalAngle,
                                     ba.DistributionGaussian(0*deg, 0.042*deg, 7, 2))

    # ROI: if None, use full detector angles; otherwise pass given angles
    if ROI_deg is None:
        phi_min, phi_max, alp_min, alp_max = _rect_to_angle_extents(distance)
        sim.detector().setRegionOfInterest(phi_min*deg, alp_min*deg, phi_max*deg, alp_max*deg)
    else:
        phi1, a1, phi2, a2 = ROI_deg
        # ensure ordering
        phi_lo, phi_hi = min(phi1, phi2), max(phi1, phi2)
        a_lo,   a_hi   = min(a1, a2), max(a1, a2)
        sim.detector().setRegionOfInterest(phi_lo*deg, a_lo*deg, phi_hi*deg, a_hi*deg)

    # Optional beamstop mask in angle units
    if beamstop_deg is not None:
        bphi1, ba1, bphi2, ba2 = beamstop_deg
        sim.detector().addMask(ba.Rectangle(bphi1*deg, ba1*deg, bphi2*deg, ba2*deg), False)

    if oneThread:
        sim.options().setNumberOfThreads(1)

    return sim

# ---------------------- Experimental data: ANGLE axes ------------------------
def tifToNpzConversion(filename: str, directory: str,
                       detectorDistBeamtime: str,
                       angle: float,
                       ROI_deg=None, beamstop_deg=None):
    """
    Convert a .tif (or any 2D file readable by BA) to .npz:
    - recenters/rotates image (your existing routines)
    - stores ANGLE axes via a matching simulation's ANGLE coordinate ranges
    """
    # Choose centering routine (your originals kept):
    if detectorDistBeamtime == 'feb':
        tif_data = center_img(real_data(filename, directory))
    else:
        tif_data = center_img2(real_data(filename, directory))

    sample = get_sampleTest()
    sim = get_simulation_2D(sample, detectorDistBeamtime, angle, ROI_deg=ROI_deg, beamstop_deg=beamstop_deg)
    result = sim.simulate()

    # Use ANGLE axes (φ, α) instead of Q
    data_axes_angles = get_axes_limits(result, ba.Coords_ANGLES)

    save_name = os.path.join(directory, filename.replace(".tif", ".npz"))
    save_npz_data(save_name, tif_data, data_axes_angles)

# ---------------------------- Axes extraction --------------------------------
def get_axes_limits(result, units):
    """
    Returns axes range as expected by imshow:
      [x_min, x_max, y_min, y_max] where x=phi (deg), y=alpha (deg) for ANGLES.
    """
    limits = []
    for i in range(result.rank()):
        ami, ama = result.axisMinMax(i, units)
        assert ami < ama, f'Invalid axis {i}: {ami} .. {ama}'
        limits.extend([ami, ama])
    return limits

# ----------------------- Image centering (unchanged) -------------------------
def center_img(img):
    pixelX_cen = 2075.96734114306
    pixelY_cen = 2048 + 2048 - 1915.635837361077
    angle = -1
    h, w = img.shape
    current_center_x, current_center_y = w // 2, h // 2
    shift_x = current_center_x - pixelX_cen
    shift_y = current_center_y - pixelY_cen
    centered_image = shift(img, shift=(shift_y, shift_x))
    rotated_image  = rotate(centered_image, angle, reshape=False)
    return rotated_image

def center_img2(img):
    # using calibrated mm -> pixel conversion for beam center
    xpos_mm = 152.048
    ypos_mm = 140.305
    pixelX_cen = xpos_mm / rayonix_pixel_size
    pixelY_cen = 2048 + 2048 - ypos_mm / rayonix_pixel_size
    angle = 0.4
    h, w = img.shape
    current_center_x, current_center_y = w // 2, h // 2
    shift_x = current_center_x - pixelX_cen
    shift_y = current_center_y - pixelY_cen
    centered_image = shift(img, shift=(shift_y, shift_x))
    rotated_image  = rotate(centered_image, angle, reshape=False)
    return rotated_image

# -------------------- Slicing & integration in ANGLE space -------------------
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_slices(arrayData, axesLimits, horiz_slice=None, vert_slice=None, desiredRange=None):
    """
    Slice a 2D array using ANGLE axes.
    - axesLimits = [phi_min, phi_max, alpha_min, alpha_max] (deg)
    - vert_slice (deg)  : pick a vertical cut at φ = vert_slice (returns profile vs α)
    - horiz_slice (deg) : pick a horizontal cut at α = horiz_slice (returns profile vs φ)
    If both are given, only vert_slice is used.
    Returns: (x, y) suitable for plt.plot(x, y)
    """
    alpha_min, alpha_max = axesLimits[2], axesLimits[3]
    phi_min,   phi_max   = axesLimits[0], axesLimits[1]

    nrows, ncols = np.shape(arrayData)  # rows=α axis, cols=φ axis (imshow default)
    # We will keep convention consistent with imshow extents:
    # columns correspond to X axis (phi), rows to Y axis (alpha)

    if vert_slice is not None:
        # Fix φ = vert_slice → take a column; x becomes α (vertical axis)
        phi_vals = np.linspace(phi_min, phi_max, ncols)
        col = find_nearest(phi_vals, vert_slice)
        y_vals = arrayData[:, col]          # all alphas at fixed phi
        x_vals = np.linspace(alpha_min, alpha_max, nrows)
        # desiredRange applies to x (alpha) bounds:
        if desiredRange is not None:
            lo = find_nearest(x_vals, desiredRange[0])
            hi = find_nearest(x_vals, desiredRange[1])
            x_vals = x_vals[lo:hi]
            y_vals = y_vals[lo:hi]
        return x_vals, y_vals

    elif horiz_slice is not None:
        # Fix α = horiz_slice → take a row; x becomes φ (horizontal axis)
        alpha_vals = np.linspace(alpha_min, alpha_max, nrows)
        row = find_nearest(alpha_vals, horiz_slice)
        y_vals = arrayData[row, :]          # all phis at fixed alpha
        x_vals = np.linspace(phi_min, phi_max, ncols)
        if desiredRange is not None:
            lo = find_nearest(x_vals, desiredRange[0])
            hi = find_nearest(x_vals, desiredRange[1])
            x_vals = x_vals[lo:hi]
            y_vals = y_vals[lo:hi]
        return x_vals, y_vals

    else:
        raise ValueError("Provide horiz_slice (alpha deg) or vert_slice (phi deg).")

def integrate_plt_slices(start, stop, data, axLim, labelname='', num=0, horiz_slice=None, vert_slice=None, normalize=False, desiredRange=None):
    """
    Integrate multiple slices between [start, stop] in ANGLES.
    - If horiz_slice is not None: integrate rows at α from start..stop
    - If vert_slice  is not None: integrate cols at φ from start..stop
    """
    if num <= 0:
        raise ValueError("num must be > 0 for integration.")

    inc = abs(stop - start) / num
    x_sum, y_sum = 0.0, 0.0

    if horiz_slice is not None:
        # integrate across α
        for i in range(num):
            x, y = plot_slices(data, axesLimits=axLim, horiz_slice=start + inc * i, desiredRange=desiredRange)
            x_sum += x
            y_sum += y
        x_sum /= num
        y_sum /= num

    elif vert_slice is not None:
        # integrate across φ
        for i in range(num):
            x, y = plot_slices(data, axesLimits=axLim, vert_slice=start + inc * i, desiredRange=desiredRange)
            x_sum += x
            y_sum += y
        x_sum /= num
        y_sum /= num

    else:
        raise ValueError("Specify horiz_slice (alpha) or vert_slice (phi).")

    if normalize:
        y_max = np.max(y_sum)
        if y_max > 0:
            y_sum = y_sum / y_max
    return x_sum, y_sum

def lineScan(data, slice_bot, slice_top, axesLimits, pixel_inc=1, along='alpha'):
    """
    Plot a family of angle slices with a colorbar.
    - along='alpha'  → multiple horizontal cuts (constant α), x-axis=φ
    - along='phi'    → multiple vertical cuts (constant φ), x-axis=α
    """
    phi_min, phi_max, alpha_min, alpha_max = axesLimits
    nrows, ncols = data.shape

    if along == 'alpha':
        # step size in α (deg) per pixel
        d_alpha = (alpha_max - alpha_min) / nrows
        inc = d_alpha * pixel_inc
        num = int((slice_top - slice_bot) / inc)
        # color scale vs α value
        norm_all = mcolors.Normalize(vmin=slice_bot, vmax=slice_bot + inc*num)
        cmap = cm.jet

        plt.title(r'Horizontal Slices (constant $\alpha_f$)')
        for i in range(num):
            alpha = slice_bot + inc * i
            x, y = plot_slices(data, axesLimits=axesLimits, horiz_slice=alpha)
            plt.plot(x, y, label=rf'$\alpha_f$: {alpha:.4f}°', color=cmap(norm_all(alpha)))

        sm = cm.ScalarMappable(norm=norm_all, cmap=cmap)
        sm.set_array([slice_bot, slice_top])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(r'$\alpha_f$ (deg)', loc='top', labelpad=-50, rotation=0)
        plt.xlabel(r'$\varphi_f$ (deg)')
        plt.ylabel('Intensity')
        plt.yscale('log')

    elif along == 'phi':
        # step size in φ (deg) per pixel
        d_phi = (phi_max - phi_min) / ncols
        inc = d_phi * pixel_inc
        num = int((slice_top - slice_bot) / inc)
        norm_all = mcolors.Normalize(vmin=slice_bot, vmax=slice_bot + inc*num)
        cmap = cm.jet

        plt.title(r'Vertical Slices (constant $\varphi_f$)')
        for i in range(num):
            phi = slice_bot + inc * i
            x, y = plot_slices(data, axesLimits=axesLimits, vert_slice=phi)
            plt.plot(x, y, label=rf'$\varphi_f$: {phi:.4f}°', color=cmap(norm_all(phi)))

        sm = cm.ScalarMappable(norm=norm_all, cmap=cmap)
        sm.set_array([slice_bot, slice_top])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(r'$\varphi_f$ (deg)', loc='top', labelpad=-50, rotation=0)
        plt.xlabel(r'$\alpha_f$ (deg)')
        plt.ylabel('Intensity')
        plt.yscale('log')

    else:
        raise ValueError("along must be 'alpha' or 'phi'.")

# ------------------------- Simulation conveniences ---------------------------
def get_simulation_line(sample_model,
                        detectorDistBeamtime,
                        angle,            # α_i in deg
                        center_horizontal_slice_values=None,  # list of α centers (deg)
                        center_vertical_slice_values=None,    # list of φ centers (deg)
                        number_slices=1,
                        ROI_deg=None,
                        beamIntensity=8e12,
                        resolution=False,
                        divergence=False,
                        oneThread=False,
                        bounds_phi=None,    # optional (phi_low, phi_high) deg
                        bounds_alpha=None,   # optional (alpha_low, alpha_high) deg
                        background = 23
                        ):
    """
    Build a simulation with thin angle bands:
      - horizontal bands  around given α centers (constant α slices)
      - vertical bands    around given φ centers (constant φ slices)
    Bounds restrict the band lengths (use ROI otherwise).
    """
    distance = _select_distance(detectorDistBeamtime)
    alpha_i_deg = float(angle)

    beam = ba.Beam(beamIntensity, wavelength, alpha_i_deg*deg)
    
    detector = create_detector(distance, resolution)
    
    sim = ba.ScatteringSimulation(beam, sample_model, detector)
    sim.options().setIncludeSpecular(False)
    sim.setBackground(ba.ConstantBackground(background))
    sim.options().setUseAvgMaterials(True)
    if divergence:
        sim.addParameterDistribution(ba.ParameterDistribution.BeamInclinationAngle,
                                     ba.DistributionGaussian(alpha_i_deg*deg, 0.016*deg, 7, 2))
        sim.addParameterDistribution(ba.ParameterDistribution.BeamAzimuthalAngle,
                                     ba.DistributionGaussian(0*deg, 0.042*deg, 7, 2))


    # ROI: if None, use full detector angles; otherwise pass given angles
    if ROI_deg is None:
        phi_min, phi_max, alp_min, alp_max = _rect_to_angle_extents(distance)
        sim.detector().setRegionOfInterest(phi_min*deg, alp_min*deg, phi_max*deg, alp_max*deg)
    else:
        phi1, a1, phi2, a2 = ROI_deg
        # ensure ordering
        phi_lo, phi_hi = min(phi1, phi2), max(phi1, phi2)
        alp_lo,   alp_hi   = min(a1, a2), max(a1, a2)
        sim.detector().setRegionOfInterest(phi_lo*deg, alp_lo*deg, phi_hi*deg, alp_hi*deg)



    # Thin band widths in angle units based on pixel sizes
    dphi_deg, dalp_deg = _angular_pixel_sizes(distance)
    sim.detector().maskAll()

    # Bounds
    if bounds_phi is None:   bounds_phi   = (phi_lo, phi_hi)
    if bounds_alpha is None: bounds_alpha = (alp_lo, alp_hi)

    # Horizontal bands: α around each center; φ spans bounds_phi
    if center_horizontal_slice_values is not None:
        for a_center in center_horizontal_slice_values:
            a1b = (a_center - number_slices * dalp_deg) * deg
            a2b = (a_center + number_slices * dalp_deg) * deg
            p1b = bounds_phi[0] * deg
            p2b = bounds_phi[1] * deg
            sim.detector().addMask(ba.Rectangle(p1b, a1b, p2b, a2b), False)

    # Vertical bands: φ around each center; α spans bounds_alpha
    if center_vertical_slice_values is not None:
        for p_center in center_vertical_slice_values:
            p1b = (p_center - number_slices * dphi_deg) * deg
            p2b = (p_center + number_slices * dphi_deg) * deg
            a1b = bounds_alpha[0] * deg
            a2b = bounds_alpha[1] * deg
            sim.detector().addMask(ba.Rectangle(p1b, a1b, p2b, a2b), False)
    if oneThread:
        sim.options().setNumberOfThreads(1)
    
    return sim

# -------------------- Graphing experimental data in ANGLES -------------------
def graph_experiment_detectorSpace(experiment_file_name: str, experiment_directory: str,
                                   detectorDistBeamtime=None, angle=None, ROI_deg=None):
    """
    Plot experimental image in ANGLE coordinates (φ, α) using the simulation's ANGLE axes.
    """
    # Load experimental npz (already stored with ANGLE axes by tifToNpzConversion)
    realData_npArray, angle_axes = load_npz_data(experiment_file_name, experiment_directory)
    # If you want to re-derive axes from a fresh simulation (same ROI), do:
    sim = get_simulation_2D(get_sampleTest(), detectorDistBeamtime, angle, ROI_deg=ROI_deg)
    result = sim.simulate()
    detectorSpaceAxes = get_axes_limits(result, ba.Coords_ANGLES)

    # Your external plotting helper (unchanged API)
    from GISAXS_Analysis import Graphing_Analysis as graphing
    graphing.plot2D(realData=realData_npArray,
                    realDat_axes=detectorSpaceAxes,
                    graphed_axes=detectorSpaceAxes,
                    zlim=[22, 50000])

# ----------------------------- Misc utilities --------------------------------
def normalize_array(arr):
    arr_max = np.max(arr)
    return arr / arr_max if arr_max > 0 else arr

# ------------------------------ (Optional) -----------------------------------
# If you ever need to convert between q and angles for reporting, here are helpers:
_lambda_nm = float(wavelength)     # nm
_k = 2.0 * np.pi / _lambda_nm      # 1/nm

def phi_from_qy_deg(qy_1_per_nm):
    return np.degrees(np.arcsin(np.clip(qy_1_per_nm / _k, -1.0, 1.0)))

def alpha_from_qz_deg(qz_1_per_nm, alpha_i_deg):
    return np.degrees(np.arcsin(np.clip(qz_1_per_nm / _k - np.sin(np.radians(alpha_i_deg)), -1.0, 1.0)))
# ============================================================================