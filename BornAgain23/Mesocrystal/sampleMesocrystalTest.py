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
import math
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


def fullrotation_get_sample():
    # --- Params ---
    R = 15.43*nm          # in-plane (equatorial) semi-axis
    Z = 5.0*nm            # polar semi-axis  -> height = 10 nm
    a1 = R3(2*R, 0.0*nm, 0.0*nm)
    a2 = R3(R,  (3**0.5)*R, 0.0*nm)
    a3 = R3(R,  (3**0.5)*R/3.0, (8.0/3.0)**0.5 * Z)  # Option-B: (R, √3 R/3, √(8/3) Z)

    z_shift = R3(0.0*nm, 0.0*nm, -1100.0*nm)
    z_angles = range(0, 120, 3)   # 0,10,...,110
    x_angles = range(0, 22, 2)     # 0,2,...,20
    y_angles = range(0, 22, 2)     # 0,2,...,20

    # --- Materials ---
    material_Core      = ba.RefractiveMaterial("Core",      3e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  4e-06, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6.0e-06, 2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,     0.0)

    # --- Form factors (shared) ---
    ff_spheroid = ba.Spheroid(R, Z*2)      # height = 2Z = 10 nm
    ff_envelope = ba.Sphere(100*nm) 

    # --- Particle & lattice (shared) ---
    particle = ba.Particle(material_Particle, ff_spheroid)
    lattice  = ba.Lattice3D(a1, a2, a3)

    # --- Build all oriented mesocrystals ---
    layout = ba.ParticleLayout()
    for z_deg in z_angles:
        for x_deg in x_angles:
            for y_deg in y_angles:
                crystal = ba.Crystal(particle, lattice)
                meso = ba.Mesocrystal(crystal, ff_envelope)

                # Apply rotations in a clear order: X, then Y, then Z
                if x_deg: meso.rotate(ba.RotationX(x_deg*deg))
                if y_deg: meso.rotate(ba.RotationY(y_deg*deg))
                if z_deg: meso.rotate(ba.RotationZ(z_deg*deg))

                meso.translate(z_shift)
                layout.addParticle(meso, 1.0)

    # Normalize by weights internally; set overall surface density:
    layout.setTotalParticleSurfaceDensity(1.054701e-05)

    # --- Layers & sample ---
    layer_top  = ba.Layer(material_Vacuum)
    layer_core = ba.Layer(material_Core, 1200*nm)
    layer_core.addLayout(layout)
    layer_sub  = ba.Layer(material_Substrate)

    sample = ba.Sample()
    sample.addLayer(layer_top)
    sample.addLayer(layer_core)
    sample.addLayer(layer_sub)
    return sample

def fullrotation_disconnect_ff_get_sample():
    # --- Params ---
    R = 31*nm          # in-plane (equatorial) semi-axis
    Z = 17*nm            # polar semi-axis  -> height = 10 nm
    a1 = R3(2*R, 0.0*nm, 0.0*nm)
    a2 = R3(R,  (3**0.5)*R, 0.0*nm)
    a3 = R3(R,  (3**0.5)*R/3.0, (8.0/3.0)**0.5 * Z)  # Option-B: (R, √3 R/3, √(8/3) Z)

    z_shift = R3(0.0*nm, 0.0*nm, -700.0*nm)
    z_angles = range(0, 120, 10)   # 0,10,...,110
    x_angles = range(0, 22, 5)     # 0,2,...,20
    y_angles = range(0, 22, 5)     # 0,2,...,20

    # --- Materials ---
    material_Core      = ba.RefractiveMaterial("Core",      3e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  4e-06, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6.0e-06, 2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,     0.0)

    # --- Form factors (shared) ---
    ff_spheroid = ba.Sphere(15.43*nm)      # height = 2Z = 10 nm
    ff_envelope = ba.Box(500*nm, 500*nm, 215*nm) 

    # --- Particle & lattice (shared) ---
    particle = ba.Particle(material_Particle, ff_spheroid)
    lattice  = ba.Lattice3D(a1, a2, a3)

    # --- Build all oriented mesocrystals ---
    layout = ba.ParticleLayout()
    for z_deg in z_angles:
        for x_deg in x_angles:
            for y_deg in y_angles:
                crystal = ba.Crystal(particle, lattice)
                meso = ba.Mesocrystal(crystal, ff_envelope)

                # Apply rotations in a clear order: X, then Y, then Z
                if x_deg: meso.rotate(ba.RotationX(x_deg*deg))
                if y_deg: meso.rotate(ba.RotationY(y_deg*deg))
                if z_deg: meso.rotate(ba.RotationZ(z_deg*deg))

                meso.translate(z_shift)
                layout.addParticle(meso, 1.0)

    # Normalize by weights internally; set overall surface density:
    layout.setTotalParticleSurfaceDensity(1.054701e-05)

    # --- Layers & sample ---
    layer_top  = ba.Layer(material_Vacuum)
    layer_core = ba.Layer(material_Core, 1200*nm)
    layer_core.addLayout(layout)
    layer_sub  = ba.Layer(material_Substrate)

    sample = ba.Sample()
    sample.addLayer(layer_top)
    sample.addLayer(layer_core)
    sample.addLayer(layer_sub)
    return sample

def justff_get_sample():
    # --- Params ---
    R = 31*nm          # in-plane (equatorial) semi-axis
    Z = 16*nm            # polar semi-axis  -> height = 10 nm
    z_shift = R3(0.0*nm, 0.0*nm, -120.0*nm)

    # --- Materials ---
    material_Core      = ba.RefractiveMaterial("Core",      3e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  4e-06, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6.0e-06, 2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,     0.0)

    # --- Form factors (shared) ---
    ff_spheroid = ba.Spheroid(R, Z)      # height = 2Z = 10 nm

    # --- Particle & lattice (shared) ---
    particle = ba.Particle(material_Particle, ff_spheroid)
    particle.translate(z_shift)

    # --- Build all oriented mesocrystals ---
    layout = ba.ParticleLayout()
    
    layout.addParticle(particle, 1.0)

    # Normalize by weights internally; set overall surface density:
    layout.setTotalParticleSurfaceDensity(1.054701e-03)

    # --- Layers & sample ---
    layer_top  = ba.Layer(material_Vacuum)
    layer_core = ba.Layer(material_Core, 230.0*nm)
    layer_core.addLayout(layout)
    layer_sub  = ba.Layer(material_Substrate)

    sample = ba.Sample()
    sample.addLayer(layer_top)
    sample.addLayer(layer_core)
    sample.addLayer(layer_sub)
    return sample

def justff_meso_get_sample():
    # --- Params ---
    R = 15.43*nm          # in-plane (equatorial) semi-axis
    Z = 10.0*nm            # polar semi-axis  -> height = 10 nm
    z_shift = R3(0.0*nm, 0.0*nm, -1100.0*nm)

    # --- Materials ---
    material_Core      = ba.RefractiveMaterial("Core",      3e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  4e-06, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6.0e-06, 2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,     0.0)

    # --- Form factors (shared) ---
    ff_envelope = ba.Sphere(400*nm)     # height = 2Z = 10 nm

    # --- Particle & lattice (shared) ---
    particle = ba.Particle(material_Particle, ff_envelope)
    particle.translate(z_shift)

    # --- Build all oriented mesocrystals ---
    layout = ba.ParticleLayout()
    
    layout.addParticle(particle, 1.0)

    # Normalize by weights internally; set overall surface density:
    layout.setTotalParticleSurfaceDensity(1.054701e-05)

    # --- Layers & sample ---
    layer_top  = ba.Layer(material_Vacuum)
    layer_core = ba.Layer(material_Core, 1200*nm)
    layer_core.addLayout(layout)
    layer_sub  = ba.Layer(material_Substrate)

    sample = ba.Sample()
    sample.addLayer(layer_top)
    sample.addLayer(layer_core)
    sample.addLayer(layer_sub)
    return sample

def get_sample():
    # --- Materials ---
    material_Core      = ba.RefractiveMaterial("Core",      3e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  4e-06, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6.0e-06, 2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,     0.0)

    # --- Form factors (shared) ---
    # NOTE: Spheroid(15.43 nm, 10 nm) means equatorial semi-axis = 15.43 nm and polar semi-axis = 10 nm (height = 20 nm).
    # If you want 10 nm HEIGHT instead, change the second value to 5*nm.
    ff_spheroid = ba.Spheroid(15.43*nm, 10*nm)
    ff_envelope = ba.Sphere(40*nm)

    # --- Particle (shared) ---
    particle = ba.Particle(material_Particle, ff_spheroid)

    # --- Lattice (shared) ---
    a1 = R3(30.86*nm, 0.0*nm,   0.0*nm)
    a2 = R3(15.43*nm, 26.726*nm, 0.0*nm)
    a3 = R3(15.43*nm,  8.909*nm, 8.165*nm)  # for oblate option-B construction
    lattice = ba.Lattice3D(a1, a2, a3)

    # --- Mesocrystals (loop over rotations) ---
    layout = ba.ParticleLayout()
    z_shift = R3(0.0*nm, 0.0*nm, -210.0*nm)
    for i, angle_deg in enumerate(range(0, 120, 10)):  # 0,10,...,110
        # Each mesocrystal gets its own Crystal (reusing shared particle & lattice)
        crystal = ba.Crystal(particle, lattice)
        meso = ba.Mesocrystal(crystal, ff_envelope)
        if angle_deg:
            meso.rotate(ba.RotationZ(angle_deg*deg))
        meso.translate(z_shift)
        layout.addParticle(meso, 1.0)

    layout.setTotalParticleSurfaceDensity(1.054701e-05)

    # --- Layers ---
    layer_top    = ba.Layer(material_Vacuum)
    layer_core   = ba.Layer(material_Core, 230.0*nm)
    layer_core.addLayout(layout)
    layer_sub    = ba.Layer(material_Substrate)

    # --- Sample ---
    sample = ba.Sample()
    sample.addLayer(layer_top)
    sample.addLayer(layer_core)
    sample.addLayer(layer_sub)
    return sample

def get_sample_full_lattice2():
    import math

    # --- Materials (as in your example) ---
    material_Core      = ba.RefractiveMaterial("Core",      0.0002, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  0.0006, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6e-06,  2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,    0.0)

    # --- Spheroid form factor (aligned along lab Z) ---
    # In-plane (equatorial) semi-axis R = 15.43 nm; polar semi-axis Z = 5 nm -> height = 10 nm
    R_nm, Z_nm = 15.43, 5.0
    ff_spheroid = ba.Spheroid(R_nm*nm, Z_nm*nm)

    # --- Option-B primitive vectors (nm) for aligned oblate spheroids (touching) ---
    # a1, a2 define the 2D hex lattice; a3' carries AB shift + single-layer vertical spacing
    a1x, a1y, a1z = 2*R_nm, 0.0, 0.0
    a2x, a2y, a2z = R_nm, math.sqrt(3.0)*R_nm, 0.0
    a3x, a3y, a3z = R_nm, (math.sqrt(3.0)*R_nm)/3.0, math.sqrt(8.0/3.0)*Z_nm

    # --- Build a base 4-layer ABAB stack inside a Compound at positions k * a3' (k = 0..3) ---
    base_compound = ba.Compound()
    for k in range(4):
        p = ba.Particle(material_Particle, ff_spheroid)
        p.translate(R3((k*a3x)*nm, (k*a3y)*nm, (k*a3z)*nm))  # x,y,z shift via a3'
        base_compound.addComponent(p)

    # --- 2D hexagonal lattice built FROM a1, a2 (not hard-coded) ---
    a_len = math.hypot(a1x, a1y)                            # ~30.86 nm
    b_len = math.hypot(a2x, a2y)                            # ~30.86 nm
    dot   = a1x*a2x + a1y*a2y
    gamma_deg = math.degrees(math.acos(dot/(a_len*b_len)))  # ~60°
    phi_deg   = math.degrees(math.atan2(a1y, a1x))          # 0° here

    lattice = ba.BasicLattice2D(a_len*nm, b_len*nm, gamma_deg*deg, phi_deg*deg)

    # --- Interference (paracrystal-like broadening) ---
    iff = ba.Interference2DLattice(lattice)
    iff_pdf = ba.Profile2DCauchy(10*nm, 10*nm, 0*deg)
    iff.setDecayFunction(iff_pdf)

    # --- Orientation sweeps (Z, X, Y), and global z-shift like before ---
    z_angles = range(0, 120, 10)  # 0,10,...,110
    x_angles = range(0, 22, 2)    # 0,2,...,20
    y_angles = range(0, 22, 2)    # 0,2,...,20
    global_shift = R3(0*nm, 0*nm, -200*nm)

    # --- Particle layout: clone + rotate the compound for every (z,x,y) combo ---
    layout = ba.ParticleLayout()
    for z_deg in z_angles:
        for x_deg in x_angles:
            for y_deg in y_angles:
                comp = ba.Compound(base_compound)  # clone the 4-layer stack
                # Apply rotations in a consistent order: X, then Y, then Z
                if x_deg: comp.rotate(ba.RotationX(x_deg*deg))
                if y_deg: comp.rotate(ba.RotationY(y_deg*deg))
                if z_deg: comp.rotate(ba.RotationZ(z_deg*deg))
                comp.translate(global_shift)
                layout.addParticle(comp, 1.0)

    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(0.00288675134595)

    # --- Layers (middle layer has thickness and layout) ---
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_Core, 100*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_Substrate)

    # --- Sample ---
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    return sample


def get_sample_full_lattice():
    # --- Materials (as in your example) ---
    material_Core      = ba.RefractiveMaterial("Core",      3e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  4e-06, 2e-08)
    material_Substrate = ba.RefractiveMaterial("Substrate", 6.0e-06, 2e-08)
    material_Vacuum    = ba.RefractiveMaterial("Vacuum",    0.0,     0.0)


    # --- Spheroid form factor (aligned along lab Z) ---
    # In-plane (equatorial) semi-axis R = 15.43 nm; polar semi-axis Z = 5 nm  -> height = 10 nm
    R_nm, Z_nm = 15.43, 10.0
    ff_spheroid = ba.Spheroid(R_nm*nm, Z_nm*nm)

    # --- HCP Option-B primitive vectors for aligned spheroids (touching) ---
    # a1, a2 = in-plane triangular lattice; a3' = AB lateral shift + single-layer vertical spacing
    a1x, a1y, a1z = 2*R_nm, 0.0, 0.0
    a2x, a2y, a2z = R_nm, math.sqrt(3.0)*R_nm, 0.0
    a3x, a3y, a3z = R_nm, (math.sqrt(3.0)*R_nm)/3.0, math.sqrt(8.0/3.0)*Z_nm  # ensures tangency

    # --- Build a 4-layer ABAB stack inside a Compound at positions k * a3' (k = 0..3) ---
    compound = ba.Compound()
    for k in range(4):
        p = ba.Particle(material_Particle, ff_spheroid)
        p.translate(R3((k*a3x)*nm, (k*a3y)*nm, (k*a3z)*nm))
        compound.addComponent(p)

    # Global z-shift of the whole stack (adjust as needed)
    compound.translate(R3(0*nm, 0*nm, -200*nm))

    # --- 2D hexagonal lattice built FROM a1, a2 (not hard-coded) ---
    a_len = math.hypot(a1x, a1y)                 # ≈ 30.86 nm
    b_len = math.hypot(a2x, a2y)                 # ≈ 30.86 nm
    dot   = a1x*a2x + a1y*a2y
    gamma_deg = math.degrees(math.acos(dot/(a_len*b_len)))  # ≈ 60°
    # Orientation of a1 vs lab-X (here 0 since a1 along x); keep general for completeness
    phi_deg = math.degrees(math.atan2(a1y, a1x))            # 0°

    lattice = ba.BasicLattice2D(a_len*nm, b_len*nm, gamma_deg*deg, phi_deg*deg)

    # --- Interference (optional paracrystal broadening) ---
    iff = ba.Interference2DLattice(lattice)
    iff_pdf = ba.Profile2DCauchy(10*nm, 10*nm, 0*deg)
    iff.setDecayFunction(iff_pdf)

    # --- Particle layout ---
    layout = ba.ParticleLayout()
    layout.addParticle(compound, 1.0)
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(0.00000000000288675134595)

    # --- Layers (thickness on middle layer) ---
    layer_1 = ba.Layer(material_Vacuum)
    layer_2 = ba.Layer(material_Core, 250*nm)
    layer_2.addLayout(layout)
    layer_3 = ba.Layer(material_Substrate)

    # --- Sample ---
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)
    return sample



# ---------- USER INPUTS ----------
exp_dir      = r"C:\BornAgainSimulations\data\exp-npz\feb"
exp_npz_file = "35_15deg.npz"     # saved with Q axes: [qy_min,qy_max,qz_min,qz_max]
alpha_i_deg  = 0.15
beamtime     = "feb"
ROI_deg      = (0, 0, 0.5, 0.5)           # (phi_min, alpha_min, phi_max, alpha_max)

# ---------- SAMPLE (BA23-compliant) ----------
sample = fullrotation_get_sample()
#sample = get_sample_full_lattice()
# ---------- SIMULATE ----------
sim = g.get_simulation_2D(sample_model=sample,
                          detectorDistBeamtime=beamtime,
                          angle=alpha_i_deg,
                          beamIntensity=8e11,
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

plt.show()
'''
sample = justff_get_sample()

# ---------- SIMULATE ----------
sim = g.get_simulation_2D(sample_model=sample,
                          detectorDistBeamtime=beamtime,
                          angle=alpha_i_deg,
                          beamIntensity=8e11,
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

sample = justff_meso_get_sample()

# ---------- SIMULATE ----------
sim = g.get_simulation_2D(sample_model=sample,
                          detectorDistBeamtime=beamtime,
                          angle=alpha_i_deg,
                          beamIntensity=8e11,
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


plt.show()
'''