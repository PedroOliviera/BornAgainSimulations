print("START!", __file__, flush=True)
from GISAXS_Analysis import GISAXS_setup_v23 as g
import bornagain as ba
from bornagain import deg, nm
from bornagain.numpyutil import Arrayf64Converter as dac
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
print('start')
# ---------- USER INPUTS ----------
exp_dir      = r"C:\BornAgainSimulations\data\exp-npz"
exp_npz_file = "35_15deg.npz"     # saved with Q axes: [qy_min,qy_max,qz_min,qz_max]
alpha_i_deg  = 0.15
beamtime     = "feb"
ROI_deg      = (-1, -2, 2, 2)          # (phi_min, alpha_min, phi_max, alpha_max)
print('here')
# ---------- SAMPLE (BA23-compliant) ----------
def sample_radial_paracrystal(radius_nm=20.0, d_mean_nm=44.3, omega_nm=7.0,
                              damping_length_nm=1000.0, density_nm2=0.01):
    m_particle  = ba.RefractiveMaterial("PS",     2.5e-6, 2.3e-9)
    m_substrate = ba.RefractiveMaterial("Si Sub", 5.0e-6, 7.8e-8)
    p = ba.Particle(m_particle, ba.Sphere(radius_nm*nm))
    layout = ba.ParticleLayout(); layout.addParticle(p)
    iff = ba.InterferenceRadialParacrystal(d_mean_nm*nm, damping_length_nm*nm)
    iff.setProbabilityDistribution(ba.Profile1DGauss(omega_nm*nm))
    layout.setInterference(iff)
    layout.setTotalParticleSurfaceDensity(density_nm2)
    top = ba.Layer(ba.Vacuum()); top.addLayout(layout)
    sub = ba.Layer(m_substrate)
    s = ba.Sample(); s.addLayer(top); s.addLayer(sub)
    return s
print('here')
sample = sample_radial_paracrystal()
print('here')
# ---------- SIMULATE ----------
sim = g.get_simulation_2D(sample_model=sample,
                          detectorDistBeamtime=beamtime,
                          angle=alpha_i_deg,
                          beamIntensity=8e12,
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


# ---------- PLOT SIDE-BY-SIDE ----------
def robust_limits(A):
    a = np.asarray(A, float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0: return 1e-1, 1.0
    return np.percentile(a, 5), np.percentile(a, 99)

vmin_e, vmax_e = robust_limits(exp_arr)
vmin_s, vmax_s = robust_limits(I_sim)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)

im1 = ax1.imshow(exp_arr, extent=exp_axes,
                 aspect="auto", norm=LogNorm(vmin=vmin_e, vmax=vmax_e))
ax1.set_title("Experiment (resampled to φ/α)")
ax1.set_xlabel(r"$\varphi_f$ (deg)"); ax1.set_ylabel(r"$\alpha_f$ (deg)")
fig.colorbar(im1, ax=ax1, label="Intensity (a.u.)")

im2 = ax2.imshow(I_sim, origin= 'lower', extent=extent_angles,
                 aspect="auto", norm=LogNorm(vmin=vmin_s, vmax=vmax_s))
ax2.set_title("Simulation (radial paracrystal, 20 nm spheres)")
ax2.set_xlabel(r"$\varphi_f$ (deg)"); ax2.set_ylabel(r"$\alpha_f$ (deg)")
fig.colorbar(im2, ax=ax2, label="Intensity (a.u.)")

ax1.set_ylim(0,2)
ax2.set_ylim(0,2)
ax1.set_xlim(0,2)
ax2.set_xlim(0,2)

plt.show()

# ---------- OPTIONAL: a vertical slice at φ=0 ----------
phi_grid = np.linspace(phi_min, phi_max, n_phi)
alpha_grid = np.linspace(a_min, a_max, n_alpha)
col = int(np.argmin(np.abs(phi_grid - 0.0)))
y_exp = exp_arr[:, col]
y_sim = I_sim[:, col]

plt.figure(figsize=(6,4))
plt.semilogy(alpha_grid, y_exp, label="Exp @ φ=0°")
plt.semilogy(alpha_grid, y_sim, label="Sim @ φ=0°")
plt.xlabel(r"$\alpha_f$ (deg)"); plt.ylabel("Intensity (a.u.)")
plt.legend(); plt.tight_layout(); plt.show()
