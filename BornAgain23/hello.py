print('hello world')
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
exp_npz_file = "33_15deg.npz"     # saved with Q axes: [qy_min,qy_max,qz_min,qz_max]
alpha_i_deg  = 0.15

exp_arr, exp_axes_q = g.load_npz_data(exp_npz_file, exp_dir)

axes_lims = g.extent_phi_alpha_from_image(exp_arr, 'feb', alpha_i_deg=alpha_i_deg)

fig, ax2 = plt.subplots(1, 1, figsize=(12,5), constrained_layout=True)

im2 = ax2.imshow(exp_arr, origin= 'lower', extent=axes_lims,
                 aspect="auto")
ax2.set_title("Simulation (radial paracrystal, 20 nm spheres)")
ax2.set_xlabel(r"$\varphi_f$ (deg)"); ax2.set_ylabel(r"$\alpha_f$ (deg)")
fig.colorbar(im2, ax=ax2, label="Intensity (a.u.)")

plt.show()