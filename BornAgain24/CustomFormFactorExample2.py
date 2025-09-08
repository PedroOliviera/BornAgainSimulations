#!/usr/bin/env python3
"""
Custom form factor in DWBA.
"""
import cmath
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom, nm
import math, cmath
from scipy.special import j0  # Bessel J0
from scipy.integrate import quad

class CosineCapFormFactor:
    """
    Axisymmetric raised half-cosine cap (base radius R, peak height H):
        z(r) = 0.5*H*[1 + cos(pi*r/R)],  0 <= r <= R;  z = 0 otherwise.

    Returns particle form-factor amplitude (no Δρ):
        F(q) = ∭_V e^{i q·r} dV
    """

    def __init__(self, R, H, *, epsabs=1e-9, epsrel=1e-7, limit=200, qz_eps=1e-12):
        self.R = float(R)
        self.H = float(H)
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit
        self.qz_eps = qz_eps
        self._two_pi = 2.0 * math.pi
        self._pi_over_R = math.pi / self.R

    def _z_of_r(self, r: float) -> float:
        return 0.5 * self.H * (1.0 + math.cos(self._pi_over_R * r))

    def formfactor(self, q: "C3"):
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

    def volume(self) -> float:
        # Exact V = 2π ∫_0^R r z(r) dr = H R^2 * (π/2 - 2/π)
        return self.H * self.R * self.R * (math.pi/2.0 - 2.0/math.pi)

    def spanZ(self, rotation):
        # for BA's layer placement helpers
        return ba.Span(0.0, self.H)

def get_sample():
    """
    Sample with particles, having a custom formfactor, on a substrate.
    """

    # materials
    vacuum = ba.RefractiveMaterial("Vacuum", 0, 0)
    material_substrate = ba.RefractiveMaterial("Substrate", 6e-6, 2e-8)
    material_particle = ba.RefractiveMaterial("Particle", 6e-4, 2e-8)

    # collection of particles
    ff = CosineCapFormFactor(20*nm, 6*nm)
    particle = ba.Particle(material_particle, ff)

    vacuum_layer = ba.Layer(vacuum)
    vacuum_layer.depositParticle(0.01, particle)
    substrate_layer = ba.Layer(material_substrate)

    """ NOTE:
    Slicing of custom formfactor is not possible.
    all layers must have number of slices equal to 1.
    It is a default situation; otherwise use
    ```
    my_layer.setNumberOfSlices(1)
    ```

    Furthermore, a custom particle should not cross layer boundaries;
    that is, the z-span should be within a single layer
    """

    # assemble sample
    sample = ba.Sample()
    sample.addLayer(vacuum_layer)
    sample.addLayer(substrate_layer)
    return sample


def get_simulation(sample):
    beam = ba.Beam(1e9, 1*angstrom, 0.14*deg)
    n = 100
    det = ba.SphericalDetector(n, -1*deg, 1*deg, n, 0, 2*deg)
    simulation = ba.ScatteringSimulation(beam, sample, det)

    # Deactivate multithreading:
    # Currently BornAgain cannot access the Python interpreter
    # from a multi-threaded C++ function
    simulation.options().setNumberOfThreads(1)

    return simulation


if __name__ == '__main__':
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    bp.plot_datafield(result)
    bp.plt.show()