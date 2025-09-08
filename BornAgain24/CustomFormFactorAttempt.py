#!/usr/bin/env python3
"""
Custom form factor in DWBA.
"""
import cmath
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom, nm
import math

import math, cmath

import math
import cmath
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
    
    # required by BA when structure factors are used
    def radialExtension(self):
        # axisymmetric cap of base radius R ⇒ radial extent is R
        return self.R
    
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
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP",3E-6, 2.35E-9) #2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    # collection of particles
    ff = CosineCapFormFactor(25*nm, 15*nm)
    particle = ba.Particle(material_PS, ff)

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, 214.8*nm)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

    omega_order = 9*nm
    spacing = 60*nm

    layer_vac.depositParticle(0.001, particle)

    #particle = ba.Particle(material_PS, ba.Sphere(5*nm))

    # Interference Functions
    #iff = ba.InterferenceRadialParacrystal(spacing, 250*nm)
    #iff_pdf = ba.Profile1DGauss(omega_order)
    #iff.setProbabilityDistribution(iff_pdf)
    #iff.setKappa(1.5) #size-distribution model

    #surface_layout = ba.StructuredLayout(iff)
    #surface_layout.setTotalParticleSurfaceDensity(0.0265)
    #surface_layout.addParticle(particle, 1)
    #layer_vac.addStruct(surface_layout)

    # assemble sample
    sample = ba.Sample()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_Top)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

def get_simulation(sample):
    beam = ba.Beam(1e9, 1.25916*angstrom, 0.13*deg)
    n = 100
    det = ba.SphericalDetector(n, 0*deg, 0.4*deg, n, 0, 1*deg)
    simulation = ba.ScatteringSimulation(beam, sample, det)

    # Deactivate multithreading:
    # Currently BornAgain cannot access the Python interpreter
    # from a multi-threaded C++ function
    simulation.options().setNumberOfThreads(1)

    return simulation

def get_sample_hemi():
    """
    Sample with particles, having a custom formfactor, on a substrate.
    """

    # materials
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    # collection of particles
    ff = ba.SpheroidalSegment(25 * nm, 7* nm, 0, 7 * nm)
    particle = ba.Particle(material_PS, ff)

    # Define layers
    layer_vac = ba.Layer(material_Vacuum)
    layer_PS_Top = ba.Layer(material_PS, 214.8*nm)
    layer_SiO2 = ba.Layer(material_SiO2, 2*nm)
    layer_Si = ba.Layer(material_Si_Sub)

    omega_order = 9*nm
    spacing = 60*nm

    layer_vac.depositParticle(0.01, particle)

    #particle = ba.Particle(material_PS, ba.Sphere(5*nm))

    # Interference Functions
    #iff = ba.InterferenceRadialParacrystal(spacing, 250*nm)
    #iff_pdf = ba.Profile1DGauss(omega_order)
    #iff.setProbabilityDistribution(iff_pdf)
    #iff.setKappa(1.5) #size-distribution model

    #surface_layout = ba.StructuredLayout(iff)
    #surface_layout.setTotalParticleSurfaceDensity(0.0265)
    #surface_layout.addParticle(particle, 1)
    #layer_vac.addStruct(surface_layout)

    # assemble sample
    sample = ba.Sample()
    sample.addLayer(layer_vac)
    sample.addLayer(layer_PS_Top)
    sample.addLayer(layer_SiO2)
    sample.addLayer(layer_Si)
    return sample

if __name__ == '__main__':
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    trafo = ba.FrameTrafo.ScatteringToQ(1.25916*angstrom, 0.13*deg)
    res = trafo.transformedDatafield(result)
    bp.plt.figure()
    bp.plot_datafield(res)
    sample = get_sample_hemi()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    trafo = ba.FrameTrafo.ScatteringToQ(1.25916*angstrom, 0.13*deg)
    res = trafo.transformedDatafield(result)
    bp.plt.figure()
    bp.plot_datafield(res)
    bp.plt.show()