#!/usr/bin/env python3
"""
Custom form factor in DWBA.
"""
import cmath
from scipy.special import j0  # Bessel J0
from scipy.integrate import quad
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom, nm
import math

def sinc(x):
    if abs(x) == 0:
        return 1.
    return cmath.sin(x)/x


class CustomFormFactor(ba.IFormFactor):
    """
    A custom defined form factor.
    The particle is a prism of height H,
    with a base in form of a Greek cross ("plus" sign) with side length L.
    """
    def __init__(self, R, H, *, epsabs=1e-9, epsrel=1e-7, limit=200, qz_eps=1e-12):
        ba.IFormFactor.__init__(self)
        self.R = float(R)
        self.H = float(H)
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit
        self.qz_eps = qz_eps
        self._two_pi = 2.0 * math.pi
        self._pi_over_R = math.pi / self.R

    def clone(self):
        """
        IMPORTANT NOTE:
        The clone method needs to call transferToCPP() on the cloned object
        to transfer the ownership of the clone to the cpp code
        """
        cloned_ff = CustomFormFactor(self.R, self.H)
        cloned_ff.transferToCPP()
        return cloned_ff
    
    def _z_of_r(self, r: float) -> float:
        return 0.5 * self.H * (1.0 + math.cos(self._pi_over_R * r))

    def formfactor(self, q):
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
    
    def radialExtension(self):
        # axisymmetric cap of base radius R ⇒ radial extent is R
        return self.R

    def spanZ(self, rotation):
        return ba.Span(0, self.H)


def get_sample():
    """
    A sample with particles, having a custom form factor, on a substrate.
    """
    # materials
    material_PS = ba.RefractiveMaterial("PS", 2.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP",3E-6, 2.35E-9) #2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08) #7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    # collection of particles
    ff = CustomFormFactor(25*nm, 7*nm)
    particle = ba.Particle(material_PS, ff)
    particle_layout = ba.ParticleLayout()
    particle_layout.addParticle(particle, 1)

    # Radial Interference Functions
    iff = ba.InterferenceRadialParacrystal(50*nm, 1000000*nm) #250
    iff_pdf = ba.Profile1DGauss(9*nm)
    iff.setProbabilityDistribution(iff_pdf)
    iff.setKappa(1.5) #size-distribution model
    # 
    # 2D Paracrystal hexagonal
    #lattice = ba.BasicLattice2D(spacing*nm, spacing*nm, 120*deg, 0) 
    #iff = ba.Interference2DParacrystal(lattice, 0, 100000*nm, 100000*nm)
    #iff.setIntegrationOverXi(True)
    #iff_pdf = ba.Profile2DCauchy(omega_order*nm, omega_order*nm, 0)
    #iff.setProbabilityDistributions(iff_pdf, iff_pdf)

    particle_layout.setInterference(iff)
    particle_layout.setTotalParticleSurfaceDensity(26500000000) #PLAY WITH THIS 0.0265


    vacuum_layer = ba.Layer(material_Vacuum)
    vacuum_layer.addLayout(particle_layout)

    
    polymer_layer = ba.Layer(material_PS)
    oxide_layer = ba.Layer(material_SiO2)
    substrate_layer = ba.Layer(material_Si_Sub)

    # assemble sample
    sample = ba.MultiLayer()
    sample.addLayer(vacuum_layer)
    sample.addLayer(polymer_layer)
    sample.addLayer(oxide_layer)
    sample.addLayer(substrate_layer)
    return sample

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
    ff = ba.HemiEllipsoid(25*nm, 25*nm, 7*nm)
    particle = ba.Particle(material_PS, ff)
    particle_layout = ba.ParticleLayout()
    particle_layout.addParticle(particle)

    vacuum_layer = ba.Layer(material_Vacuum)
    polymer_layer = ba.Layer(material_PS)
    oxide_layer = ba.Layer(material_SiO2)
    substrate_layer = ba.Layer(material_Si_Sub)

    # assemble sample
    sample = ba.MultiLayer()
    sample.addLayer(vacuum_layer)
    vacuum_layer.addLayout(particle_layout)
    sample.addLayer(polymer_layer)
    sample.addLayer(oxide_layer)
    sample.addLayer(substrate_layer)
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


if __name__ == '__main__':
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    bp.plot_simulation_result(result)