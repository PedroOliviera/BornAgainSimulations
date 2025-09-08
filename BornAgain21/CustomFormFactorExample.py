#!/usr/bin/env python3
"""
Custom form factor in DWBA.
"""
import cmath
import bornagain as ba
from bornagain import ba_plot as bp, deg, angstrom, nm


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

    def __init__(self, L, H):
        ba.IFormFactor.__init__(self)
        # parameters describing the form factor
        self.L = L
        self.H = H

    def clone(self):
        """
        IMPORTANT NOTE:
        The clone method needs to call transferToCPP() on the cloned object
        to transfer the ownership of the clone to the cpp code
        """
        cloned_ff = CustomFormFactor(self.L, self.H)
        cloned_ff.transferToCPP()
        return cloned_ff

    def formfactor(self, q):
        qzhH = 0.5*q.z()*self.H
        qxhL = 0.5*q.x()*self.L
        qyhL = 0.5*q.y()*self.L
        return 0.5*self.H*self.L**2*cmath.exp(complex(0., 1.)*qzhH)*\
               sinc(qzhH)*(sinc(0.5*qyhL)*(sinc(qxhL)-0.5*sinc(0.5*qxhL))+\
               sinc(0.5*qxhL)*sinc(qyhL))

    def spanZ(self, rotation):
        return ba.Span(0, self.H)


def get_sample():
    """
    A sample with particles, having a custom form factor, on a substrate.
    """
    # defining materials
    vacuum = ba.RefractiveMaterial("Vacuum", 0, 0)
    material_substrate = ba.RefractiveMaterial("Substrate", 6e-6, 2e-8)
    material_particle = ba.RefractiveMaterial("Particle", 6e-4, 2e-8)

    # collection of particles
    ff = CustomFormFactor(20*nm, 15*nm)
    particle = ba.Particle(material_particle, ff)
    particle_layout = ba.ParticleLayout()
    particle_layout.addParticle(particle)
    vacuum_layer = ba.Layer(vacuum)
    vacuum_layer.addLayout(particle_layout)
    substrate_layer = ba.Layer(material_substrate)

    # assemble sample
    sample = ba.MultiLayer()
    sample.addLayer(vacuum_layer)
    sample.addLayer(substrate_layer)
    return sample


def get_simulation(sample):
    beam = ba.Beam(1e9, 1*angstrom, 0.2*deg)
    n = 100
    det = ba.SphericalDetector(n, -1*deg, 1*deg, n, 0, 2*deg)
    simulation = ba.ScatteringSimulation(beam, sample, det)
    simulation.options().setNumberOfThreads(
        1)  # deactivate multithreading (why?)
    return simulation


if __name__ == '__main__':
    sample = get_sample()
    simulation = get_simulation(sample)
    result = simulation.simulate()
    bp.plot_simulation_result(result)