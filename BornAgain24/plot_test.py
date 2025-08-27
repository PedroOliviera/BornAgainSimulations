#!/usr/bin/env python3
"""
In this example we demonstrate how to plot a simulation result with
axes in different units (nbins, mm, degs and QyQz).
"""
import bornagain as ba
from bornagain import angstrom, ba_plot as bp, deg, nm
from matplotlib import rcParams


def get_sample():
    # Materials
    material_air = ba.RefractiveMaterial("Air", 0, 0)
    material_particle = ba.RefractiveMaterial("Particle", 0.0006, 2e-08)
    material_substrate = ba.RefractiveMaterial("Substrate", 6e-06, 2e-08)

    # Particles
    R = 2.5*nm
    ff = ba.Spheroid(R, R)
    particle = ba.Particle(material_particle, ff)

    # Interference function
    lattice = ba.SquareLattice2D(10*nm, 2*deg)
    interference = ba.Interference2DLattice(lattice)
    interference_pdf = ba.Profile2DCauchy(50*nm, 50*nm, 0)
    interference.setDecayFunction(interference_pdf)

    # Particle layout
    layout = ba.StructuredLayout(interference)
    layout.addParticle(particle)

    # Layers
    l_air = ba.Layer(material_air)
    l_air.addStruct(layout)
    l_substrate = ba.Layer(material_substrate)

    # Sample
    sample = ba.Sample()
    sample.addLayer(l_air)
    sample.addLayer(l_substrate)
    return sample


def get_simulation(sample, wavelength, alpha_i):
    beam = ba.Beam(1e9, wavelength, alpha_i)
    n = 200
    detector = ba.SphericalDetector(n, -1*deg, 1*deg, n, 0, 1*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation

if __name__ == '__main__':
    sample = get_sample()
    wavelength = 0.04*nm
    alpha_i = 0.2*deg
    simulation = get_simulation(sample, wavelength, alpha_i)
    result = simulation.simulate()

    trafo = ba.FrameTrafo.ScatteringToQ(wavelength, alpha_i)
    res2 = trafo.transformedDatafield(result)

    bp.plot_datafield(res2)
    bp.plt.show()