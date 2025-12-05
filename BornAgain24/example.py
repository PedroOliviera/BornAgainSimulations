#!/usr/bin/env python3
"""
Dilute film of small spheres
"""
import bornagain as ba
from bornagain import ba_plot as bp, deg, nm


def get_sample(approximation):
    # Materials
    material_particle = ba.RefractiveMaterial("Particle", 0.0006, 2e-08)
    material_substrate = ba.RefractiveMaterial("Substrate", 6e-06, 2e-08)
    vacuum = ba.RefractiveMaterial("Vacuum", 0, 0)

    # Particles
    ff = ba.Sphere(4*nm)
    particle = ba.Particle(material_particle, ff)

    # Layers
    layer_1 = ba.Layer(vacuum)
    layer_2 = ba.Layer(material_substrate, 30*nm)
    layer_3 = ba.Layer(material_substrate)
    layer_2.plugLiquid(.002, particle, approximation)

    # Sample
    sample = ba.Sample()
    sample.addLayer(layer_1)
    sample.addLayer(layer_2)
    sample.addLayer(layer_3)

    return sample


def get_simulation(sample):
    beam = ba.Beam(1e9, 0.1*nm, 0.2*deg)
    n = 100
    detector = ba.SphericalDetector(1, -1*deg, 1*deg, n, 0., 2*deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    return simulation


if __name__ == '__main__':
    samples = [
        get_sample(ba.Random3D_PY),
        get_sample(ba.Random3D_Dilute),
    ]
    results = [ get_simulation(sample).simulate() for sample in samples ]
    for r in results:
        bp.plot_datafield(r.flat())
    bp.plt.show()
