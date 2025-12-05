import bornagain as ba
from bornagain import deg, nm, R3


def get_sample():
    # --- Materials ---
    material_Core      = ba.RefractiveMaterial("Core",      3.5e-06, 1e-08)
    material_Particle  = ba.RefractiveMaterial("Particle",  3.7e-06, 2e-08)
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
    z_shift = R3(0.0*nm, 0.0*nm, -220.0*nm)
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
