import bornagain as ba
from bornagain import deg, nm, R3
import numpy as np
from scipy.stats import norm

# --- Experimental Parameters ---
WAVELENGTH = 0.1 * nm            # Beam wavelength (12.4 keV approx)
ALPHA_YONEDA = 0.166 * deg       # Experimental Yoneda angle
PEAK_PHI_EXP = 0.17 * deg        # Experimental lateral peak position

# Estimate lattice constant from the first peak
# D = lambda / phi_peak
AVG_DISTANCE = WAVELENGTH / PEAK_PHI_EXP

def get_sample(core_radius, peak_distance):
    """
    Constructs the sample using ba.Compound for the Core-Shell Particle.
    """
    # 1. Materials
    # Delta (refractive index decrement) estimated from Yoneda position
    delta_poly = (ALPHA_YONEDA**2) / 2.0
    
    m_air = ba.RefractiveMaterial("Air", 0.0, 0.0)
    m_polymer = ba.RefractiveMaterial("PolymerMatrix", delta_poly, 2e-8)
    m_substrate = ba.RefractiveMaterial("Substrate", 7e-6, 2e-7)
    m_core = ba.RefractiveMaterial("Au_Core", 1.2e-4, 1e-7)
    m_shell = ba.RefractiveMaterial("Silica_Shell", 3.0e-6, 2e-8)

    # 2. Form Factors
    shell_thickness = 2.0 * nm
    
    # Core Form Factor
    core_h = core_radius * 2.0
    ff_core = ba.Spheroid(core_radius, core_h)
    particle_core = ba.Particle(m_core, ff_core)
    
    # Shell Form Factor
    shell_R = core_radius + shell_thickness
    shell_h = core_h + (2.0 * shell_thickness)
    ff_shell = ba.Spheroid(shell_R, shell_h)
    particle_shell = ba.Particle(m_shell, ff_shell)

    # 3. Composition (Core-Shell) using ba.Compound (v22/v23)
    particle = ba.Compound()
    
    # Add Shell at (0,0,0) - the reference
    particle.addComponent(particle_shell)
    
    # Add Core shifted up by shell_thickness (to center it inside the shell)
    # kvector_t(x, y, z) defines the relative position
    core_position = R3(0, 0, shell_thickness)
    particle.addComponent(particle_core, core_position)

    # 4. Interference: 1D Radial Paracrystal
    damping_length = peak_distance * 3.0
    interference = ba.InterferenceRadialParacrystal(peak_distance, damping_length)
    
    # Add structural disorder (pdf of the lattice points)
    pdf = ba.Profile1DGauss(peak_distance * 0.15)
    interference.setProbabilityDistribution(pdf)

    # 5. Layout & Layers
    layout = ba.ParticleLayout()
    layout.addParticle(particle, 1.0)
    layout.setInterference(interference)
    # Density approx: 1 particle per D^2
    layout.setTotalParticleSurfaceDensity(1.0 / (peak_distance**2))

    layer_air = ba.Layer(m_air)
    layer_film = ba.Layer(m_polymer, 80 * nm)
    layer_film.addLayout(layout)
    layer_film.setNumberOfSlices(20) # Crucial for Yoneda simulation
    layer_substrate = ba.Layer(m_substrate)

    sample = ba.Sample()
    sample.addLayer(layer_air)
    sample.addLayer(layer_film)
    sample.addLayer(layer_substrate)
    
    return sample

def run_LMA_simulation():
    """
    Runs the Local Monodisperse Approximation loop manually using NumPy.
    """
    # LMA Parameters
    avg_radius = 4.0 * nm
    sigma = 0.1 * avg_radius
    n_points = 11
    
    # Generate distribution points manually (Robust for all versions)
    radii = np.linspace(avg_radius - 3*sigma, avg_radius + 3*sigma, n_points)
    weights = norm.pdf(radii, loc=avg_radius, scale=sigma)
    weights /= weights.sum() # Normalize

    total_intensity = None
    
    print(f"Simulating LMA... (Average Spacing: {AVG_DISTANCE:.1f} nm)")

    for radius, weight in zip(radii, weights):
        # Spacing scales with particle size (packing constraint)
        spacing = AVG_DISTANCE * (radius / avg_radius)
        
        sample = get_sample(radius, spacing)
        
        # Define Beam
        beam = ba.Beam(1e8, WAVELENGTH, 0.13*deg)
        
        # Define Lateral Detector Strip (Phi Scan)
        # Scan phi from 0 to 1.0 degrees (covering the 0.17 and 0.34 peaks)
        n_phi = 200
        detector = ba.SphericalDetector(
        4000, 0*deg, 3*deg,
        4000, 0*deg, 3*deg
    )
        
        simulation = ba.ScatteringSimulation(beam, sample, detector)
        result = simulation.simulate()
        
        if total_intensity is None:
            total_intensity = result.array() * weight
        else:
            total_intensity += result.array() * weight

    return total_intensity

if __name__ == '__main__':
    data = run_LMA_simulation()
    
    # Simple Plot
    import matplotlib.pyplot as plt
    phi_axis = np.linspace(0, 1.0, 200)
    
    plt.figure(figsize=(10,6))
    plt.semilogy(phi_axis, data.flatten(), label='LMA Simulation')
    plt.xlabel(r'$\phi_f$ (deg)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Lateral Profile (LMA Paracrystal)')
    plt.legend()
    plt.grid(True)
    plt.show()