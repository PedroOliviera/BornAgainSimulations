# BornAgain v23+ (Python)
import math
import bornagain as ba
from bornagain import deg, nm

# ---- your GALAXY / RAYONIX geometry (edit as needed) ----
rayonix_npx = 4096
rayonix_npy = 4096
rayonix_pixel_size_mm = 0.073242   # mm
# beam_xpos, beam_ypos are pixel indices where the direct beam would hit the panel
beam_xpos = 2048                   # e.g., middle of the panel; EDIT to your logs
beam_ypos = 2048

wavelength = 0.125916*nm           # your beamline wavelength
# ----------------------------------------------------------

def _mm_over_mm_to_angle(x_over_D):
    # exact small-angle mapping using atan
    return math.atan(x_over_D)

def create_spherical_detector(
    detector_distance_mm: float,
    npx: int = rayonix_npx,
    npy: int = rayonix_npy,
    pixel_size_mm: float = rayonix_pixel_size_mm,
    beam_xpos_px: float = beam_xpos,
    beam_ypos_px: float = beam_ypos,
) -> ba.SphericalDetector:
    """
    Build a SphericalDetector whose angular span matches a flat panel
    at distance D, with pixel size 'pixel_size_mm', and a direct-beam
    intersection at (beam_xpos_px, beam_ypos_px) in pixel units.
    """
    width_mm  = npx * pixel_size_mm
    height_mm = npy * pixel_size_mm
    u0_mm = beam_xpos_px * pixel_size_mm
    v0_mm = beam_ypos_px * pixel_size_mm
    D = detector_distance_mm

    # Horizontal (phi): left edge is at -u0, right edge at (width - u0)
    phi_min  = _mm_over_mm_to_angle((-u0_mm)        / D)
    phi_max  = _mm_over_mm_to_angle((width_mm-u0_mm)/ D)

    # Vertical (alpha): bottom edge is at -v0, top edge at (height - v0)
    alpha_min = _mm_over_mm_to_angle((-v0_mm)          / D)
    alpha_max = _mm_over_mm_to_angle((height_mm-v0_mm) / D)

    # Build spherical detector with angular ranges (in radians)
    detector = ba.SphericalDetector(npx, phi_min, phi_max, npy, alpha_min, alpha_max)

    return detector

def get_simulation_2D(sample_model,
                          detectorDistBeamtime='feb',
                          angle=None,                 # incidence angle in degrees
                          beamIntensity=1.3e12,
                          ROI=None,
                          background=23.0):
    """
    v23+ replacement using SphericalDetector. Keeps your old knobs:
      - detectorDistBeamtime: 'feb' or 'dec' -> distance in mm
      - angle: incidence angle in degrees (alpha_i)
      - ROI: optional (x1_deg, y1_deg, x2_deg, y2_deg) in detector angular units
    """
    if detectorDistBeamtime == 'feb':
        detectorDist = 2337.126  # mm
    elif detectorDistBeamtime == 'dec':
        detectorDist = 3052.624  # mm
    else:
        raise ValueError("detectorDistBeamtime must be 'feb' or 'dec'")

    alpha_i = angle * deg
    beam = ba.Beam(beamIntensity, wavelength, alpha_i)

    detector = create_spherical_detector(detectorDist)

    simulation = ba.ScatteringSimulation(beam, sample_model, detector)
    simulation.options().setIncludeSpecular(True)   # same as before
    simulation.setBackground(ba.ConstantBackground(background))

    # Example: mask (coordinates are in detector angular units)
    # simulation.detector().addMask(ba.Rectangle(148.07, 140.59, 152.02, 177.91), True)

    # Region of interest (still supported in v23; args are in deg)
    if ROI is not None:
        x1, y1, x2, y2 = ROI
        simulation.detector().setRegionOfInterest(x1, y1, x2, y2)

    return simulation

def get_sampleTest():
    # --- materials ---
    material_PS    = ba.RefractiveMaterial("PS",    2.51433698e-06, 2.35385822e-09)
    material_P2VP  = ba.RefractiveMaterial("P2VP",  1.656e-06,      1.096e-09)
    material_FA    = ba.RefractiveMaterial("FA",    3.90901641e-06, 1.79148728e-07)
    material_Si_Sub= ba.RefractiveMaterial("Si Sub",5.04218633e-06, 7.83926453e-08)
    material_SiO2  = ba.RefractiveMaterial("SiO2",  4.7465490081665e-06, 4.1351946628761e-08)
    vacuum         = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    # --- particle (PS “hemisphere” in old comment used a full Sphere; keep Sphere unless you intend hemispheres) ---
    radius_PS = 48/2 * nm
    ff_PS = ba.Sphere(radius_PS)
    particle_PS = ba.Particle(material_PS, ff_PS)

    # --- structure / interference ---
    # Old: InterferenceHardDisk(spacing/2*nm, eta)  # liquid-like order
    # New: use Radial Paracrystal with nearest-neighbor distance = spacing
    spacing = 70.95529824561405 * nm

    iff = ba.InterferenceRadialParacrystal(spacing, 250*nm)          # damping_length=0 by default
    # pick a modest nearest-neighbor spread; tune omega to taste
    iff.setProbabilityDistribution(ba.Profile1DGauss(0.02*spacing))

    # Build a structured layout from the interference function
    layout = ba.StructuredLayout(iff)
    layout.addParticle(particle_PS)

    # v24 note: for RadialParacrystal you should set the areal particle density (in 1/nm^2)
    # A simple first guess is ~ 1/spacing^2; adjust by packing fraction if needed.
    #layout.setParticleDensity(1.0 / ((spacing/nm)*(spacing/nm)))  # 1/nm^2

    # --- layers ---
    top = ba.Layer(vacuum)
    # If particles sit at the top surface, add the struct to the top layer
    top.addStruct(layout)

    sio2 = ba.Layer(material_SiO2, 2*nm)
    sub  = ba.Layer(material_Si_Sub)

    # --- sample ---
    sample = ba.Sample()
    sample.addLayer(top)
    sample.addLayer(sio2)
    sample.addLayer(sub)
    return sample
