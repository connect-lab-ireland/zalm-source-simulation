# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: config.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Central configuration file for the ZALM Entanglement Swapping simulation.
All experimental parameters, noise models, and operational modes are defined here.
"""

# =============================================================================
#  SIMULATION MODE AND RUNTIME PARAMETERS
# =============================================================================
# 'REALISTIC': Includes noise, loss, and component inefficiencies.
# 'IDEAL': Assumes perfect components and no decoherence.
SIM_MODE = 'REALISTIC' #'REALISTIC'

# Number of entanglement swapping attempts to simulate for EACH PAIR of sources
NUM_RUNS = 10000

# If True HOM Visibility is set to 1.0
HOM_VISIBILITY_AT_MAX = False
# =============================================================================
#  PHYSICAL COMPONENT PARAMETERS (from literature and datasheets)
# =============================================================================

# --- SPDC SOURCE PARAMETERS (Based on Covesion SPDC-1550-5-PG) ---
# List of DWDM CHANNEL INDICES to place SPDC sources on.
# These are indices into the generated grid, not specifically ITU channel numbers.
#SPDC_SOURCES_INDICES = [32, 32] #, 20, 25, 30]

# Pump wavelength in nanometers.
SPDC_PUMP_WAVELENGTH_NM = 775.0

# Degeneracy bandwidth FWHM in nanometers.
SPDC_DEGENERACY_BANDWIDTH_FWHM_NM = 1.0 # 5.0

# Defines how idler frequencies are determined relative to signal frequencies.
SPDC_MODE = 'DICHROIC'  # 'DICHROIC' or 'PBS'

# Probability of the source successfully emitting a pair when triggered.
EMISSION_SUCCESS_PROBABILITY = 0.95

# --- COMPONENT & NOISE PARAMETERS ---
# Defines the shape of the DWDM filter transmission window.
# 'BRICK_WALL': Ideal filter with perfect transmission inside the passband, zero outside.
# 'GAUSSIAN': Realistic filter with sloped edges.
FILTER_MODEL = 'GAUSSIAN'  # Options: 'BRICK_WALL', 'GAUSSIAN'

# Passband width of the DWDM filter as a fraction of the grid spacing.
# A common rule of thumb for low crosstalk.
FILTER_PASSBAND_FRACTION = 0.8 # 80%

# --- DWDM FLEXGRID PARAMETERS ---
# Define which optical bands to include in the simulation grid.
ENABLED_BANDS = ['C']  # Options: ['S'], ['C'], ['L'], or combinations like ['C', 'L']

# Define the grid granularity in GHz. This is the channel spacing.
GRID_GRANULARITY_GHZ = 100  # Options: 100, 50, 25, 12.5

# Insertion loss (in dB) for each optical component at its center wavelength.
# Reference: Thorlabs datasheets for typical fiber-coupled components.
BEAMSPLITTER_INSERTION_LOSS_DB = 0.2
PBS_INSERTION_LOSS_DB = 0.2
DWDM_FILTER_INSERTION_LOSS_DB = 0.5 # Filters are typically lossier

BEAMSPLITTER_HOM_THRESHOLD = 0.99 # SHould probably be .99

PBS_EXTINCTION_RATIO = 0.001
# --- Detector Parameters ---
# Choose detector type for the Bell State Measurement station.
# 'STANDARD': Standard single-photon detectors (cannot distinguish |Φ> states).
# 'PNR': Photon-Number-Resolving detectors (can distinguish all 4 Bell states).
DETECTOR_TYPE = 'PNR'  # Options: 'STANDARD', 'PNR'

# Efficiency of the Single-Photon Detectors (SPDs).
# Reference: State-of-the-art SNSPDs (Superconducting Nanowire Single-Photon Detectors)
# P. V. Morozov et al., "SNSPD with >98% polarization-independent system detection efficiency" (2023)
DETECTOR_EFFICIENCY = 0.98  # 98% chance of detecting an arriving photon

# =============================================================================
#  PHOTON INDISTINGUISHABILITY PARAMETERS
# =============================================================================
# --- Spectral Properties ---
# The spectral width (Full-Width at Half-Maximum) of the photons from the SPDC sources.
PHOTON_FWHM_GHZ = 30.0

# --- Temporal Properties ---
# The temporal jitter in photon arrival times at the beam splitter.
# This value is the standard deviation of a Gaussian distribution.
TEMPORAL_JITTER_STDEV_PS = 20.0 #5.0

# =============================================================================
#  QUANTUM PROCESSOR AND NOISE PARAMETERS
# =============================================================================
# --- Quantum Memory Decoherence ---
# Depolarization rate for a qubit sitting idle in memory.
# Units: Hz (events per second). Set to 0 for ideal memory.
MEMORY_DEPOLAR_RATE = 1e3  # 1 kHz rate, a reasonable value for good quantum memories

# --- Gate Errors ---
# Error rates for physical instructions, applied as time-independent probabilities.
# These represent the imperfection of applying a gate.
# Reference: "Blueprint for a microwave trapped ion quantum computer" Nature (2018) ??
# gives single-qubit gate error ~1e-4 and two-qubit gate error ~1e-3.
GATE_ERROR_PROB_SINGLE_QUBIT = 1e-4  # 0.01% error probability
GATE_ERROR_PROB_TWO_QUBIT = 1e-3   # 0.1% error probability
MEASUREMENT_DEPHASE_PROB = 1e-3      # 0.1% dephasing error on measurement

# =============================================================================
#  PLOTTING AND ANALYSIS PARAMETERS
# =============================================================================
# Define the center channel for the success metric plot.
# This allows focusing the visualization on a specific region of interest.
PLOT_TARGET_CENTER_CHANNEL = 34

# Define the span of channels to display on either side of the center channel.
# For example, a span of 10 will show 21 channels in total (center ± 10).
PLOT_CHANNEL_SPAN = 10e-5

# -----------------------------------------------------------------------------
# Distance between source and first nodes
# -----------------------------------------------------------------------------
INTERNODE_LENGTH = 5 # km
# --- Define physical constants ---
SPEED_OF_LIGHT_IN_VACUUM = 299792458.0  # m/s
REFRACTIVE_INDEX_OF_FIBER = 1.468      # For standard SMF-28 fiber at 1550 nm

# =============================================================================
#  COMPUTATIONAL PARAMETERS
# =============================================================================
# Number of parallel worker processes to use for the simulation.
# Set to 0 to auto-detect and use all available CPU cores minus one.
# Set to 1 to run in a single process (useful for debugging).
NUM_WORKERS = 0
"""
def apply_config_mode():
    # -----------------------------------------------------------------------------
    # Apply IDEAL settings if mode is selected
    # -----------------------------------------------------------------------------
    print("Updating configuration parameters")
    if SIM_MODE == 'IDEAL':
        print("INFO: Running in IDEAL mode. Overriding all noise and loss parameters.")
        # Use `globals()` to modify the module's global variables
        g = globals()
        g[EMISSION_SUCCESS_PROBABILITY] = 1.0
        g[FILTER_MODEL] = 'BRICK_WALL' # Ideal mode must use a perfect filter
        g[BEAMSPLITTER_INSERTION_LOSS_DB] = 0.0
        g[PBS_INSERTION_LOSS_DB] = 0.0
        g[DWDM_FILTER_INSERTION_LOSS_DB] = 0.0
        g[DETECTOR_EFFICIENCY] = 1.0
        g[MEMORY_DEPOLAR_RATE] = 0
        g[GATE_ERROR_PROB_SINGLE_QUBIT] = 0
        g[GATE_ERROR_PROB_TWO_QUBIT] = 0
        g[MEASUREMENT_DEPHASE_PROB] = 0
        g[SPDC_DEGENERACY_BANDWIDTH_FWHM_NM] = 0.0
        g[TEMPORAL_JITTER_STDEV_PS] = 0
        g[PHOTON_FWHM_GHZ] = 0.1 # Use a very small but non-zero value
        g[HOM_VISIBILITY_AT_MAX] = True
        g[PBS_EXTINCTION_RATIO] = 0.0
        g[INTERNODE_LENGTH] = 10e-5
"""
# -----------------------------------------------------------------------------
# Apply IDEAL settings if mode is selected
# -----------------------------------------------------------------------------
if SIM_MODE == 'IDEAL':
    print("INFO: Running in IDEAL mode. Overriding all noise and loss parameters.")
    EMISSION_SUCCESS_PROBABILITY = 1.0
    FILTER_MODEL = 'BRICK_WALL' # Ideal mode must use a perfect filter
    BEAMSPLITTER_INSERTION_LOSS_DB = 0.0
    PBS_INSERTION_LOSS_DB = 0.0
    DWDM_FILTER_INSERTION_LOSS_DB = 0.0
    DETECTOR_EFFICIENCY = 1.0
    MEMORY_DEPOLAR_RATE = 0
    GATE_ERROR_PROB_SINGLE_QUBIT = 0
    GATE_ERROR_PROB_TWO_QUBIT = 0
    MEASUREMENT_DEPHASE_PROB = 0
    SPDC_DEGENERACY_BANDWIDTH_FWHM_NM = 0.0
    TEMPORAL_JITTER_STDEV_PS = 0
    PHOTON_FWHM_GHZ = 0.1 # Use a very small but non-zero value
    HOM_VISIBILITY_AT_MAX = True
    PBS_EXTINCTION_RATIO = 0.0
#    INTERNODE_LENGTH = 10e-5 # km
# --- CONFIGURATION VALIDATION ---
EFFECTIVE_FILTER_BANDWIDTH_GHZ = GRID_GRANULARITY_GHZ * FILTER_PASSBAND_FRACTION
if FILTER_MODEL == 'BRICK_WALL' and PHOTON_FWHM_GHZ > EFFECTIVE_FILTER_BANDWIDTH_GHZ:
    raise ValueError(
        f"\n\n*** CONFIGURATION ERROR ***\n"
        f"Photon spectral width (FWHM) is larger than the DWDM filter passband.\n"
        f"  - Photon FWHM:           {PHOTON_FWHM_GHZ:.1f} GHz\n"
        f"  - Filter Passband:       {EFFECTIVE_FILTER_BANDWIDTH_GHZ:.1f} GHz\n\n"
        f"Please DECREASE 'PHOTON_FWHM_GHZ' or INCREASE 'GRID_GRANULARITY_GHZ'/'FILTER_PASSBAND_FRACTION' in config.py.\n"
    )

# Call it once on initial import to set the default mode
#apply_config_mode()
"""
class Config:
    def __init__(self):
        # --- Default REALISTIC settings ---
        SIM_MODE = 'REALISTIC'
        NUM_RUNS = 10000
        HOM_VISIBILITY_AT_MAX = False
        SPDC_PUMP_WAVELENGTH_NM = 775.0
        SPDC_DEGENERACY_BANDWIDTH_FWHM_NM = 5.0 # 5.0
        SPDC_MODE = 'PBS'  # 'SYMMETRIC' or 'DICHROIC' or 'PBS'
        EMISSION_SUCCESS_PROBABILITY = 0.95
"""
