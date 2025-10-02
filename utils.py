# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: config.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Utility functions for the ZALM simulation, including logging, component creation,
and physics calculations.
"""
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from typing import Dict, List
import datetime
from netsquid.qubits import qubitapi as qapi
from netsquid.components import instructions as instr
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction

import io  # Required for capturing print output
from contextlib import redirect_stdout # Required for capturing print output

import config

from netsquid.qubits.qformalism import QFormalism

from statsmodels.nonparametric.smoothers_lowess import lowess

# --- PROGRAMMATIC FLEXGRID GENERATION ---
BAND_RANGES_THZ = {
    'S': (186.0, 191.6), 'C': (191.7, 196.1), 'L': (196.2, 200.6)
}

def generate_flex_dwdm_grid(bands: List[str], granularity_ghz: float) -> Dict[float, int]:
    """
    Generates a flexible ITU-T DWDM channel grid dictionary.

    Args:
        bands (List[str]): A list of bands to include (e.g., ['C', 'L']).
        granularity_ghz (float): The channel spacing in GHz (100, 50, 25, 12.5).

    Returns:
        Dict[float, int]: A dictionary mapping frequency (THz) to a unique channel index.
    """
    if not all(band in BAND_RANGES_THZ for band in bands):
        raise ValueError(f"Invalid band specified. Use 'C', 'L', or 'S'.")

    grid, channel_index = {}, 0
    spacing_thz = granularity_ghz / 1000.0
    sorted_bands = sorted(bands, key=lambda b: BAND_RANGES_THZ[b][0])

    for band in sorted_bands:
        start_freq_thz, end_freq_thz = BAND_RANGES_THZ[band]
        if granularity_ghz == 100:
            channel_index = int((start_freq_thz * 10) % 100)
        freqs = np.arange(start_freq_thz, end_freq_thz, spacing_thz)
        for freq in freqs:
            rounded_freq = round(freq, 5)
            if rounded_freq not in grid:
                grid[rounded_freq] = channel_index
                channel_index += 1

    return grid

DWDM_CHANNELS = generate_flex_dwdm_grid(config.ENABLED_BANDS, config.GRID_GRANULARITY_GHZ)
DWDM_FREQS_ARRAY = np.asarray(list(DWDM_CHANNELS.keys()))
# =============================================================================
# Create the reversed lookup dictionary for channel -> frequency
# =============================================================================
# This is done once at startup for maximum efficiency.
# It swaps the keys and values of the DWDM_CHANNELS dictionary.
REVERSED_DWDM_CHANNELS = {channel_index: freq_thz for freq_thz, channel_index in DWDM_CHANNELS.items()}
#print(f"INFO: Generated FlexGrid for bands {config.ENABLED_BANDS} "
#      f"with {config.GRID_GRANULARITY_GHZ} GHz spacing, "
#      f"containing {len(DWDM_CHANNELS)} channels.")

LOG = None
def setup_logging(level=logging.ERROR, active=True):
    """Configures the root logger for the simulation.
    Can be disabled for worker processes.
    """
    global LOG
    if LOG is None:
        LOG = logging.getLogger("ZALM_SIM")
        # If not active, set level to something very high to silence it
        log_level = level if active else logging.CRITICAL + 1
        LOG.setLevel(log_level)

        # Only add a handler if one doesn't exist and logging is active
        if not LOG.handlers and active:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            LOG.addHandler(handler)
    return LOG

def create_processor(name: str, num_positions: int) -> QuantumProcessor:
    """Factory to create a quantum processor with realistic noise models."""
    # Explicitly set the formalism for all noise models
    formalism = QFormalism.DM


    single_qubit_noise = DepolarNoiseModel(depolar_rate=config.GATE_ERROR_PROB_SINGLE_QUBIT, time_independent=True, qstate_formalism=formalism)
    two_qubit_noise = DepolarNoiseModel(depolar_rate=config.GATE_ERROR_PROB_TWO_QUBIT, time_independent=True, qstate_formalism=formalism)
    measure_noise = DephaseNoiseModel(dephase_rate=config.MEASUREMENT_DEPHASE_PROB, time_independent=True, qstate_formalism=formalism)

    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, quantum_noise_model=single_qubit_noise),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, quantum_noise_model=single_qubit_noise),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, quantum_noise_model=single_qubit_noise),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True, quantum_noise_model=two_qubit_noise),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, quantum_noise_model=measure_noise)]
    memory_noise = DepolarNoiseModel(depolar_rate=config.MEMORY_DEPOLAR_RATE, qstate_formalism=formalism)
    processor = QuantumProcessor(name, num_positions=num_positions, memory_noise_models=[memory_noise] * num_positions, phys_instructions=physical_instructions)
    return processor

def calculate_visibility(delta_f_ghz: float, delta_t_ps: float) -> float:
    """Calculates the total HOM visibility from spectral and temporal mismatch."""
    vis_spectral = 1.0
    if config.PHOTON_FWHM_GHZ > 0:
        sigma_f = config.PHOTON_FWHM_GHZ / (2 * np.sqrt(2 * np.log(2)))
        vis_spectral = np.exp(-(delta_f_ghz**2) / (2 * sigma_f**2))

    vis_temporal = 1.0
    fwhm_duration_ps = (1000 / config.PHOTON_FWHM_GHZ) if config.PHOTON_FWHM_GHZ > 0 else float('inf')
    if fwhm_duration_ps > 0 and delta_t_ps > 0:
        sigma_t = fwhm_duration_ps / (2 * np.sqrt(2 * np.log(2)))
        vis_temporal = np.exp(-(delta_t_ps**2) / (2 * sigma_t**2))
    return vis_spectral * vis_temporal

def find_channel_from_freq(freq_thz: float) -> int:
    """Finds the nearest DWDM channel for a given frequency value."""
    if len(DWDM_FREQS_ARRAY) == 0: return -1
    idx = (np.abs(DWDM_FREQS_ARRAY - freq_thz)).argmin()
    return DWDM_CHANNELS[DWDM_FREQS_ARRAY[idx]]

def find_freq_from_channel(channel_index: int) -> float:
    """
    Finds the center frequency (in THz) for a given DWDM channel index.

    This is the reverse of find_channel_from_freq. It uses a pre-computed
    lookup table for O(1) efficiency.

    Args:
        channel_index (int): The integer index of the DWDM channel.

    Returns:
        float: The corresponding center frequency in THz.

    Raises:
        KeyError: If the channel_index is not found in the generated grid.
    """
    try:
        return REVERSED_DWDM_CHANNELS[channel_index]
    except KeyError:
        print(f"ERROR: Channel index {channel_index} is not a valid channel in the current DWDM grid.")
        raise

def determineBellState(detector1: str, detector2: str) -> str:
    """
    Determines the heralded Bell state based on which two detectors clicked.

    Args:
        detector1 (str): The name of the first detector, e.g., "FilterAH".
        detector2 (str): The name of the second detector, e.g., "FilterBV".

    Returns:
        str: The name of the heralded Bell state ("PSI+", "PSI-", "PHI+", "PHI-").
    """
    #s1, p1 = detector1[-2], detector1[-1]; s2, p2 = detector2[-2], detector2[-1]
    #if s1 == s2: return "PHI+" if p1 == p2 else "PHI-"
    #else: return "PSI+" if p1 == p2 else "PSI-"
    Filter1 = detector1[-2:]
    Filter2 = detector2[-2:]
    if (Filter1 == "AH" and Filter2 == "BV") or (Filter1 == "AV" and Filter2 == "BH"): # |01> - |10>
        return "PSI-"
    elif (Filter1 == "BV" and Filter2 == "AH") or (Filter1 == "BH" and Filter2 == "AV"): # |01> - |10>
        return "PSI-"
    elif (Filter1 == "AH" and Filter2 == "AV") or (Filter1 == "BH" and Filter2 == "BV"): # |01> + |10>
        return "PSI+"
    elif (Filter1 == "AV" and Filter2 == "AH") or (Filter1 == "BV" and Filter2 == "BH"): # |01> + |10>
        return "PSI+"
    elif (Filter1 == "AH" and Filter2 == "AH") or (Filter1 == "AV" and Filter2 == "AV"): # |00> - |11>
        return "PHI-"
    elif (Filter1 == "BV" and Filter2 == "BV") or (Filter1 == "BH" and Filter2 == "BH"): # |00> + |11>
        return "PHI+"
    else:
        return "NULL"

def analyze_and_plot_results(results_df: pd.DataFrame, directory="./"):
    """
    Performs analysis, prints a summary, generates plots, and saves all
    information to a summary text file.
    """
    # --- Generate a unique timestamp and base filename for this run ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    band_str = "".join(config.ENABLED_BANDS)
    base_filename = (f"results_{config.SIM_MODE}_{band_str}_{config.GRID_GRANULARITY_GHZ}GHz_"
                     f"{config.DETECTOR_TYPE}_{timestamp}")

    # Use an in-memory text stream to capture all print statements
    output_capture = io.StringIO()
    with redirect_stdout(output_capture):
        # --- All print statements inside this block will be captured ---
        print("="*50)
        print(" " * 15 + "SIMULATION RESULTS ANALYSIS")
        print("="*50)

        num_total = len(results_df)
        if num_total == 0:
            print("No results to analyze.")
            return

        # --- Basic Configuration Info ---
        cfg_str = (f"Mode: {config.SIM_MODE}, Grid: {config.GRID_GRANULARITY_GHZ}GHz, "
                   f"Filter: {config.FILTER_MODEL}, Detectors: {config.DETECTOR_TYPE}")
        print(f"\nConfiguration Summary: {cfg_str}")

        # --- Overall Outcome Counts ---
        status_counts = results_df['status'].value_counts()
        print("\n--- Overall Outcome Counts ---")
        print(status_counts)
        if config.NUM_RUNS - num_total != 0:
            print(f"Fibre Losses\t{config.NUM_RUNS-num_total} over {config.INTERNODE_LENGTH}km")

        # --- Success Analysis ---
        success_df = results_df[results_df['status'] == 'SUCCESS'].copy()
        num_success = len(success_df)
        success_rate = num_success / num_total if num_total > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.2%}")

        if num_success > 0:
            avg_fidelity_overall = success_df['fidelity'].mean()
            print(f"Average Fidelity of All Successful Swaps: {avg_fidelity_overall:.4f}")
            print("\n--- BSM Outcome Distribution (on Success) ---")
            bsm_counts = success_df['bsm_outcome'].value_counts()
            print(bsm_counts)

        # --- Failure Analysis ---
        failure_df = results_df[results_df['status'] == 'FAILURE'].copy()
        num_failure = len(failure_df)
        failure_rate = num_failure / num_total if num_total > 0 else 0
        print(f"\nFailure Rate: {failure_rate:.2%}")
        if num_failure > 0 and 'failure_reason' in failure_df.columns:
            print("\n--- Reasons for Failure ---")
            failure_reasons = failure_df['failure_reason'].value_counts()
            print(failure_reasons)

    # --- Now that the block is over, the captured text is in the variable ---
    analysis_summary_str = output_capture.getvalue()

    # --- Print the captured summary to the console so the user still sees it ---
    print(analysis_summary_str)

    # --- Write everything to a summary text file ---
    config_summary_str = get_config_summary()
    summary_filename = f"{base_filename}.txt"
    print(f"Writing summary report to {summary_filename}...")
    with open(directory+"/"+summary_filename, 'w') as f:
        f.write(config_summary_str)
        f.write(analysis_summary_str)

    # --- Plotting ---
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        print("Warning: 'seaborn-whitegrid' style not found. Using 'ggplot'.")
        plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(f'Simulation Results ({num_total} runs)\n{cfg_str}', fontsize=16)

    # Plot 1: Overall Status Distribution
    status_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%',
                       colors=['#4C72B0', '#C44E52', '#55A868'],
                       textprops={'fontsize': 12}, startangle=90)
    ax1.set_title('Overall Run Status', fontsize=14); ax1.set_ylabel('')

    """
    # Plot 2: Failure Reason Breakdown
    if num_failure > 0 and 'failure_reason' in failure_df.columns:
        failure_reasons.sort_index().plot(kind='barh', ax=ax2, color='#C44E52', alpha=0.8)
        ax2.set_title('Breakdown of Failure Reasons', fontsize=14)
        ax2.set_xlabel('Number of Runs')
    """
    # --- Plot 2: Success Metrics per DWDM Channel (Dual-Axis Plot) ---
    ax2.set_title('Success Metrics by DWDM Channel', fontsize=14)

    # 1. Define the full x-axis range from the config file
    center_chan = config.PLOT_TARGET_CENTER_CHANNEL
    span = config.PLOT_CHANNEL_SPAN
    full_x_axis_channels = np.arange(center_chan - span, center_chan + span + 1)

    if num_success > 0 and 'dwdm_channel' in success_df.columns:
        # 2. Aggregate the actual successful run data
        success_df['dwdm_channel'] = success_df['dwdm_channel'].astype(int)
        channel_analysis = success_df.groupby('dwdm_channel').agg(
            success_count=('status', 'size'),
            avg_fidelity=('fidelity', 'mean')
        )

        # 3. THE CRUCIAL STEP: Reindex the data to the full desired axis range.
        # This will create rows for all channels in `full_x_axis_channels`.
        # For channels with no successes, 'success_count' will be filled with 0,
        # and 'avg_fidelity' will be filled with NaN (which is correct).
        plot_data = channel_analysis.reindex(full_x_axis_channels).fillna({'success_count': 0})

        print("\n--- Success Metrics Across Plotting Window ---")
        print(plot_data)

        # 4. Plot the data using the new, complete `plot_data` DataFrame
        color_bar = '#4C72B0'
        ax2.set_xlabel('DWDM Channel Index', fontsize=12)
        ax2.set_ylabel('Number of Successful Runs', color=color_bar, fontsize=12)
        ax2.bar(plot_data.index, plot_data['success_count'],
                color=color_bar, alpha=0.8, label='Success Count', width=0.8)
        ax2.tick_params(axis='y', labelcolor=color_bar)

        # 5. Set x-ticks and labels to prevent crowding
        tick_locations = plot_data.index
        tick_labels = plot_data.index
        # If there are too many ticks, only show every Nth label
        if len(tick_locations) > 20:
             tick_locations = tick_locations[::2] # Show every 2nd tick
             tick_labels = tick_labels[::2]
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

        # 6. Plot fidelity on the twin axis
        ax2_twin = ax2.twinx()
        color_line = '#C44E52'
        ax2_twin.set_ylabel('Average Fidelity', color=color_line, fontsize=12)
        ax2_twin.plot(plot_data.index, plot_data['avg_fidelity'],
                      color=color_line, marker='o', ms=5, linestyle='--', label='Avg. Fidelity')
        ax2_twin.tick_params(axis='y', labelcolor=color_line)
        ax2_twin.set_ylim(0, 1.05)
        ax2_twin.grid(False)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_bar, alpha=0.8, label='Success Count'),
                           plt.Line2D([0], [0], color=color_line, marker='o', linestyle='--', label='Avg. Fidelity')]
        ax2.legend(handles=legend_elements, loc='upper left')

    else:
        ax2.text(0.5, 0.5, 'No Failures Recorded', ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Breakdown of Failure Reasons', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot with the same base filename
    plot_filename = f"{base_filename}.png"
    print(f"\nSaving summary plot to {plot_filename}...")
    plt.savefig(directory+"/"+plot_filename, dpi=300)
    #plt.show()

def analyze_and_plot_results_old(results_df: pd.DataFrame):
    """
    Performs analysis on the final DataFrame and generates plots.
    This version includes a dual-axis plot showing success counts and
    average fidelity per DWDM channel over a user-defined, fixed range.
    """
    print("\n" + "="*50)
    print(" " * 15 + "SIMULATION RESULTS ANALYSIS")
    print("="*50)

    num_total = len(results_df)
    if num_total == 0:
        print("No results to analyze."); return

    cfg_str = (f"Mode: {config.SIM_MODE}, SPDC Mode: {config.SPDC_MODE}, Grid: {config.GRID_GRANULARITY_GHZ}GHz, "
               f"Filter: {config.FILTER_MODEL}, Detectors: {config.DETECTOR_TYPE}")
    print(f"\nConfiguration: {cfg_str}")

    status_counts = results_df['status'].value_counts()
    print("\n--- Overall Outcome Counts ---"); print(status_counts)

    success_df = results_df[results_df['status'] == 'SUCCESS'].copy()
    num_success = len(success_df)
    print(f"\nSuccess Rate: {num_success / num_total:.2%}")

    if num_success > 0:
        avg_fidelity_overall = success_df['fidelity'].mean()
        print(f"Average Fidelity of All Successful Swaps: {avg_fidelity_overall:.4f}")

    # --- Plotting ---
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        print("Warning: 'seaborn-whitegrid' style not found. Using 'ggplot'.")
        plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(f'Simulation Results ({num_total} runs)\n{cfg_str}', fontsize=16)

    # --- Plot 1: Overall Status Distribution (Pie Chart) ---
    status_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%',
                       colors=['#4C72B0', '#C44E52', '#55A868'],
                       textprops={'fontsize': 12}, startangle=90)
    ax1.set_title('Overall Run Status', fontsize=14); ax1.set_ylabel('')

    # --- Plot 2: Success Metrics per DWDM Channel (Dual-Axis Plot) ---
    ax2.set_title('Success Metrics by DWDM Channel', fontsize=14)

    # 1. Define the full x-axis range from the config file
    center_chan = config.PLOT_TARGET_CENTER_CHANNEL
    span = config.PLOT_CHANNEL_SPAN
    full_x_axis_channels = np.arange(center_chan - span, center_chan + span + 1)

    if num_success > 0 and 'dwdm_channel' in success_df.columns:
        # 2. Aggregate the actual successful run data
        success_df['dwdm_channel'] = success_df['dwdm_channel'].astype(int)
        channel_analysis = success_df.groupby('dwdm_channel').agg(
            success_count=('status', 'size'),
            avg_fidelity=('fidelity', 'mean')
        )

        # 3. THE CRUCIAL STEP: Reindex the data to the full desired axis range.
        # This will create rows for all channels in `full_x_axis_channels`.
        # For channels with no successes, 'success_count' will be filled with 0,
        # and 'avg_fidelity' will be filled with NaN (which is correct).
        plot_data = channel_analysis.reindex(full_x_axis_channels).fillna({'success_count': 0})

        print("\n--- Success Metrics Across Plotting Window ---")
        print(plot_data)

        # 4. Plot the data using the new, complete `plot_data` DataFrame
        color_bar = '#4C72B0'
        ax2.set_xlabel('DWDM Channel Index', fontsize=12)
        ax2.set_ylabel('Number of Successful Runs', color=color_bar, fontsize=12)
        ax2.bar(plot_data.index, plot_data['success_count'],
                color=color_bar, alpha=0.8, label='Success Count', width=0.8)
        ax2.tick_params(axis='y', labelcolor=color_bar)

        # 5. Set x-ticks and labels to prevent crowding
        tick_locations = plot_data.index
        tick_labels = plot_data.index
        # If there are too many ticks, only show every Nth label
        if len(tick_locations) > 20:
             tick_locations = tick_locations[::2] # Show every 2nd tick
             tick_labels = tick_labels[::2]
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

        # 6. Plot fidelity on the twin axis
        ax2_twin = ax2.twinx()
        color_line = '#C44E52'
        ax2_twin.set_ylabel('Average Fidelity', color=color_line, fontsize=12)
        ax2_twin.plot(plot_data.index, plot_data['avg_fidelity'],
                      color=color_line, marker='o', ms=5, linestyle='--', label='Avg. Fidelity')
        ax2_twin.tick_params(axis='y', labelcolor=color_line)
        ax2_twin.set_ylim(0, 1.05)
        ax2_twin.grid(False)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_bar, alpha=0.8, label='Success Count'),
                           plt.Line2D([0], [0], color=color_line, marker='o', linestyle='--', label='Avg. Fidelity')]
        ax2.legend(handles=legend_elements, loc='upper left')

    else:
        # Handle the case where there are no successful runs at all
        ax2.text(0.5, 0.5, 'No Successful Runs to Analyze',
                 ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_xticks(full_x_axis_channels[::2]) # Still show the empty axis
        ax2.set_xticklabels(full_x_axis_channels[::2], rotation=45, ha='right', va='center', fontsize=14, transform=ax2.transAxes)

    # --- Finalize and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    band_str = "".join(config.ENABLED_BANDS)
    filename = (f"results_{config.SIM_MODE}_{band_str}_{config.GRID_GRANULARITY_GHZ}GHz_"
                f"{config.DETECTOR_TYPE}_{timestamp}.png")
    print(f"\nSaving summary plot to {filename}...")
    plt.savefig(filename, dpi=300)
    #plt.show()

def analyze_and_plot_failures_old(results_df: pd.DataFrame):
    """
    Performs a detailed analysis of failure modes and generates a dedicated plot.

    Creates a stacked bar chart showing the counts of different error types at
    each location where failures occurred.
    """
    print("\n" + "="*50)
    print(" " * 15 + "FAILURE MODE ANALYSIS")
    print("="*50)

    # 1. Filter the DataFrame to include all non-successful runs
    failure_df = results_df[results_df['status'] != 'SUCCESS'].copy()
    num_failures = len(failure_df)

    if num_failures == 0 or 'error_type' not in failure_df.columns:
        print("No failures with 'error_type' recorded. Skipping failure plot.")
        return

    # 2. Aggregate the data: Count error_type by location
    # We use pivot_table to get the data in the perfect shape for a stacked bar chart.
    # NaN values (runs without a specific error) are filled with 0.
    failure_pivot = pd.pivot_table(
        failure_df,
        index='location',
        columns='error_type',
        aggfunc='size', # 'size' is a way to count occurrences
        fill_value=0
    )

    if failure_pivot.empty:
        print("No data to plot for failure analysis.")
        return

    print("\n--- Failure Counts by Location and Type ---")
    print(failure_pivot)

    # Calculate total errors for percentage calculation
    total_errors = failure_pivot.sum().sum()

    # 3. Create the plot
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the stacked bar chart. Using a log scale for the y-axis.
    failure_pivot.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        #logy=True, # Use a logarithmic scale for the counts
        width=0.8
    )

    ax.set_title('Breakdown of Simulation Failure Modes', fontsize=16)
    ax.set_xlabel('Component Location of Failure', fontsize=12)
    ax.set_ylabel('Number of Occurrences', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Error Type')

    # Add percentage labels to the stacked bars
    # This is a bit more complex for stacked bars, but very informative
    for c in ax.containers:
        # Get the labels (error types) for this container
        labels = [f"{v.get_height()/total_errors:.1%}" if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', weight='bold')

    # --- Finalize and Save ---
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    band_str = "".join(config.ENABLED_BANDS)
    filename = (f"failure_analysis_{config.SIM_MODE}_{band_str}_{config.GRID_GRANULARITY_GHZ}GHz_"
                f"{timestamp}.png")
    print(f"\nSaving failure analysis plot to {filename}...")
    plt.savefig(filename, dpi=300)
    plt.show()

def analyze_and_plot_failures(results_df: pd.DataFrame, directory="./"):
    """
    Performs a detailed analysis of failure modes using a multi-panel plot
    to clearly distinguish failure locations from failure reasons.
    """
    print("\n" + "="*50)
    print(" " * 15 + "FAILURE MODE ANALYSIS")
    print("="*50)

    failure_df = results_df[results_df['status'] != 'SUCCESS'].copy()
    num_failures = len(failure_df)

    if num_failures == 0 or 'error_type' not in failure_df.columns or 'location' not in failure_df.columns:
        print("No failures with 'error_type' and 'location' recorded. Skipping failure plot.")
        return

    failure_df.dropna(subset=['error_type', 'location'], inplace=True)
    if failure_df.empty:
        print("No valid failure data to plot.")
        return

    # --- Aggregate Data ---
    # 1. Count failures by location
    location_counts = failure_df['location'].value_counts()

    # 2. Count failures by error type
    error_type_counts = failure_df['error_type'].value_counts()

    print("\n--- Failure Counts by Location ---")
    print(location_counts)
    print("\n--- Failure Counts by Error Type ---")
    print(error_type_counts)

    cfg_str = (f"Mode: {config.SIM_MODE}, SPDC Mode: {config.SPDC_MODE}, Grid: {config.GRID_GRANULARITY_GHZ}GHz, "
               f"Filter: {config.FILTER_MODEL}, Detectors: {config.DETECTOR_TYPE}")

    # --- Create the Multi-Panel Plot ---
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Breakdown of Simulation Failure Modes\n'+cfg_str, fontsize=18)

    # --- Plot 1: Pie Chart of Failure Locations ---
    # Use a threshold to group small slices into an "Other" category for clarity
    threshold = 0.02 # Group slices smaller than 2%
    other_sum = location_counts[location_counts / num_failures < threshold].sum()
    main_counts = location_counts[location_counts / num_failures >= threshold]
    if other_sum > 0:
        main_counts['Other'] = other_sum

    ax1.pie(main_counts, labels=main_counts.index, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Where Failures Occur (by Location)', fontsize=14)

    # --- Plot 2: Bar Chart of Failure Reasons ---
    error_type_counts.plot(kind='bar', ax=ax2, color='#C44E52', alpha=0.8)
    ax2.set_title('Why Failures Occur (by Error Type)', fontsize=14)
    ax2.set_ylabel('Number of Occurrences')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # Add count labels on top of the bars
    ax2.bar_label(ax2.containers[0], label_type='edge', padding=3, fontsize=10)
    # Adjust y-limit to make space for labels
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.1)

    # --- Finalize and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ... (saving logic is the same) ...
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    band_str = "".join(config.ENABLED_BANDS)
    filename = (f"failure_analysis_{config.SIM_MODE}_{band_str}_{config.GRID_GRANULARITY_GHZ}GHz_"
                f"{timestamp}.png")
    print(f"\nSaving failure analysis plot to {filename}...")
    plt.savefig(directory+"/"+filename, dpi=300)
    #plt.show()

def get_config_summary() -> str:
    """
    Scans the config.py module and returns a formatted string of all settings.
    """
    summary = ["="*50, "SIMULATION CONFIGURATION", "="*50]
    # Get all uppercase global variables from the config module
    settings = {key: getattr(config, key) for key in dir(config) if key.isupper()}

    for key, value in settings.items():
        summary.append(f"{key:<35}: {value}")
    summary.append("="*50 + "\n")
    return "\n".join(summary)

def plot_performance_vs_distance(results_df: pd.DataFrame, directory="."):
    """
    Analyzes and plots the success rate and fidelity as a function of distance.
    """
    print("\n" + "="*50)
    print(" " * 15 + "PERFORMANCE VS. DISTANCE ANALYSIS")
    print("="*50)

    if 'distance_km' not in results_df.columns:
        print("'distance_km' column not found. Skipping distance plot.")
        return

    """
    # --- Aggregate Data by Distance ---
    # Calculate success rate for each distance
    success_rate_by_dist = results_df.groupby('distance_km')['status'].apply(
        lambda x: (x == 'SUCCESS').mean()
    ).reset_index(name='success_rate')
    """
    # --- Aggregate Data by Distance ---
    # 1. Calculate success rate for each distance
    # Using .size() for total count and .sum() for success count is more direct
    agg_data = results_df.groupby('distance_km')['status'].agg(
        total_runs='size',
        success_runs=lambda x: (x == 'SUCCESS').sum()
    ).reset_index()
    agg_data['success_rate'] = agg_data['success_runs'] / agg_data['total_runs']

    # Filter for successful runs to analyze fidelity
    success_df = results_df[results_df['status'] == 'SUCCESS']

    if not success_df.empty:
        fidelity_stats = success_df.groupby('distance_km')['fidelity'].agg(
            mean_fidelity='mean',
            std_fidelity='std'
        ).reset_index()
        # Merge the fidelity stats back into our main aggregated data
        agg_data = pd.merge(agg_data, fidelity_stats, on='distance_km', how='left')
        # Fill NaN for distances with zero successes to avoid plotting errors
        agg_data.fillna(0, inplace=True)
    else:
        # If no successes, create empty columns
        agg_data['mean_fidelity'] = 0
        agg_data['std_fidelity'] = 0

    # --- Create the Plot ---
    plt.style.use('seaborn-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.suptitle('Entanglement Swapping Performance vs. Fiber Length', fontsize=18)



    # --- Plot 1: Success Rate (Left Y-Axis) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Fiber Length (km)', fontsize=14)
    ax1.set_ylabel('Success Rate (pairs per use)', color=color1, fontsize=14)
    #ax1.plot(success_rate_by_dist['distance_km'], success_rate_by_dist['success_rate'],
    #         color=color1, marker='o', linestyle='-', label='Success Rate')
    ax1.plot(agg_data['distance_km'], agg_data['success_rate'],
             color=color1, marker='o', linestyle='-', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    # Use a log scale for success rate if it drops exponentially, which is common
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0)

    # --- Plot 2: Fidelity (Right Y-Axis) ---
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Fidelity', color=color2, fontsize=14)
    ax2.plot(agg_data['distance_km'], agg_data['mean_fidelity'],
             color=color2, marker='x', linestyle='--', label='Mean Fidelity')

    # Create the shaded error band for one standard deviation
    ax2.fill_between(
        agg_data['distance_km'],
        agg_data['mean_fidelity'] - agg_data['std_fidelity'],
        agg_data['mean_fidelity'] + agg_data['std_fidelity'],
        color=color2,
        alpha=0.2, # Make the shaded region semi-transparent
        label='Fidelity (±1 Std Dev)'
    )

    # Apply lowess smoothing for y (y1) and y2
    smoothed_y = lowess(agg_data['success_rate'], agg_data['distance_km'], frac=0.1)  # adjust `frac` for smoothness
    smoothed_y2 = lowess(agg_data['mean_fidelity'], agg_data['distance_km'], frac=0.1)

    # Plot smoothed trend lines
    ax1.plot(smoothed_y[:, 0], smoothed_y[:, 1], color='black', linestyle='--', label='Rate trend')
    ax2.plot(smoothed_y2[:, 0], smoothed_y2[:, 1], color='green', linestyle='--', label='Fidelity trend')

    """
    # Use seaborn for a beautiful boxplot
    # We plot this first so it's in the background
    sns.boxplot(
        x='distance_km',
        y='fidelity',
        data=success_df,
        ax=ax2,
        boxprops=dict(facecolor=color2, alpha=0.3),
        whiskerprops=dict(color=color2),
        capprops=dict(color=color2),
        medianprops=dict(color=color2, linewidth=2)
    )
    """
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False) # Turn off the grid for the boxplot axis for clarity

    # Create a unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    # --- Finalize and Save ---
    # Set x-axis ticks to be at reasonable intervals
    x_ticks = ax1.get_xticks()
    if len(x_ticks) > 10: # If too many ticks, thin them out
        ax1.set_xticks(x_ticks[::int(len(x_ticks)/10)])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"performance_vs_distance_{timestamp}.png"
    print(f"\nSaving performance vs. distance plot to {filename}...")
    plt.savefig(directory+"/"+filename, dpi=300)
    #plt.show()

def serialize_qstate(qubits) -> list:
    """
    Extracts the density matrix from one or more qubits and converts it
    to a JSON-serializable list of lists.

    Args:
        qubits: A single qubit or a list of qubits.

    Returns:
        list: The density matrix as a list of lists of complex numbers' string representations.
              Returns None if the qubit has no state.
    """

    if qubits is None or (isinstance(qubits, list) and not qubits):
        return None

    # Ensure we are working with a list
    if not isinstance(qubits, list):
        qubits = [qubits]

    if qubits[0].qstate is None:
        return None

    # Get the density matrix as a NumPy array
    dm_array = qapi.reduced_dm(qubits)
    """
    # Convert the NumPy array to a Python list of lists
    # We must also convert the complex numbers to a parsable format.
    # A list [real, imag] is a common way.
    def complex_to_list(c):
        return [c.real, c.imag]

    # Use a nested list comprehension to apply the conversion
    serializable_list = [[complex_to_list(c) for c in row] for row in dm_array]
    """
    # Helper function to convert a complex NumPy number to a list of standard Python floats.
    def complex_to_pyfloat_list(c):
        # By explicitly calling float(), we convert from np.float64 to a standard float.
        return [float(c.real), float(c.imag)]

    # Use a nested list comprehension to apply this conversion to every element.
    serializable_list = [
        [complex_to_pyfloat_list(element) for element in row]
        for row in dm_array
    ]

    return serializable_list

def convert_hz_to_nm(frequency_hz: float) -> float:
    """
    Converts a frequency in Hertz (Hz) to its corresponding wavelength in nanometers (nm).

    This calculation assumes the wave is propagating in a vacuum.

    Args:
        frequency_hz (float): The frequency of the wave in Hertz.

    Returns:
        float: The corresponding wavelength in nanometers.

    Raises:
        ValueError: If the frequency is zero or negative, as this would lead
                    to a non-physical wavelength.
    """
    if frequency_hz <= 0:
        raise ValueError("Frequency must be a positive number to convert to wavelength.")

    # Calculate wavelength in meters: λ = c / f
    wavelength_meters = config.SPEED_OF_LIGHT_IN_VACUUM / frequency_hz

    # Convert meters to nanometers (1 meter = 1e9 nanometers)
    wavelength_nanometers = wavelength_meters * 1e9

    return wavelength_nanometers
