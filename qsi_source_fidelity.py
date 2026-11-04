# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: qsi_source_fidelity.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports
import numpy as np
import time
import json
import uuid
import ast, re
from qsi.coordinator import Coordinator
from qsi.state import State, StateProp
import config
from utils import convert_hz_to_nm
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def print_json(data, title=""):
    if title: print(f"\n--- {title} ---")
    print(json.dumps(data, indent=2))


def print_config(data, title=""):
    if title: print(f"\n--- {title} ---")
    for name in dir(data):
        if name.isupper():
            value = getattr(config, name)
            print(value)

def parse_custom_repr_string(s: str) -> dict:
    """
    Safely parses a string containing a Python dictionary representation,
    including custom object __repr__ strings like DenseDMRepr.

    Args:
        s (str): The input string to parse.

    Returns:
        dict: The parsed dictionary.
    """
    
    processed_s = re.sub(r"np\.float64\((.*?)\)", r"\1", s)

    try:
        # Use ast.literal_eval on the processed string.
        # It can now handle the single quotes and basic types correctly.
        parsed_dict = ast.literal_eval(processed_s)
        return parsed_dict
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing string: {e}")
        return None


def calculate_qsi_fidelity(qsi_state: State, target_pure_state: np.ndarray) -> float:
    """
    Calculates the fidelity of a QSI state with respect to a pure target state.

    Args:
        qsi_state (State): The (potentially mixed) state from the simulation.
        target_pure_state (np.ndarray): A 1D numpy array (ket vector) for the ideal pure state.

    Returns:
        float: The fidelity (a value between 0 and 1).
    """
    # Ensure the target is a proper ket (column vector)
    if target_pure_state.ndim == 1:
        target_pure_state = target_pure_state.reshape(-1, 1) # Make it a column vector

    # Get the density matrix from the QSI state
    rho = qsi_state.state

    # Get the conjugate transpose of the target state (the bra)
    bra = target_pure_state.conj().T

    # Calculate <ψ|ρ|ψ>
    # The result is a 1x1 matrix, so we extract the single complex number
    fidelity_complex = (bra @ rho @ target_pure_state)[0, 0]

    # Fidelity is the real part of this value (the imaginary part should be ~0)
    return float(np.real(fidelity_complex))


def plot_fidelity_vs_distance(all_results: list):
    """
    Processes a list of simulation results and plots the final fidelity
    as a function of symmetric fiber distance.

    Args:
        all_results (list): A list of result dictionaries. Each dictionary
                            must contain 'distance_km' and 'final_state' keys.
    """
    if not all_results:
        print("No results to plot.")
        return

    # Convert the results data into a Pandas DataFrame
    results_df = pd.DataFrame(all_results)

    # --- 2. Aggregate the data for plotting ---
    # Group by distance and calculate the mean and standard deviation of fidelity
    agg_data = results_df.groupby('distance')['fidelity'].agg(
        mean_fidelity='mean',
        std_fidelity='std'
    ).reset_index()

    print("\n--- Aggregated Fidelity Data ---")
    print(agg_data)

    # --- 3. Create the Plot ---
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Entanglement Fidelity vs. Symmetric Fiber Length)', fontsize=16)

    # Plot the mean fidelity as a line
    ax.plot(agg_data['distance'], agg_data['mean_fidelity'],
             color='tab:red', marker='o', linestyle='-', label='Mean Fidelity')

    # Create the shaded error band for one standard deviation
    ax.fill_between(
        agg_data['distance'],
        agg_data['mean_fidelity'] - agg_data['std_fidelity'],
        agg_data['mean_fidelity'] + agg_data['std_fidelity'],
        color='tab:red',
        alpha=0.2, # Make the shaded region semi-transparent
        label='Fidelity (±1 Std Dev)'
    )

    ax.set_xlabel('Fiber Length (km)', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, which='both', linestyle='--')
    ax.legend(loc='upper right')

    # --- 4. Finalize and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"fidelity_vs_distance_{timestamp}.png"
    print(f"\nSaving plot to {filename}...")
    plt.savefig(filename, dpi=300)
    #plt.show()


def main():
    # Create coordinator and start the modules:
    coordinator = Coordinator(port=25000)
    # Coherent source
    zalm = coordinator.register_component(module="zalm_source.py", runtime="python")
    fiber1 = coordinator.register_component(module="fiber.py", runtime="python")
    fiber2 = coordinator.register_component(module="fiber.py", runtime="python")

    # Run the coordinator process
    coordinator.run()
    time.sleep(1) # Give the component time to start

    psi_minus_ket = (1 / np.sqrt(2)) * np.array([0, 1, -1, 0])
    num_runs = 10000

    #distances_to_sweep = range(1, 51, 1) # 0 to 50 km in 1km steps
    distances_to_sweep = range(0, 50, 1) # 0 to 50 km in 1km steps
    results = []

    counter = 0
    success = 0

    for dist_km in [0]: #distances_to_sweep:
        # Set both fibers to the same length for this batch of runs
        fiber1.set_param("length", dist_km)
        fiber1.set_param("n", 1.45)
        fiber1.send_params()

        fiber2.set_param("length", dist_km)
        fiber2.set_param("n", 1.45)
        fiber2.send_params()

        for _ in range(num_runs):
            counter += 1
            response = coordinator.send_and_return_response(zalm.port, {"msg_type": "state_init"})
            response = coordinator.send_and_return_response(zalm.port, {"msg_type": "channel_query"})

            print(f"Run # {counter}")
            if response["message"] == "None":
                continue



            messages = parse_custom_repr_string(response["message"])


            if messages['status'] == "SUCCESS":
                success += 1
                # Create StateProp for the first photon (q1)
                Signal1 = StateProp(
                    wavelength=convert_hz_to_nm(messages['q1_frequency']),
                    bandwidth=config.PHOTON_FWHM_GHZ,
                    truncation=2,
                    polarization=messages['q1_polarisation'],
                    state_type="light"
                )

                # Create StateProp for the second photon (q2)
                Signal2 = StateProp(
                    wavelength=convert_hz_to_nm(messages['q2_frequency']), # This can be different from q1!
                    bandwidth=config.PHOTON_FWHM_GHZ,
                    truncation=2,
                    polarization=messages['q2_polarisation'],
                    state_type="light"
                )

                final_qsi_state = State(empty=True)
                list_of_lists = messages['final_dm']
                complex_list_of_lists = [
                    [complex(c[0], c[1]) for c in row]
                    for row in list_of_lists
                ]

                density_matrix_array = np.array(complex_list_of_lists, dtype=complex)
                final_qsi_state.state = density_matrix_array
                final_qsi_state.state_props = [Signal1, Signal2]
                final_qsi_state.dimensions = 4

                prop_signal1 = final_qsi_state.state_props[0]
                prop_signal2 = final_qsi_state.state_props[1]

                response1, operators1 = fiber1.channel_query(final_qsi_state, {"input": prop_signal1.uuid})
                final_qsi_state.apply_kraus_operators(operators1, [prop_signal1])

                response2, operators2 = fiber2.channel_query(final_qsi_state, {"input": prop_signal2.uuid})
                final_qsi_state.apply_kraus_operators(operators2, [prop_signal2])

                fidelity = calculate_qsi_fidelity(final_qsi_state, psi_minus_ket)

                results.append({'fidelity': fidelity})

    print(f"{success} / {counter} qubits successfully entangled")
    print(f"bits per use is {success/counter}")
    fidelities = [r["fidelity"] for r in results]
    run_value = 1
    run_type = f"DEFAULT_DET_EFF_{run_value}"
    summary_data = {
        "configuration_name": run_value,
        "num_trials": num_runs,
        "metrics": {
            "fidelity": {
                "mean": np.mean(fidelities),
                "std_dev": np.std(fidelities)
            },
            "avg_photon_number": success/counter,
        }
    }

    # Save the aggregated data to a JSON file
    output_filename = f"{run_type}.json"
    with open(output_filename, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"\nSuccessfully saved results to {output_filename}")
    print(f"  - Average Fidelity: {summary_data['metrics']['fidelity']['mean']:.4f} "
          f"(+/- {summary_data['metrics']['fidelity']['std_dev']:.4f})")
    print(f"  - Average Photon Number: {summary_data['metrics']['avg_photon_number']:.4f} ")
    coordinator.terminate()

    #plot_fidelity_vs_distance(results)

if __name__ == '__main__':
    main()
