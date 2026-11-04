import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_data_from_json(files):
    """Loads summarized data from a list of JSON files."""
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            data_list.append({
                'config': data['configuration_name'],
                'avg_fidelity': data['metrics']['fidelity']['mean'],
                'std_fidelity': data['metrics']['fidelity']['std_dev'],
                'avg_photons': data['metrics']['avg_photon_number']
            })
    return pd.DataFrame(data_list)

def plot_line_chart(df):
    """
    Generates a line chart showing metrics as a function of length, without error bars.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Plot 1: Fidelity vs. Length on the left Y-axis ---
    color = 'tab:blue'
    ax1.set_xlabel('Internode Length (km)')
    ax1.set_ylabel('Average Fidelity', color=color)
    ax1.plot(df['length'], df['avg_fidelity'], color=color, marker='o', label='Avg. Fidelity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--')
    ax1.set_ylim(0, 1)

    # --- Plot 2: Photon Number vs. Length on the right Y-axis ---
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('ebits per use', color=color)
    ax2.plot(df['length'], df['avg_photons'], color=color, marker='s', linestyle='--', label='Bits per use')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, .2)

    # Title and final layout adjustments
    plt.title('Entanglement Fidelity, ebits per use vs. Internode Length (DM, 1NM, PNR)')
    fig.tight_layout()

    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.savefig("performance_vs_length.png", dpi=300)
    plt.show()

def main():
    json_files = glob.glob('*.json')
    if not json_files:
        print("Error: No .json result files found in this directory.")
        return

    df = load_data_from_json(json_files)

    # Extract the numerical length from the 'config' string column
    df['length'] = df['config'].str.extract(r'(\d+)').astype(int)

    # Sort the DataFrame by length to ensure the line plot connects points correctly
    df = df.sort_values('length').reset_index(drop=True)

    print("\nProcessed Data with Extracted Lengths:")
    print(df)

    plot_line_chart(df)

if __name__ == "__main__":
    main()
