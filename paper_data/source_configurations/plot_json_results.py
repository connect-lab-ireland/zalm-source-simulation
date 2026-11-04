import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot_grouped_bar_chart(df):
    """
    Generates a grouped bar chart, which is excellent for this kind of comparison.
    A double Y-axis is often confusing and is generally discouraged in data visualization.
    """
    labels = df['config']
    fidelity_means = df['avg_fidelity']
    photon_means = df['avg_photons']
    fidelity_std = df['std_fidelity']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plotting the bars
    rects1 = ax.bar(x - width/2, fidelity_means, width, label='Avg. Fidelity',
                    #yerr=fidelity_std, capsize=5, color='skyblue')
                    yerr=0, capsize=5, color='skyblue')
    rects2 = ax.bar(x + width/2, photon_means, width, label='Avg. bits per use',
                    yerr=0, capsize=5, color='sandybrown')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Value')
    ax.set_title('Source Performance by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Set Y-axis to start from a reasonable value if all values are high
    ax.set_ylim(bottom=min(0, df['avg_fidelity'].min() - 0.1, df['avg_photons'].min() - 0.1))

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("source_performance_comparison.png", dpi=300)
    plt.show()

def main():
    # Find all .json files in the current directory
    json_files = glob.glob('*.json')
    if not json_files:
        print("Error: No .json result files found in this directory.")
        print("Please run 'run_single_configuration.py' first.")
        return

    print(f"Found files: {', '.join(json_files)}")

    # Load data into a pandas DataFrame
    df = load_data_from_json(json_files)

    # Sort by configuration name for consistent plot order
    df = df.sort_values('config').reset_index(drop=True)

    print("\nAggregated Data:")
    print(df)

    # Generate the plot
    plot_grouped_bar_chart(df)


if __name__ == "__main__":
    main()
