import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_json_metrics(directory_path, output_filename="source_fildelity_photon_jitter_new4.pdf"):
    data_list = []

    # 1. Parse JSON files
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Extracting nested values
                    data_list.append({
                        "config": float(data["configuration_name"]),
                        #"config": data["configuration_name"],
                        "fidelity_mean": data["metrics"]["fidelity"]["mean"],
                        "fidelity_std": data["metrics"]["fidelity"]["std_dev"],
                        "avg_photon": data["metrics"]["avg_photon_number"]
                    })
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Skipping {filename} due to error: {e}")

    # Convert to DataFrame and sort for consistent plotting
    df = pd.DataFrame(data_list).sort_values(by="config")
    # You also need this if names are values
    #df['config'] = df['config'].astype(str).str.replace('.0', '', regex=False)

    # 2. Setup Plot Style (Journal Grade)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "figure.dpi": 300
    })

    plt.rcParams.update({
        #"figure.dpi": 300,           # High-res display in your IDE
        "savefig.dpi": 600,          # 600 DPI for saved files
        "font.family": "serif",      # Match journal serif fonts (Times New Roman)
        #"font.serif": ["Times New Roman"],
        "ps.usedistiller": "xpdf",   # Ensures better EPS export
        "pdf.fonttype": 42,          # VERY IMPORTANT: Embeds fonts in the PDF
        "ps.fonttype": 42            # so the journal doesn't reject it
    })

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 3. Plot Y1: Fidelity (Left Axis)
    color_y1 = '#1f77b4'  # Professional Blue
    ax1.set_xlabel('SPDC jitter (ps)', fontweight='bold')
    ax1.set_ylabel('Fidelity', color=color_y1, fontweight='bold')

    lowess = sm.nonparametric.lowess(df['fidelity_mean'], df['config'], frac=0.5)
    ax1.plot(lowess[:, 0], lowess[:, 1], color='darkblue',
             linewidth=2.5, linestyle='-', label='LOWESS Trend', zorder=5)

    ax1.set_ylim(0.6, 1.0)
    # Error bar plot
    #eb = ax1.errorbar(
    #    df['config'], df['fidelity_mean'], yerr=df['fidelity_std'],
    #    fmt='o-', color=color_y1, capsize=5, elinewidth=1,
    #    markeredgecolor='black', label='Fidelity (Mean ± StdDev)'
    #)
    line1 = ax1.plot(
        df['config'], df['fidelity_mean'],
        'o-', color=color_y1, alpha=0.8, label='Fidelity'
    )
    ax1.tick_params(axis='y', labelcolor=color_y1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    #ax1.set_xscale('log')

    # 4. Plot Y2: Avg Photon Number (Right Axis)
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color_y2 = '#d62728'  # Professional Red
    ax2.set_ylabel('Source efficiency (ebits/use)', color=color_y2, fontweight='bold')
    ax2.set_yscale('log')

    line2 = ax2.plot(
        df['config'], df['avg_photon'],
        's--', color=color_y2, alpha=0.8, label='Source efficiency'
    )
    ax2.tick_params(axis='y', labelcolor=color_y2)

    # Handle X-axis labels (rotate if they are long)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 5. Legend and Final Touches
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', frameon=True)

    # Horizontal average line (Green)
    y1_avg = df['fidelity_mean'].mean()
    #ax1.axhline(y=y1_avg, color='green', label='Average Fidelity')
    # Vertical line at 0.24 (Cyan)
    line_x = 6.24
    ax1.axvline(line_x, color='black', alpha=0.6, linestyle='--', label='ratio')
    ax1.annotate(
    '$\sigma_{j}$/$\sigma_{t}$ = 1',        # The text
    xy=(line_x, 0.92),          # Point to (x, y). y=0.5 is middle of plot
    xytext=(line_x + 1.5, 0.915),# Where to put the text (offset)
    fontsize=12,
    fontweight='bold',
    arrowprops=dict(           # The arrow style
        facecolor='black',
        shrink=0.05,
        width=2,
        headwidth=8
    )
)
    # Vertical line at 0.8 (Magenta)
    #ax1.axvline(x=0.8, color='magenta', linestyle='--', label='Channel Spacing')

    #plt.title('Performance Metrics by SPDC MODE and DETECTOR TYPE', fontsize=14, pad=20)
    fig.tight_layout()

    # Save the plot
    plt.savefig(output_filename, format="pdf", bbox_inches='tight')
    #plt.show()
    print(f"Plot saved as {output_filename}")

# --- Execution ---
# Replace 'your_folder_path' with the actual path to your JSON files
plot_json_metrics('run_source_fildelity_photon_jitter')
