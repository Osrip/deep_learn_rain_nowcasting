import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Use correct wdir
os.chdir('../../..')

def plot_bimodal_distributions(file_path):
    """
    Plots a 10x10 grid of randomly selected bimodal distributions from a CSV file.
    Each histogram bin has equal width.

    Args:
        file_path (str): Full path to the CSV file
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return

    # Extract just the filename without extension for the output file
    file_dir = os.path.dirname(file_path)
    file_basename = os.path.basename(file_path)
    file_pattern = os.path.splitext(file_basename)[0]

    # Load the CSV file
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Get the distribution bin columns (all columns except time, h, w)
    bin_columns = [col for col in df.columns if col not in ['time', 'h', 'w']]

    # Check if there are enough distributions
    sample_size = min(100, len(df))
    if len(df) < 100:
        print(f"Found {len(df)} distributions, using all available.")
    else:
        print(f"Found {len(df)} distributions, randomly sampling 100.")

    # Randomly sample the distributions
    sample_indices = random.sample(range(len(df)), sample_size)
    sample_df = df.iloc[sample_indices].reset_index(drop=True)

    # Create a 10x10 grid (or adjust if fewer samples)
    rows = cols = 10
    if sample_size < 100:
        rows = cols = int(np.ceil(np.sqrt(sample_size)))

    # Set up the figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Number of bins
    num_bins = len(bin_columns)

    # Create equal-width bins (just use indices instead of actual values)
    bin_indices = np.arange(num_bins)
    bin_edges = np.arange(num_bins + 1)  # One more edge than bins

    # Plot each distribution
    for i in range(sample_size):
        ax = axes[i]
        row = sample_df.iloc[i]

        # Get distribution values for this row
        distribution = [row[col] for col in bin_columns]

        # Plot the distribution as a histogram with equal-width bins
        # We use bin indices (0, 1, 2, ...) instead of actual bin values
        weights = distribution
        ax.hist(bin_indices, bins=bin_edges, weights=weights,
                alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add a title with time and position coordinates
        time_str = str(row['time']).split('T')[0]
        ax.set_title(f"Time: {time_str}, Pos: ({int(row['h'])},{int(row['w'])})", fontsize=8)

        # Set x-ticks to use bin indices (simplified)
        ax.set_xticks([0, num_bins // 3, 2 * num_bins // 3, num_bins - 1])
        ax.set_xticklabels(['0', f'{num_bins // 3}', f'{2 * num_bins // 3}', f'{num_bins - 1}'], fontsize=6)

        # Set y-ticks
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis='y', labelsize=6)

        # Add labels for edge plots only
        if i % cols == 0:  # Left edge
            ax.set_ylabel('Probability', fontsize=8)
        if i >= (rows - 1) * cols:  # Bottom edge
            ax.set_xlabel('Bin Index', fontsize=8)

    # Hide unused subplots if any
    for i in range(sample_size, len(axes)):
        axes[i].set_visible(False)

    # Add an overall title
    plt.suptitle(f"Bimodal Distributions from {file_pattern}", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save the figure to the same directory as the input file
    output_filename = os.path.join(file_dir, f"{file_pattern}_equal_width_hist.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")

    # Show the plot
    plt.show()


# Full path to the file
file_path = "/home/jan/Programming/remote/first_CNN_on_Radolan/notebooks/bimodality_dlbd/dlbd_results/bimodal_distributions_20250516_110510.csv"

# Run the function
plot_bimodal_distributions(file_path)