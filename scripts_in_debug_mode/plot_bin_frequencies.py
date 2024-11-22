# --- Plotting the bin frequencies ---
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Call this at the end of 
calc_bin_frequencies()
"""

# Generate bin numbers
bin_numbers = np.arange(len(bin_frequencies))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(bin_numbers, bin_frequencies, align='center', alpha=0.7, color='C0')
plt.yscale('log')
plt.xlabel('Bin Number')
plt.ylabel('Frequency')
plt.title('Bin Frequencies')

# Create labels for the legend with bin bounds
bin_labels = []
for i in range(len(bin_frequencies)):
    lower_bound = linspace_binning_with_max_unnormed[i]
    upper_bound = linspace_binning_with_max_unnormed[i + 1]
    bin_labels.append(f'Bin {i}: [{lower_bound:.2f}, {upper_bound:.2f})')

# Create custom legend handles
handles = [mpatches.Patch(color='C0', label=label) for label in bin_labels]

# Add the legend to the plot
plt.legend(handles=handles, title='Bin Bounds', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for the legend
plt.tight_layout()

plt.savefig('plots/bin_frequencies.png', dpi=200, bbox_inches='tight')

# Display the plot
plt.show()