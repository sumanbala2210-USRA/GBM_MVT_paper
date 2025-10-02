import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Define the list of data files to plot ---
data_path = Path('Lc_collections/plot_data')
['DATA_1_3_6.2_complex_pulse_long.npz', 'DATA_5_5_4.5_complex_pulse_short.npz', 'DATA_100_2_4.5_complex_pulse_short_2p3ms.npz', 'DATA_10_1_6.2_complex_pulse_short_2p10ms.npz']
data_files = [
    data_path / 'DATA_1_3_6.2_complex_pulse_long.npz',
    data_path / 'DATA_5_5_4.5_complex_pulse_short.npz',
    data_path / 'DATA_10_1_6.2_complex_pulse_short_2p10ms.npz',
    data_path / 'DATA_100_2_4.5_complex_pulse_short_2p3ms.npz'
    #data_path / 'DATA_5_2_6.2_complex_pulse_long.npz',
    #data_path / 'DATA_5_2_6.2_complex_pulse_short.npz',
    #data_path / 'DATA_5_2_6.2_complex_pulse_short_2p10ms.npz',
    #data_path / 'DATA_5_2_6.2_complex_pulse_short_2p3ms.npz',
]

# --- 2. Set up the subplot grid ---
num_files = len(data_files)
ncols = 1
nrows = (num_files + ncols - 1) // ncols

plt.style.use('seaborn-v0_8-whitegrid')
# MODIFICATION 1: Use constrained_layout for better automatic spacing
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(10, 15),
    squeeze=False,
    sharex=True,
    #tight_layout=True,
    #constrained_layout=True  # This is often better than tight_layout
)

axes = axes.flatten()
plt.subplots_adjust(hspace=0)
fig.text(0.5, 0.08, 'Time (s)', ha='center', fontsize=18)
# --- 3. Loop through each file and plot on a separate subplot ---
for i, data_file in enumerate(data_files):
    ax = axes[i]

    try:
        data = np.load(data_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"Warning: Data file not found at {data_file}. Skipping.")
        ax.text(0.5, 0.5, 'File not found', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    # --- 4. Extract data and parameters for the current plot ---
    times = data['times']
    total_counts = data['total_counts']
    source_only_counts = data['source_only_counts']
    ideal_background = data['ideal_background_counts'].item()
    params = data['parameters'].item()

    bin_width = params.get('bin_width_for_plot', 0.01)
    t_start = params.get('t_start')
    t_stop = params.get('t_stop')
    background_level_cps = params['background_level'] * params.get('scale_factor', 1.0)
    
    pulse_shape_name_dict = {
            'complex_pulse_short': 'Short',
            'complex_pulse_long': 'Long',
            'complex_pulse_short_2p10ms': 'Short + Fixed 10ms (4.95s)',
            'complex_pulse_short_2p3ms': 'Short + Fixed 3ms (4.95s)',
        }
    pulse_shape_name_dict = {
            'complex_pulse_short': 'Short',
            'complex_pulse_long': 'Long',
            'complex_pulse_short_2p10ms': 'Short',
            'complex_pulse_short_2p3ms': 'Short',
        }
    pulse_shape_nice = pulse_shape_name_dict.get(params['pulse_shape'], params['pulse_shape'])
    
    # --- 5. Recreate the original plot on the current subplot (ax) ---
    ax.step(times, total_counts, where='mid', label='Total Signal', color='rosybrown', lw=1.5)
    ax.fill_between(times, total_counts, step="mid", color='rosybrown', alpha=0.6)
    
    ax.step(times, source_only_counts, where='mid', label='Source Signal', color='darkgreen', lw=1.5)
    ax.fill_between(times, source_only_counts, step="mid", color='darkgreen', alpha=0.4)

    ax.axhline(ideal_background, color='k', linestyle='--', label=f'Ideal Bkg ({background_level_cps:.1f} cps)')

    if params['pulse_shape'] in ['complex_pulse_short_2p10ms', 'complex_pulse_short_2p3ms']:
        # Create the vertical line and store its handle
        pulse_sigma = params['pulse_shape'][-4:-2]  # Extract '10' or '3' from the shape name
        if pulse_sigma == 'p3':
            pulse_sigma = 3
        pulse_line = ax.axvline(4.95, color='orange', linestyle='-.', label=f'Fixed Pulse at 4.95s\nSigma = {pulse_sigma} ms\nPeak Amp Ratio = 2', lw=1.0, zorder=50)

        # Use only that line in the legend
        ax.legend(handles=[pulse_line], loc='upper center', bbox_to_anchor=(0.5, 1),
              fontsize=14, framealpha=0.9, edgecolor='lightgray', frameon=True, ncol=1)


    ax.vertical_line = ax.axvline(params['position'], color='tab:red', linestyle='--', label='Varying Pulse Position', lw=1.0, zorder=50)
    #ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel(f"Counts per {int(bin_width*1000)} ms Bin", fontsize=16, labelpad=10, fontfamily='serif' )
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.set_xlim(t_start, 15)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=14)

    sigma = params.get('sigma', 'N/A')
    position_shift = params.get('position', 'N/A')
    annotation_text = f"{pulse_shape_nice}\n" \
                      f"Varying {int(sigma*1000)} ms ({position_shift}s)\n" \
                      f"Overall Amp: {params['overall_amplitude']}\n" \
                      f"Peak Amp Ratio: {params['peak_amp_ratio']}\n" 
    props = dict(boxstyle='round,pad=0.5', facecolor='lightgoldenrodyellow', alpha=0.8, edgecolor='lightgray')
    ax.text(0.97, 0.97, annotation_text,
        transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=props, fontfamily='serif', fontsize=14)

    #ax.set_xlim(t_start, t_stop)
    if i == 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
                fontsize=14, framealpha=0.9, edgecolor='lightgray', frameon=True, ncol=1)

# --- 6. Clean up unused subplots ---
for j in range(num_files, len(axes)):
    axes[j].axis('off')

# --- 7. Save the final plot and then show it ---
output_filename = 'combined_light_curves.png'
plt.subplots_adjust(hspace=0.02, top=0.92)
fig.text(0.5, 0.08, 'Time (s)', ha='center', fontsize=18)

# MODIFICATION 2: Save the figure with bbox_inches='tight' to ensure everything fits
plt.savefig(output_filename, dpi=100, bbox_inches='tight')
#plt.subplots_adjust(hspace=0)
#plt.subplots_adjust(hspace=0, bottom=0.8)

print(f"Plot successfully saved to: {output_filename}")

plt.show()