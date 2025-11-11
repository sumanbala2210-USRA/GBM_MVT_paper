import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- 1. Configuration ---
#csv_filename = '07_plotted_data_MVT_snr_vs_median_mvt_ms-2.csv'
csv_filename = '07_Norris_summary_results_0.01.csv'
csv_filename = '07_Norris_summary_results.csv'


x_col = 'MVT_snr'
y_col = 'median_mvt_ms'
group_col = 'decay_time'
marker_col = 'bin_width_ms'

output_base_name = f"{csv_filename[0:2]}_MVT_vs_{x_col}_by_{group_col}"
output_png = f"{output_base_name}.png"
output_pdf = f"{output_base_name}.pdf"

# --- 2. Publication Style Configuration ---

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'legend.title_fontsize': 20,
    'font.family': 'serif'
})

# --- 3. Data Preparation ---
try:
    df = pd.read_csv(csv_filename)
    df['decay_time'] = df['decay_time'] * 1000  # Use decay_time1 for plotting
    df['rise_time'] = df['rise_time'] * 1000  # Use rise_time1 for plotting
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

df_filtered = df[df[y_col] > 0].copy()
df_filtered = df_filtered[df_filtered['rise_time'] == 10.0].copy()


# --- 4. Plot Styling ---
# Fixed color map for decay_time
color_map = {
    1.0:    'tab:blue',
    3.0:    'tab:orange',
    10.0:   'black',
    30.0:   'tab:red',
    100.0:  'tab:purple',
    300.0:  'tab:brown',
    1000.0: 'tab:pink',
}

# --- MODIFIED PART: Using a list of marker property dictionaries for maximum distinction ---
all_possible_bins = sorted(df_filtered[marker_col].unique())

# Each dictionary defines a unique, visually distinct marker style.
# The 's_mult' key is a size multiplier to balance visual weight.
marker_definitions = [
    {'marker': 'o', 'fillstyle': 'none',  's_mult': 1.0},  # Hollow Circle
    {'marker': 's', 'fillstyle': 'none',  's_mult': 1.0},
    {'marker': 'o', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Circle
    {'marker': 's', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Square
    {'marker': '^', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Triangle
    {'marker': 'D', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Diamond # Hollow Square
    {'marker': 'X', 'fillstyle': 'full',  's_mult': 1.1},  # Filled X (larger)
    {'marker': '*', 'fillstyle': 'full',  's_mult': 1.4},  # Star (even larger)
    {'marker': 'P', 'fillstyle': 'full',  's_mult': 1.2}   # Plus (larger)
]
marker_map = {bin_val: marker_definitions[i % len(marker_definitions)] for i, bin_val in enumerate(all_possible_bins)}


# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(10, 8.3))
BASE_MARKERSIZE = 7  # Define a base size for markers

for decay_val, d_df in df_filtered.groupby(group_col):
    for bin_val, b_df in d_df.groupby(marker_col):
        subset = b_df
        if subset.empty:
            continue
         
        y_error = [subset['mvt_err_lower'], subset['mvt_err_upper']]
        
        # Get the custom marker properties from our map
        marker_props = marker_map.get(bin_val, marker_definitions[0])
        
        ax.errorbar(
            x=subset[x_col], y=subset[y_col], yerr=y_error,
            # Apply custom marker properties individually
            marker=marker_props.get('marker'),
            markersize=BASE_MARKERSIZE * marker_props.get('s_mult', 1.0),
            fillstyle=marker_props.get('fillstyle'),
            color=color_map.get(decay_val, 'black'),
            capsize=3, linestyle='none', alpha=0.9,
            markeredgecolor='black',
            markeredgewidth=0.7
        )

# --- 6. Smart Legend Creation ---
color_legend_elements = [Line2D([0], [0], color=color_map.get(d, 'black'), lw=2, label=f'{d} ms') for d in sorted(df_filtered[group_col].unique())]

# Build legend handles using the custom marker properties
marker_legend_elements = []
for b in all_possible_bins:
    props = marker_map.get(b)
    marker_legend_elements.append(Line2D([0], [0], 
                                         marker=props.get('marker'), 
                                         fillstyle=props.get('fillstyle'),
                                         markersize=10 * props.get('s_mult', 1.0),
                                         color='gray', label=f'{b}',
                                         linestyle='None', 
                                         markeredgecolor='black', 
                                         markeredgewidth=0.7))




# Create the FIRST legend for bin_width_ms (marker) in the upper right
legend1 = ax.legend(handles=marker_legend_elements, title='Bin Width (ms)', loc='upper center', ncol=2)
ax.add_artist(legend1)

# Create the SECOND legend for decay_time (color) and place it below the first one
legend2 = ax.legend(handles=color_legend_elements, title='Decay Time', loc='upper right')


# --- 7. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(0.7, 100)
x_label = x_col.replace('_', ' ').title()
y_label = y_col.replace('_', ' ').title()
x_label = x_label.replace('Mvt Snr', r'SNR$_{{\mathrm{MVT}}}$')
y_label = y_label.replace('Median Mvt Ms', 'Median MVT (ms)')
ax.text(0.02, 0.07, 'Rise Time 10 ms', transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()


# --- 8. Save Output ---
plt.savefig(output_png, dpi=300)
plt.savefig(output_pdf)

print(f"Plot saved to '{output_png}' and '{output_pdf}'")