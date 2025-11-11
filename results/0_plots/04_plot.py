import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- 1. Configuration ---
csv_filename = '04_plotted_data_MVT_snr_vs_median_mvt_ms-2.csv'
x_col = 'MVT_snr'
y_col = 'median_mvt_ms'
group_col = 'sigma'
marker_col = 'bin_width_ms' # New column for marker style

# Generate the output filename automatically using your specified format
output_base_name = f"{csv_filename[0:2]}_MVT_vs_{x_col}_by_{group_col}"
output_png = f"{output_base_name}.png"
output_pdf = f"{output_base_name}.pdf"


# --- 2. Publication Style Configuration ---
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 12, 'legend.title_fontsize': 14,
    'font.family': 'serif'
})
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'legend.title_fontsize': 18,
    'font.family': 'serif'
})

# --- 3. Data Preparation ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

# Filter only for valid data points. The sigma >= 1.0 filter is removed.
df_filtered = df[(df[y_col] > 0) & (df[x_col] > 0)].copy()
df_filtered['sigma'] = df_filtered['sigma'].astype(float)*1000  # Ensure sigma is float for proper mapping


# --- 4. Plot Styling (with Consistent Color and Marker Maps) ---
# Create a permanent color map for sigma values
all_possible_sigmas = sorted(df_filtered[group_col].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(all_possible_sigmas)))
color_map = {
    1.0:    'tab:blue',
    3.0:    'tab:orange',
    10.0:   'black',
    30.0:   'tab:red',
    100.0:  'tab:purple',
    300.0:  'tab:brown',
    1000.0: 'tab:pink',
}

# Create a permanent marker map for bin_width_ms values
all_possible_bins = sorted(df_filtered[marker_col].unique())
marker_definitions = [
    {'marker': 'o', 'fillstyle': 'none',  's_mult': 1.0},  # Hollow Circle
    {'marker': 's', 'fillstyle': 'none',  's_mult': 1.0},
    {'marker': 'o', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Circle
    {'marker': 's', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Square
    {'marker': '^', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Triangle
    {'marker': 'D', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Diamond # Hollow Square
    {'marker': 'X', 'fillstyle': 'full',  's_mult': 1.1},  # Filled X (larger)
    {'marker': '*', 'fillstyle': 'full',  's_mult': 1.8},  # Star (even larger)
    {'marker': 'P', 'fillstyle': 'full',  's_mult': 1.2}   # Plus (larger)
]
marker_map = {bin_val: marker_definitions[i % len(marker_definitions)] for i, bin_val in enumerate(all_possible_bins)}


# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(10, 8))
BASE_MARKERSIZE = 7

# Loop through each sigma (for color) and then each bin_width (for marker)
for sigma_val in all_possible_sigmas:
    if sigma_val != 5.0:
        for bin_val in all_possible_bins:
            # Select the subset of data for this specific combination
            subset = df_filtered[(df_filtered[group_col] == sigma_val) & (df_filtered[marker_col] == bin_val)]
            if subset.empty:
                continue

            y_error = [subset['mvt_err_lower'], subset['mvt_err_upper']]
            marker_props = marker_map.get(bin_val, marker_definitions[0])
            ax.errorbar(
                x=subset[x_col], y=subset[y_col], yerr=y_error,
                # Apply custom marker properties individually
                marker=marker_props.get('marker'),
                markersize=BASE_MARKERSIZE * marker_props.get('s_mult', 1.0),
                fillstyle=marker_props.get('fillstyle'),
                color=color_map.get(sigma_val, 'black'),
                capsize=3, linestyle='none', alpha=0.9,
                markeredgecolor='black',
                markeredgewidth=0.7
            )


# --- 6. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')

#ax.set_ylabel(y_col.replace('_', ' ').title())
ax.grid(True, which='both', linestyle='--', linewidth=0.5)


x_label = x_col.replace('_', ' ').title()
y_label = y_col.replace('_', ' ').title()
x_label = x_label.replace('Mvt Snr', r'SNR$_{{\mathrm{MVT}}}$')
y_label = y_label.replace('Median Mvt Ms', 'Median MVT (ms)')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

# --- 7. Create Custom Dual Legends ---
# Legend for Sigma (Color)
color_legend_elements = [Line2D([0], [0], color=color_map.get(s, 'black'), lw=5, label=f'{s} ms') for s in all_possible_sigmas if s != 5.0]
legend1 = ax.legend(handles=color_legend_elements,  loc='lower left', ncol=2, handlelength=0.5)

# Legend for Bin Width (Marker)
marker_legend_elements = [Line2D([0], [0], marker=marker_map.get(b, {'marker': 'o', 'fillstyle': 'full', 's_mult': 1.0})['marker'], color='gray', label=f'{b}',
                                 linestyle='None', markersize=8) for b in all_possible_bins]
ax.legend(handles=marker_legend_elements, title='Bin Width (ms)', loc='upper right', ncols=3)

# Add the first legend back to the plot
ax.add_artist(legend1)

plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to make space for legends

# --- 8. Save Output ---
plt.savefig(output_png, dpi=300)
#plt.savefig(output_pdf)

print(f"Plot saved to '{output_png}'")