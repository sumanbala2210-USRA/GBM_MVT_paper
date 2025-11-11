import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors

# --- 1. Configuration ---
csv_filename = '03_BW_AMP_Gauss_ALL.csv'

x_col = 'MVT_snr'
y_col = 'median_mvt_ms'
color_col = 'success_rate'
marker_col = 'bin_width_ms'
size_col = 'sigma'

output_base_name = "13_MVT_vs_SNR_by_All"
output_png = f"{output_base_name}.png"
output_pdf = f"{output_base_name}.pdf"


# --- 2. Publication Style Configuration ---
plt.rcParams.update({
    'font.size': 16, 'axes.labelsize': 18, 'xtick.labelsize': 16,
    'ytick.labelsize': 16, 'legend.fontsize': 12, 'legend.title_fontsize': 14,
    'font.family': 'serif'
})


# --- 3. Data Preparation ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

# Filter for valid data points and calculate success rate
df_filtered = df[(df['median_mvt_ms'] > 0) & (df['MVT_snr'] > 0)].copy()
df_filtered['success_rate'] = df_filtered['successful_runs'] / df_filtered['total_sim'] * 100


# --- 4. Plot Styling ---
all_possible_bins = sorted(df_filtered[marker_col].unique())
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*']
marker_map = {bin_val: marker_styles[i % len(marker_styles)] for i, bin_val in enumerate(all_possible_bins)}

all_possible_sigmas = sorted(df_filtered[size_col].unique())
size_range = np.linspace(40, 200, len(all_possible_sigmas))
size_map = {sigma_val: size_range[i] for i, sigma_val in enumerate(all_possible_sigmas)}

colormap = cm.cividis
norm = colors.Normalize(vmin=df_filtered[color_col].min(), vmax=df_filtered[color_col].max())


# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(12, 9))
scatter_handle = None # To store the scatter plot object for the colorbar

for sigma_val in all_possible_sigmas:
    for bin_val in all_possible_bins:
        subset = df_filtered[(df_filtered[size_col] == sigma_val) & (df_filtered[marker_col] == bin_val)]
        if subset.empty:
            continue

        scatter_handle = ax.scatter(
            x=subset[x_col], 
            y=subset[y_col],
            s=size_map.get(sigma_val, 40),
            marker=marker_map.get(bin_val, 'o'),
            c=subset[color_col],
            cmap=colormap,
            norm=norm,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            zorder=3
        )
        ax.errorbar(
            x=subset[x_col], y=subset[y_col], 
            yerr=[subset['mvt_err_lower'], subset['mvt_err_upper']],
            fmt='none', capsize=0, ecolor='gray', alpha=0.6, zorder=1
        )

# --- 6. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

x_label = x_col.replace('_', ' ').title().replace('Mvt Snr', r'SNR$_{{\mathrm{MVT}}}$')
y_label = y_col.replace('_', ' ').title().replace('Median Mvt Ms', 'Median MVT (ms)')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

# --- 7. Create Legends and Colorbar ---
# --- MODIFIED: Legends moved to the left for a cleaner layout ---
# Create a legend for marker SHAPE (Bin Width)
shape_legend_elements = [Line2D([0], [0], marker=marker_map[b], color='gray', label=f'{b}',
                                linestyle='None', markersize=10, markeredgecolor='black') for b in all_possible_bins]

# --- MODIFIED: Added 'pad' argument to reduce the gap ---
# Add a COLORBAR for Success Rate
if scatter_handle:
    fig.colorbar(scatter_handle, ax=ax, label='Success Rate (%)', pad=0.01)



legend1 = ax.legend(handles=shape_legend_elements, title='Bin Width (ms)', loc='upper right', ncol=3)
ax.add_artist(legend1)

# Create a legend for marker SIZE (Sigma)
size_legend_elements = [plt.scatter([], [], s=size_map[s], c='gray', edgecolor='black', label=f'{s}') for s in all_possible_sigmas]
legend2 = ax.legend(handles=size_legend_elements, title='$\\sigma$ (ms)', loc='upper center', ncol=2)
ax.add_artist(legend2)


plt.tight_layout(rect=[0.01, 0.01, 1.05, 1])


# --- 8. Save Output ---
plt.savefig(output_png, dpi=300)
#plt.savefig(output_pdf)

print(f"Plot saved to '{output_png}' and '{output_pdf}'")