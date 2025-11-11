import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D

# --- 1. Configuration ---
csv_filename = '02_GAUSS_ALL.csv'

x_col = 'peak_amplitude'
y_col = 'median_mvt_ms'
color_col = 'success_rate'
marker_col = 'sigma'

output_base_name = f"MVT_vs_{x_col}_by_sigma_and_success_rate"
output_png = f"03_{output_base_name}.png"
output_pdf = f"{csv_filename.split('_')[0]}_{output_base_name}.pdf"


# --- 2. Publication Style Configuration ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'legend.title_fontsize': 12,
    'font.family': 'serif'
})
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'font.family': 'serif'
})
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
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

# Filter for valid data and sigma >= 1.0
df_filtered = df[(df['median_mvt_ms'] > 0) & (df[marker_col] >= 1.0)].copy()
df_filtered['success_rate'] = (df_filtered['successful_runs'] / df_filtered['total_sim']) * 100


# --- 4. Plot Styling ---
colormap = cm.cividis
norm = colors.Normalize(vmin=0, vmax=100)

unique_sigmas = sorted(df_filtered[marker_col].unique())
# CORRECTED: Removed duplicate marker to ensure all styles are unique
marker_definitions = [
    {'marker': 'v', 'fillstyle': 'full',  's_mult': 1.2},  # Filled Triangle
    {'marker': 's', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Square
    {'marker': 'o', 'fillstyle': 'full',  's_mult': 1.1},  # Filled Circle
    {'marker': '^', 'fillstyle': 'full',  's_mult': 1.2},  # Filled Triangle
    {'marker': 'D', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Diamond
    {'marker': 'X', 'fillstyle': 'full',  's_mult': 1.2},  # Filled X (larger)
    {'marker': '*', 'fillstyle': 'full',  's_mult': 1.8},  # Star (even larger)
]
# M
marker_map = {sigma_val: marker_definitions[i % len(marker_definitions)] for i, sigma_val in enumerate(unique_sigmas)}
BASE_MARKERSIZE = 10


# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(12, 9))
scatter_handle = None # Initialize handle for the colorbar

for sigma_val in unique_sigmas:
    subset = df_filtered[df_filtered[marker_col] == sigma_val]
    if subset.empty:
        continue

    marker_props = marker_map.get(sigma_val, marker_definitions[0])

    ax.errorbar(
        x=subset[x_col], y=subset[y_col],
        yerr=[subset['mvt_err_lower'], subset['mvt_err_upper']],
        fmt='none', capsize=0, ecolor='gray', alpha=0.6, zorder=1
    )
    
    point_colors = colormap(norm(subset[color_col]))
    facecolors = point_colors
    edgecolors = 'black'
    
    if marker_props.get('fillstyle') == 'none':
        facecolors = 'none'
        edgecolors = point_colors

    scatter_handle = ax.scatter(
        x=subset[x_col], y=subset[y_col],
        s=(BASE_MARKERSIZE * marker_props.get('s_mult', 1.0))**2,
        marker=marker_props.get('marker'),
        facecolors=facecolors,
        edgecolors=edgecolors,
        alpha=0.9,
        linewidths=1.0,
        zorder=2
    )


# --- 6. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(x_col.replace('_', ' ').title())
ax.set_ylabel(y_col.replace('_', ' ').title().replace('Median Mvt Ms', 'Median MVT (ms)'))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)


# --- 7. Create Legend and Colorbar ---
marker_legend_elements = []
for s in unique_sigmas:
    props = marker_map.get(s)
    face_color = 'gray' if props.get('fillstyle') == 'full' else 'none'
    marker_legend_elements.append(Line2D([0], [0],
                                         marker=props.get('marker'),
                                         markerfacecolor=face_color,
                                         markeredgecolor='black',
                                         markersize=10 * props.get('s_mult', 1.0),
                                         label=f'{s}',
                                         linestyle='None',
                                         markeredgewidth=1.0))
# MODIFIED: Moved legend to upper left to avoid colorbar
ax.legend(handles=marker_legend_elements, title='$\\sigma$ (ms)', loc='upper right', ncol=3)

# MODIFIED: Added 'pad' argument to reduce gap and created a mappable object
if scatter_handle:
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Success Rate (%)', pad=0.01)

plt.tight_layout()


# --- 8. Save Output ---
plt.savefig(output_png, dpi=300, bbox_inches='tight')
#plt.savefig(output_pdf, bbox_inches='tight')

print(f"Plot saved to '{output_png}'")