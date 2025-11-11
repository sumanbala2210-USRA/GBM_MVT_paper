import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- 1. Configuration ---
# Update the filename and the column for the x-axis
csv_filename = '02_plotted_data_peak_amplitude_vs_median_mvt_ms.csv'
x_col = 'peak_amplitude'
y_col = 'median_mvt_ms'
group_col = 'sigma'

# Generate the output filename automatically
output_base_name = f"MVT_vs_{x_col}_by_{group_col}"
output_png = f"{csv_filename[0:2]}_{output_base_name}.png"
output_pdf = f"{csv_filename[0:2]}_{output_base_name}.pdf"


# --- 2. Publication Style Configuration ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'font.family': 'serif'
})

regime_bounds = {
    1.0:   2e2,
    3.0:   1e2,
    10.0:  20,
    30.0:  4,
    100: 2,
    300: 1,
    1000: 0.5,
}
# --- 3. Data Preparation ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

# Filter data: group_col >= 1 and valid MVT
df_filtered = df[(df[y_col] > 0) & (df[group_col] >= 1.0)].copy()


# --- 4. Plot Styling (with Consistent Color & Marker Maps) ---
# Get the unique sigma values that are present in the filtered data
unique_sigmas = sorted(df_filtered[group_col].unique())

# Your specified color map for consistency with other plots
color_map = {
    1.0:    'tab:blue',
    3.0:    'tab:orange',
    10.0:   'black',
    30.0:   'tab:red',
    100.0:  'tab:purple',
    300.0:  'tab:brown',
    1000.0: 'tab:pink',
}

# The list of distinct marker properties from your final version
marker_definitions = [
    {'marker': 'v', 'fillstyle': 'full',  's_mult': 1.2},  # Filled Triangle
    {'marker': 's', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Square
    {'marker': 'o', 'fillstyle': 'full',  's_mult': 1.1},  # Filled Circle
    {'marker': '^', 'fillstyle': 'full',  's_mult': 1.2},  # Filled Triangle
    {'marker': 'D', 'fillstyle': 'full',  's_mult': 1.0},  # Filled Diamond
    {'marker': 'X', 'fillstyle': 'full',  's_mult': 1.2},  # Filled X (larger)
    {'marker': '*', 'fillstyle': 'full',  's_mult': 1.8},  # Star (even larger)
]
# Map each sigma value to a unique marker style
marker_map = {sigma_val: marker_definitions[i % len(marker_definitions)] for i, sigma_val in enumerate(unique_sigmas)}
BASE_MARKERSIZE = 12


# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each sigma group separately
for group_value in unique_sigmas:
    group_df = df_filtered[df_filtered[group_col] == group_value]
    
    y_error = [group_df['mvt_err_lower'], group_df['mvt_err_upper']]
    marker_props = marker_map.get(group_value, marker_definitions[0])
    
    ax.errorbar(
        x=group_df[x_col],
        y=group_df[y_col],
        yerr=y_error,
        # Apply custom marker properties from the map
        marker=marker_props.get('marker'),
        markersize=BASE_MARKERSIZE * marker_props.get('s_mult', 1.0),
        fillstyle=marker_props.get('fillstyle'),
        # Apply custom color from the map
        color=color_map.get(group_value, 'gray'),
        # Other style properties
        capsize=3, linestyle='none', alpha=0.9,
        markeredgecolor='black', markeredgewidth=0.7,
        label=f'$\\sigma={int(group_value)}$'
    )



x_green_list = []
y_green_list = []
# ---- green curve by hand (manually specified anchor points) ----
# EDIT THESE TWO LISTS by hand
for s, xk in regime_bounds.items():
    g = df_filtered[df_filtered[group_col] == s].sort_values(x_col)
    # interpolate MVT at xk
    yk = np.interp(xk, g[x_col], g[y_col])
    x_green_list.append(xk)
    y_green_list.append(yk)

x_green = np.array(x_green_list)
y_green = np.array(y_green_list)

#print("x_green:", x_green)
#print("y_green:", y_green)
# sort in case user entered unordered
order = np.argsort(x_green)
x_green = x_green[order]
y_green = y_green[order]


lx = np.log10(x_green)
ly = np.log10(y_green)

# 2nd order polynomial fit: ly = a*lx^2 + b*lx + c
coeff = np.polyfit(lx, ly, deg=2)   # [a,b,c]
poly = np.poly1d(coeff)

# build dense curve
lx_fit = np.linspace(lx.min(), lx.max(), 300)



xmin_plot = df_filtered[x_col].min()        # e.g. ~0.1
xmax_poly  = 10**lx.max()


ly_fit = poly(lx_fit)

x_fit = 10**lx_fit
y_fit = 10**ly_fit


lx_min_extended = np.log10(xmin_plot)

lx_fit = np.linspace(lx_min_extended, lx.max(), 500)
ly_fit = poly(lx_fit)
x_fit  = 10**lx_fit
y_fit  = 10**ly_fit




# fill under line
#ax.fill_between(x_green, y_green, y2=ax.get_ylim()[0],
#                color='red', alpha=0.10, zorder=0)

ax.fill_between(x_fit, y_fit, y2=ax.get_ylim()[0],
                color='red', alpha=0.10, zorder=0)

# --- 6. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')
x_label = x_col.replace('_', ' ').title()
y_label = y_col.replace('_', ' ').title()
#x_label = x_label.replace()
y_label = y_label.replace('Median Mvt Ms', 'Median MVT (ms)')
ax.set_xlabel(x_col.replace('_', ' ').title())
ax.set_ylabel(y_label)

# A simple legend is now sufficient as labels are set in the plot loop
ax.legend(title='$\\sigma$ (ms)', ncol=2, loc='upper right', bbox_to_anchor=(1, 1))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# --- 7. Save Output ---
plt.savefig(output_png, dpi=300, bbox_inches='tight')
#plt.savefig(output_pdf, bbox_inches='tight')

print(f"Plot saved to '{output_png}' and '{output_pdf}'")