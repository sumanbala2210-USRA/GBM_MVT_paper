import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- 1. Configuration ---
# Define columns to plot
csv_filename = '01_plotted_data_bin_width_ms_vs_median_mvt_ms.csv'

x_col = 'bin_width_ms'
y_col = 'median_mvt_ms'
group_col = 'sigma'

# Generate the output filename automatically
output_base_name = f"{csv_filename.split('_')[0]}_MVT_vs_{x_col}_by_{group_col}"
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
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

# Filter data: group_col >= 1 and valid MVT
df_filtered = df[(df[y_col] > 0) & (df[group_col] >= 1.0)].copy()


# --- 4. Plot Styling ---
# Get the unique sigma values that are present in the filtered data
unique_sigmas = sorted(df_filtered[group_col].unique())

# Using your specified color map for consistency with other plots
color_map = {
    1.0:    'tab:blue',
    3.0:    'tab:orange',
    10.0:   'black',
    30.0:   'tab:red',
    100.0:  'tab:purple',
    300.0:  'tab:brown',
    1000.0: 'tab:pink',
}

# CORRECTED: Your list of markers with the duplicate removed
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
regime_bounds = {
    1.0:   2e-3,
    3.0:   1e-3,
    10.0:  1e-1,
    30.0:  9e-1,
    100.0: 10,      # plateau always → no hollow
    300.0: 10,
    1000.0:10,
}


regime_bounds = {
    1.0:   1.1e-3,
    3.0:   7e-3,
    10.0:  6e-2,
    30.0:  9e-1,
    100: 5,
    300: 9,
}


# Plot each sigma group separately
for group_value in unique_sigmas:
    group_df = df_filtered[df_filtered[group_col] == group_value]
    
    y_error = [group_df['mvt_err_lower'], group_df['mvt_err_upper']]
    marker_props = marker_map.get(group_value, marker_definitions[0])
    
    ax.errorbar(
        x=group_df[x_col],
        y=group_df[y_col],
        yerr=y_error,
        marker=marker_props.get('marker'),
        markersize=BASE_MARKERSIZE * marker_props.get('s_mult', 1.0),
        fillstyle=marker_props.get('fillstyle'),
        color=color_map.get(group_value, 'gray'), # Default to gray if sigma not in map
        capsize=3, linestyle='none', alpha=0.9,
        markeredgecolor='black', markeredgewidth=0.7,
        label=f'$\\sigma={int(group_value)}$'
    )
    # inside your plotting loop — AFTER errorbar call
    # hollow overlay for unresolved regime
    """
    if group_value in regime_bounds:
        left = group_df[group_df[x_col] > regime_bounds[group_value]]
        ax.scatter(
            left[x_col],
            left[y_col],
            marker=marker_props['marker'],
            s=(BASE_MARKERSIZE * marker_props['s_mult'] * 40),
            facecolors='none',
            edgecolors=color_map[group_value],
            linewidths=1.5,
            alpha=0.9
        )
    """

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
ly_fit = poly(lx_fit)

x_fit = 10**lx_fit
y_fit = 10**ly_fit



# fill under line
#ax.fill_between(x_green, y_green, y2=ax.get_ylim()[0],
#                color='red', alpha=0.10, zorder=0)

ax.fill_between(x_fit, y_fit, y2=ax.get_ylim()[0],
                color='red', alpha=0.10, zorder=0)

# draw the line + markers
#ax.plot(x_green, y_green, color='red', linewidth=0.1, zorder=2)
#ax.scatter(x_green, y_green, s=80, facecolors='white',
#           edgecolors='red', linewidths=2, zorder=3)


# --- 6. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')
x_label = x_col.replace('_', ' ').title()
y_label = y_col.replace('_', ' ').title()
x_label = x_label.replace('Bin Width Ms', 'Bin Width (ms)')
y_label = y_label.replace('Median Mvt Ms', 'Median MVT (ms)')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

ax.legend(title='$\\sigma$ (ms)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# --- 7. Save Output ---
plt.savefig(output_png, dpi=300, bbox_inches='tight')
#plt.savefig(output_pdf, bbox_inches='tight')

print(f"Plot saved to '{output_png}' and '{output_pdf}'")