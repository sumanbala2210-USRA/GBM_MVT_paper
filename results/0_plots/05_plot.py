import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- 1. Configuration ---
csv_filename = '05_plotted_data_bin_width_ms_vs_median_mvt_ms.csv'

x_col = 'bin_width_ms'
y_col = 'median_mvt_ms'
# MODIFIED: Swapped roles
color_col = 'peak_time_ratio'
marker_col = 'width'

output_base_name = f"{csv_filename[0:2]}_MVT_vs_{x_col}_by_Width_and_Ratio"
output_png = f"{output_base_name}.png"
output_pdf = f"{output_base_name}.pdf"

# MODIFIED: Your specified color map for 'width'
color_map = {
    1.0:    'tab:blue',
    3.0:    'tab:orange',
    10.0:   'black',
    30.0:   'tab:red',
    100.0:  'tab:purple',
    300.0:  'tab:brown',
    1000.0: 'tab:pink',
}

# --- 2. Publication Style Configuration ---


plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'legend.title_fontsize': 20,
    'font.family': 'serif'
})

# --- 3. Data Preparation ---
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

# MODIFIED: Updated filtering logic based on your new design
ratios_to_plot = [0.2, 0.5, 0.8]
df_filtered = df[
    (df['median_mvt_ms'] > 0) & 
    (df['width'] > 1.0) & 
    (df['width'] != 50.0) &
    (df['peak_time_ratio'].isin(ratios_to_plot))
].copy()


# --- 4. Plot Styling ---
# MODIFIED: Style map now controls color AND fill style for Peak Time Ratio
style_map = {
    0.2: {'color': 'tab:blue', 'fillstyle': 'none'},
    0.5: {'color': 'black',    'fillstyle': 'full'},
    0.8: {'color': 'tab:orange', 'fillstyle': 'none'}
}
all_possible_ratios = sorted(df_filtered[color_col].unique())

# MODIFIED: Marker map now controls the shape for Width
all_possible_widths = sorted(df_filtered[marker_col].unique())
marker_styles = ['o', 's', '^', 'D', 'v', 'P']
marker_map = {width_val: marker_styles[i % len(marker_styles)] for i, width_val in enumerate(all_possible_widths)}
BASE_MARKERSIZE = 9

# MODIFIED: Symmetrical offset map for dodging 0.2 and 0.8 ratios
offset_map = {0.2: 0.97, 0.5: 1.0, 0.8: 1.03}


# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(10, 8))

for width_val in all_possible_widths:
    for ratio_val in all_possible_ratios:
        subset = df_filtered[(df_filtered[marker_col] == width_val) & (df_filtered[color_col] == ratio_val)]
        if subset.empty:
            continue
            
        y_error = [subset['mvt_err_lower'], subset['mvt_err_upper']]
        x_shifted = subset[x_col] * offset_map.get(ratio_val, 1.0)
        
        style_props = style_map.get(ratio_val)
        color_props = color_map.get(width_val, 'gray')
        
        ax.errorbar(
            x=x_shifted, y=subset[y_col], yerr=y_error,
            marker=marker_map.get(width_val, 'o'), # Shape from width
            color=color_props,        # Color from width
            fillstyle=style_props.get('fillstyle'),# Fill from ratio
            markersize=BASE_MARKERSIZE,
            capsize=3, linestyle='none', alpha=0.9,
            markeredgecolor=color_props,
            markeredgewidth=1.5
        )

# --- 6. Smart Legend Creation ---
# Legend for 'width' (marker shape)
width_legend_elements = [Line2D([0], [0], marker=marker_map[w], color=color_map.get(w, 'gray'), label=f'{w}',
                                linestyle='None', markersize=10, markeredgecolor='black') for w in all_possible_widths]

# Legend for 'peak_time_ratio' (color and fill style)
ratio_legend_elements = []
for r in all_possible_ratios:
    props = style_map.get(r)
    ratio_legend_elements.append(
        Line2D([0], [0], marker='s', label=f'{r}', linestyle='None', markersize=10,
               markerfacecolor='gray' if props.get('fillstyle') == 'full' else 'none',
               markeredgecolor='gray', markeredgewidth=1.5)
    )

legend1 = ax.legend(handles=width_legend_elements, title='Width (ms)', loc='upper left')
ax.add_artist(legend1)
legend2 = ax.legend(handles=ratio_legend_elements, title='Peak Time Ratio', loc='lower right')


# --- 7. Layout and Theming ---
ax.set_xscale('log')
ax.set_yscale('log')
x_label = x_col.replace('_', ' ').title()
y_label = y_col.replace('_', ' ').title()
x_label = x_label.replace('Bin Width Ms', 'Bin Width (ms)')
y_label = y_label.replace('Median Mvt Ms', 'Median MVT (ms)')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
#ax.set_xlabel(x_col.replace('_', ' ').title())
#ax.set_ylabel(y_col.replace('_', ' ').title())
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()


# --- 8. Save Output ---
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"Plot saved to '{output_png}' and '{output_pdf}'")