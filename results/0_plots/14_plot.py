import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors

# --- 1. Master Configuration ---
csv_filename = '07_Norris_summary_results.csv'
x_col = 'MVT_snr'
y_col = 'median_mvt_ms'
marker_col = 'bin_width_ms'
rise_time_col = 'rise_time'
decay_time_col = 'decay_time'
filter_col = 'peak_amplitude'
success_rate_col = 'success_rate'

output_base_name = f"14_Combined_MVT_Analysis"
output_png = f"{output_base_name}.png"
output_pdf = f"{output_base_name}.pdf"


# --- 2. Publication Style Configuration ---
plt.rcParams.update({
    'font.size': 24, 'axes.labelsize': 24, 'xtick.labelsize': 20,
    'ytick.labelsize': 22, 'legend.fontsize': 18, 'legend.title_fontsize': 20,
    'font.family': 'serif'
})


# --- 3. Data Preparation and Splitting ---
try:
    df = pd.read_csv(csv_filename)
    df['decay_time'] = df['decay_time'] * 1000  # Use decay_time1 for plotting
    df['rise_time'] = df['rise_time'] * 1000  # Use
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found.")
    exit()

df_filtered = df[df[y_col] > 0].copy()

# Split the data into two parts based on the filter column
df_top = df_filtered[df_filtered[filter_col] < 5000].copy()
df_bottom = df_filtered[df_filtered[filter_col] < 5000].copy()

# Calculate success_rate for the bottom part
if 'successful_runs' in df_bottom.columns and 'total_sim' in df_bottom.columns:
    df_bottom[success_rate_col] = (df_bottom['successful_runs'] / df_bottom['total_sim']) * 100
else:
    print("Warning: 'successful_runs'/'total_sim' columns not found for bottom data. Using random data for success_rate.")
    df_bottom[success_rate_col] = np.random.uniform(50, 100, size=len(df_bottom))


# --- 4. Styling Definitions ---
# --- Top Plot Styles (Categorical Color) ---
color_map = {
    1.0: 'tab:blue', 3.0: 'tab:orange', 10.0: 'black',
    30.0: 'tab:red', 100.0: 'tab:purple', 300.0: 'tab:brown',
    1000.0: 'tab:pink',
}

# --- Bottom Plot Styles (Continuous Color) ---
colormap = cm.cividis
norm = colors.Normalize(vmin=df_bottom[success_rate_col].min(), vmax=df_bottom[success_rate_col].max())

# --- Shared Marker Styles ---
all_possible_bins = sorted(df_filtered[marker_col].unique())
marker_definitions = [
    {'marker': 'o', 'fillstyle': 'none', 's_mult': 1.0}, {'marker': 's', 'fillstyle': 'none', 's_mult': 1.0},
    {'marker': 'o', 'fillstyle': 'full', 's_mult': 1.0}, {'marker': 's', 'fillstyle': 'full', 's_mult': 1.0},
    {'marker': '^', 'fillstyle': 'full', 's_mult': 1.0}, {'marker': 'D', 'fillstyle': 'full', 's_mult': 1.0},
    {'marker': 'X', 'fillstyle': 'full', 's_mult': 1.1}, {'marker': '*', 'fillstyle': 'full', 's_mult': 1.4},
    {'marker': 'P', 'fillstyle': 'full', 's_mult': 1.2}
]
marker_map = {bin_val: marker_definitions[i % len(marker_definitions)] for i, bin_val in enumerate(all_possible_bins)}

# --- 5. Create the Figure and Axes ---
# 4 rows, 2 columns, sharing both axes
fig, axes = plt.subplots(4, 2, figsize=(20, 26), sharex=True, sharey=True)
axes = axes.flatten()
BASE_MARKERSIZE = 10

# --- 6. Plotting the TOP FOUR panels (by Decay Time) ---
rise_times_top = sorted(df_top[rise_time_col].unique())
for i, rise_time in enumerate(rise_times_top):
    ax = axes[i]
    panel_df = df_top[df_top[rise_time_col] == rise_time]
    for decay_val, d_df in panel_df.groupby(decay_time_col):
        for bin_val, b_df in d_df.groupby(marker_col):
            if b_df.empty: continue
            marker_props = marker_map.get(bin_val, marker_definitions[0])
            ax.errorbar(
                x=b_df[x_col], y=b_df[y_col], yerr=[b_df['mvt_err_lower'], b_df['mvt_err_upper']],
                marker=marker_props.get('marker'), markersize=BASE_MARKERSIZE * marker_props.get('s_mult', 1.0),
                fillstyle=marker_props.get('fillstyle'), color=color_map.get(decay_val, 'black'),
                capsize=3, linestyle='none', alpha=0.9, markeredgecolor='black', markeredgewidth=0.7
            )
    ax.text(0.02, 0.06, f'Rise Time {rise_time} ms', transform=ax.transAxes, fontsize=24,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- 7. Plotting the BOTTOM FOUR panels (by Success Rate) ---
rise_times_bottom = sorted(df_bottom[rise_time_col].unique())
for i, rise_time in enumerate(rise_times_bottom):
    ax = axes[i + 4] # Offset by 4 to target the bottom panels
    panel_df = df_bottom[df_bottom[rise_time_col] == rise_time]
    for decay_val, d_df in panel_df.groupby(decay_time_col):
        for bin_val, b_df in d_df.groupby(marker_col):
            if b_df.empty: continue
            marker_props = marker_map.get(bin_val, marker_definitions[0])
            point_color = colormap(norm(b_df[success_rate_col].iloc[0]))
            ax.errorbar(
                x=b_df[x_col], y=b_df[y_col], yerr=[b_df['mvt_err_lower'], b_df['mvt_err_upper']],
                marker=marker_props.get('marker'), markersize=BASE_MARKERSIZE * marker_props.get('s_mult', 1.0),
                fillstyle=marker_props.get('fillstyle'), color=point_color,
                capsize=3, linestyle='none', alpha=1, markeredgecolor='black', markeredgewidth=0.2
            )
    ax.text(0.02, 0.06, f'Rise Time {rise_time} ms', transform=ax.transAxes, fontsize=22,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- 8. Final Touches: Legends, Labels, and Layout ---
# --- Legends for the Top Plots ---
marker_legend_elements = [Line2D([0], [0], marker=marker_map.get(b)['marker'], fillstyle=marker_map.get(b)['fillstyle'],
                                 markersize=12, color='gray', label=f'{b}', linestyle='None', markeredgecolor='black')
                          for b in all_possible_bins]
color_legend_elements = [Line2D([0], [0], color=color_map.get(d, 'black'), lw=4, label=f'{d}')
                         for d in sorted(df_top[decay_time_col].unique())]

axes[0].legend(handles=marker_legend_elements, title='Bin Width (ms)', loc='upper right', ncol=3)
axes[1].legend(handles=color_legend_elements, title='Decay Time (ms)', loc='upper right')

# --- Legend for the Bottom Plots ---
axes[4].legend(handles=marker_legend_elements, title='Bin Width (ms)', loc='upper right', ncol=3)

# --- Add the Colorbar for the Bottom Plots ---
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.38]) # Position: [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Success Rate (%)', rotation=270, labelpad=25)

# --- Set common properties for all axes ---
for i, ax in enumerate(axes):
    if i in [3, 7]:  # Only for the rightmost plots
        ax.set_ylim(0.7, 100)
    else:
        ax.set_ylim(0.7, 100)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5, 20000)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Add Shared Axis Labels ---
fig.supxlabel(r'SNR$_{{\mathrm{MVT}}}$', y=0.02)
fig.supylabel('Median MVT (ms)', x=0.01)

# --- Final Layout Adjustment ---
plt.tight_layout(rect=[0.0, 0.0, 0.92, 1])

# --- 9. Save Output ---
plt.savefig(output_png, dpi=300)
plt.savefig(output_pdf)
print(f"Combined plot saved to '{output_png}' and '{output_pdf}'")