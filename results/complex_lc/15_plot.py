import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- 1. Configuration ---
#csv_filename = '09_complex_combined_bw_selected.csv'

# --- 1. Configuration ---
#csv_original_filename = '09_complex_combined_bw_selected.csv'
#csv_new_filename = '09_complex_combined_2P_rename.csv' # New data file

# --- 1. Configuration ---
csv_filename = '10_complex_S_3ms.csv'
csv_3 = '10_C_S_3_01_real.csv'
#csv_filename_new = '09_complex_combined_2P_rename.csv' # Your new data
#csv_filename_new =  '09_complex_combined_2P_rename_new.csv'

output_png = '15_figure_master_grid_landscape_app.png'
output_pdf = '15_figure_master_grid_landscape_app.pdf'

# --- 2. Publication Style Configuration ---
plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 22, 'xtick.labelsize': 16,
    'ytick.labelsize': 18, 'legend.fontsize': 16, 'legend.title_fontsize': 18,
    'font.family': 'serif', 'axes.titlesize': 18
})

# --- 3. Data Preparation ---
try:
    df1 = pd.read_csv(csv_filename)
    #df2 = pd.read_csv(csv_2)
    df3 = pd.read_csv(csv_3)
    df = pd.concat([df1, df3], ignore_index=True)
    print(f"Data loaded successfully from '{csv_filename}', and '{csv_3}'.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure all CSV files are present.")
    exit()

    
df_filtered = df[df['median_mvt_ms'] > 0].copy()
df_filtered['sigma_feature'] = df_filtered['sigma_feature'] * 1000  # Convert to ms if needed

# --- 4. Define Parameters for Plotting ---
peak_amp_ratios = sorted(df_filtered['peak_amp_relative'].unique())
positions_to_compare = [4.5, 6.2]
sigmas_to_compare = [3.0, 10.0]
bins_to_compare = [0.01, 0.1]
shapes_to_compare = ['complex_pulse_long', 'complex_pulse_short']
#comparison_colors = ['black', 'skyblue']
comparison_colors = ['black', 'tab:orange']
marker_styles = ['o', 's']

# --- 5. Create the Figure Grid ---
fig, axes = plt.subplots(4, len(peak_amp_ratios), figsize=(28, 19), 
                         sharex=True, sharey=True,
                         gridspec_kw={'wspace': 0, 'hspace': 0})

# --- Loop through each COLUMN (peak_amp_relative) ---
for col_idx, peak_val in enumerate(peak_amp_ratios):
    if peak_val == 0.01:
        #axes[0, col_idx].set_title(f"Peak Amp Rel \\sim 0", pad=15, fontsize=20)
        axes[0, col_idx].set_title(r"RPA $\mathrm{\approx}$ 0", pad=15, fontsize=20)

    else:
        axes[0, col_idx].set_title(f"RPA = {peak_val}", pad=15, fontsize=20)

    # --- ROW 1: Varying Position (Color) vs. Sigma (Marker) ---
    """
    ax = axes[0, col_idx]
    df_row = df_filtered[(df_filtered['pulse_shape'] == 'complex_pulse_short') & (df_filtered['bin_width_ms'] == 0.1) & (df_filtered['peak_amp_relative'] == peak_val)]
    for i, pos in enumerate(positions_to_compare):
        for j, sig in enumerate(sigmas_to_compare):
            subset = df_row[(df_row['position'] == pos) & (df_row['sigma_feature'] == sig)]
            ax.errorbar(subset['MVT_snr'], subset['median_mvt_ms'], yerr=[subset['mvt_err_lower'], subset['mvt_err_upper']],
                        fmt=marker_styles[j], color=comparison_colors[i], capsize=4, alpha=0.8, markeredgecolor='black', 
                        markersize=10, markeredgewidth=0.8)
            ax.text(0.05, 0.95, f'A{col_idx + 1}', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', lw=0, alpha=0.3))
    """
    
    # --- ROW 2: Varying Sigma Feature (Color) vs. Position (Marker) ---
    ax = axes[0, col_idx]
    df_row = df_filtered[(df_filtered['pulse_shape'] == 'complex_pulse_short') & (df_filtered['bin_width_ms'] == 0.1) & (df_filtered['peak_amp_relative'] == peak_val)]
    for i, sig in enumerate(sigmas_to_compare):
        for j, pos in enumerate(positions_to_compare):
            subset = df_row[(df_row['sigma_feature'] == sig) & (df_row['position'] == pos)]
            ax.errorbar(subset['MVT_snr'], subset['median_mvt_ms'], yerr=[subset['mvt_err_lower_ms'], subset['mvt_err_upper_ms']],
                        fmt=marker_styles[j], color=comparison_colors[i], capsize=4, alpha=0.8, markeredgecolor='black',
                        markersize=10, markeredgewidth=0.8)
            ax.text(0.05, 0.95, f'A{col_idx + 1}', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', lw=0, alpha=0.7))

    # --- ROW 3: Varying Bin Width (Color) vs. Position (Marker) ---
    ax = axes[3, col_idx]
    df_row = df_filtered[(df_filtered['pulse_shape'] == 'complex_pulse_short') & (df_filtered['sigma_feature'] == 3.0) & (df_filtered['peak_amp_relative'] == peak_val)]
    for i, b_width in enumerate(bins_to_compare):
        for j, pos in enumerate(positions_to_compare):
            subset = df_row[(df_row['bin_width_ms'] == b_width) & (df_row['position'] == pos)]
            ax.errorbar(subset['MVT_snr'], subset['median_mvt_ms'], yerr=[subset['mvt_err_lower_ms'], subset['mvt_err_upper_ms']],
                        fmt=marker_styles[j], color=comparison_colors[i], capsize=4, alpha=0.8, markeredgecolor='black',
                        markersize=10, markeredgewidth=0.8)
            ax.text(0.05, 0.95, f'D{col_idx + 1}', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', lw=0, alpha=0.7))
    
    # --- ROW 4: Varying Pulse Shape (Color) vs. Position (Marker) ---
    ax = axes[1, col_idx]
    df_row = df_filtered[(df_filtered['sigma_feature'] == 10.0) & (df_filtered['bin_width_ms'] == 0.1) & (df_filtered['peak_amp_relative'] == peak_val)]
    for i, shape in enumerate(shapes_to_compare):
        for j, pos in enumerate(positions_to_compare):
            subset = df_row[(df_row['pulse_shape'] == shape) & (df_row['position'] == pos)]
            ax.errorbar(subset['MVT_snr'], subset['median_mvt_ms'], yerr=[subset['mvt_err_lower_ms'], subset['mvt_err_upper_ms']],
                        fmt=marker_styles[j], color=comparison_colors[i], capsize=4, alpha=0.8, markeredgecolor='black',
                        markersize=10, markeredgewidth=0.8)
            ax.text(0.05, 0.95, f'B{col_idx + 1}', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', lw=0, alpha=0.7))

    # --- MODIFIED ROW 5: Compare 10ms vs 5ms Pulses ---
    ax = axes[2, col_idx]
    new_shapes_to_compare = ['complex_pulse_short_2p10ms', 'complex_pulse_short_2p3ms']
    df_row = df_filtered[(df_filtered['pulse_shape'].isin(new_shapes_to_compare)) & 
                         (df_filtered['bin_width_ms'] == 0.1) & 
                         (df_filtered['peak_amp_relative'] == peak_val)]
    
    # Swapped logic: Color = Pulse Shape, Marker = Position
    for i, shape in enumerate(new_shapes_to_compare):
        for j, pos in enumerate(positions_to_compare):
            subset = df_row[(df_row['pulse_shape'] == shape) & (df_row['position'] == pos)]
            ax.errorbar(subset['MVT_snr'], subset['median_mvt_ms'], yerr=[subset['mvt_err_lower_ms'], subset['mvt_err_upper_ms']],
                        fmt=marker_styles[j], color=comparison_colors[i], capsize=4, alpha=0.8, markeredgecolor='black',
                        markersize=10, markeredgewidth=0.8)
            ax.text(0.05, 0.95, f'C{col_idx + 1}', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', lw=0, alpha=0.7))

# --- 6. Final Layout, Legends, and Theming ---
all_row_info = [
    #{'fixed': "shape=short, bin=0.1", 'color_title': 'Position', 'marker_title': 'Sigma', 'color_labs': positions_to_compare, 'marker_labs': sigmas_to_compare},
    {'fixed': "shape=short, bin=0.1ms", 'color_title': 'Sigma (ms)', 'marker_title': 'Position (s)', 'color_labs': sigmas_to_compare, 'marker_labs': positions_to_compare},
    {'fixed': "sigma=10, bin=0.1ms", 'color_title': 'Pulse Shape', 'marker_title': 'Position (s)', 'color_labs': ['Complex-Long', 'Complex-Short'], 'marker_labs': positions_to_compare},
    # MODIFIED LEGEND INFO FOR ROW 5
    {'fixed pulse': "RPA = 2.0", 'color_title': 'Two Pulses', 'marker_title': 'Position (s)', 'color_labs': ['Vary: 3ms, Fix: 10ms', 'Vary: 10ms, Fix: 3ms'], 'marker_labs': positions_to_compare},
    {'fixed': "shape=short, sigma=3ms", 'color_title': 'Bin Width (ms)', 'marker_title': 'Position (s)', 'color_labs': bins_to_compare, 'marker_labs': positions_to_compare},
]











for row_idx in range(4): # Loop through all 5 rows
    try:
        axes[row_idx, 0].text(0.05, 0.05, f"Fixed:\n{all_row_info[row_idx]['fixed']}", transform=axes[row_idx, 0].transAxes, fontsize=18, va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    except:
        axes[row_idx, 0].text(0.05, 0.05, f"Fixed Pulse:\n{all_row_info[row_idx]['fixed pulse']}", transform=axes[row_idx, 0].transAxes, fontsize=18, va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    if len(peak_amp_ratios) > 0:
        info = all_row_info[row_idx]
        # Using your preferred legend style (split across last two columns)
        leg_handles1 = [Line2D([0], [0], color=c, lw=5, label=p) for p, c in zip(info['color_labs'], comparison_colors)]
        leg_handles2 = [Line2D([0], [0], marker=m, color='gray', linestyle='None', label=s, markersize=12, markeredgecolor='black', markeredgewidth=0.8) for s, m in zip(info['marker_labs'], marker_styles)]
        
        leg1 = axes[row_idx, -1].legend(handles=leg_handles1, title=info['color_title'], loc='upper right')
        axes[row_idx, -1].add_artist(leg1)
        axes[row_idx, -2].legend(handles=leg_handles2, title=info['marker_title'], loc='upper right')

# Set clean row labels on the first column
#axes[0, 0].set_ylabel('Position')
axes[0, 0].set_ylabel('Sigma')
axes[1, 0].set_ylabel('Pulse Shape')
axes[2, 0].set_ylabel('Two Pulses')
axes[3, 0].set_ylabel('Bin Width')

for ax in axes.flatten():
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(2, 300)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

fig.supxlabel(r'SNR$_{{\mathrm{MVT}}}$', fontsize=24, y=0.02)
fig.supylabel('Median MVT (ms)', fontsize=24, x=0.015)
plt.tight_layout(rect=[0.01, 0.01, 0.96, 0.96])

# --- 7. Save Output ---
plt.savefig(output_png, dpi=300)
plt.savefig(output_pdf)
print(f"Master figure saved to '{output_png}' and '{output_pdf}'")