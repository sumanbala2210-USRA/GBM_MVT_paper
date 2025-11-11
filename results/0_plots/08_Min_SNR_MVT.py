import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from adjustText import adjust_text

# --- Global Analysis Parameters ---
#BW_STOP_THRESHOLD = 0.05
#BIAS_GATE_THRESHOLD = 0.05
#RELERR_LIMIT = 0.90
# --- Global Analysis Parameters ---
BW_STOP_THRESHOLD = 0.1
BIAS_GATE_THRESHOLD = 0.1
RELERR_LIMIT = 0.10  # <-- Change this value


def load_data(filepath):
    """
    Loads the simulation data CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"--- Successfully loaded {filepath} (Shape: {df.shape})")
        
        # Define all columns that *might* be used
        all_cols_to_numeric = [
            'bin_width_ms', 'peak_amplitude', 'median_mvt_ms',
            'mvt_err_lower', 'mvt_err_upper', 'MVT_snr', 
            'sigma', 'rise_time', 'decay_time', 
            'width' # <-- ADDED THIS
        ]
        
        # Find which of these columns actually exist in the loaded file
        cols_in_df = [col for col in all_cols_to_numeric if col in df.columns]
        
        # Convert only the existing columns
        for col in cols_in_df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows where any of these critical columns have NaNs
        # We use a subset of cols that MUST be present for analysis
        critical_cols = ['bin_width_ms', 'median_mvt_ms', 'mvt_err_lower', 
                         'mvt_err_upper', 'MVT_snr']
        # Find which critical cols are in this dataframe to check for NaNs
        cols_to_check = [col for col in critical_cols if col in df.columns]
        
        df = df.dropna(subset=cols_to_check, how='any')
        print(f"--- Data cleaned. Remaining shape: {df.shape}")
        return df
    
    except FileNotFoundError:
        print(f"--- ERROR: File not found at {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"--- ERROR: Could not load {filepath}. Reason: {e}", file=sys.stderr)
        return None

def calculate_t_true(group_df):
    """
    Finds the "True MVT" (T_true) for a given parameter group.
    """
    sorted_group = group_df.sort_values(
        by=['bin_width_ms', 'MVT_snr'], 
        ascending=[True, False]
    )
    t_true = sorted_group.iloc[0]['median_mvt_ms']
    return t_true

def calculate_bw_star(group_df, t_true, threshold):
    """
    Finds the Optimal Bin Width (BW*) for a given parameter group.
    
    [V2 LOGIC]
    BW* is the *largest* bin_width that is still "unbiased"
    relative to T_true.
    
    Logic:
    1. For each bin_width, find the MVT at MAX SNR.
    2. Calculate the bias for each of these: |MVT - t_true| / t_true
    3. Filter for all bins where bias <= threshold (e.g., 0.15)
    4. BW* is the *maximum* bin_width from this filtered, "unbiased" set.
    """
    # 1. For each bin_width, get the row with the max SNR
    max_snr_rows = group_df.loc[group_df.groupby('bin_width_ms')['MVT_snr'].idxmax()].copy()
    
    if max_snr_rows.empty:
        # Should not happen if t_true was calculated, but as a safeguard
        return group_df['bin_width_ms'].min()

    # 2. Calculate bias for each of these max-SNR-per-bin rows
    max_snr_rows['bias'] = (max_snr_rows['median_mvt_ms'] - t_true).abs() / t_true
    
    # 3. Filter for all unbiased bins
    unbiased_bins = max_snr_rows[max_snr_rows['bias'] <= threshold]
    
    if unbiased_bins.empty:
        # If no bin is unbiased (can happen with noisy data),
        # default to the smallest bin_width (which defined t_true)
        return max_snr_rows['bin_width_ms'].min()
    
    # 4. BW* is the *largest* bin_width that passed the bias test
    bw_star = unbiased_bins['bin_width_ms'].max()
    
    return bw_star

def analyze_envelope(df, param_cols):
    """
    Main analysis function to find the MVT envelope.
    [V5] Returns T_true in the envelope DataFrames.
    """
    if df is None or df.empty:
        print("--- Skipping analysis, DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    group_by_cols = ['pulse_shape'] + param_cols
    print(f"--- Grouping by: {group_by_cols}")
    grouped = df.groupby(group_by_cols)
    param_lookup = {}
    
    # --- Part 1: Loop to find refined T_true and BW* ---
    print("--- Running 2-step analysis for T_true and BW*...")
    
    for name, group in grouped:
        if group.empty: continue
        t_true_anchor = calculate_t_true(group)
        bw_star = calculate_bw_star(group, t_true_anchor, BIAS_GATE_THRESHOLD)
        
        max_snr_rows = group.loc[group.groupby('bin_width_ms')['MVT_snr'].idxmax()]
        stable_rows = max_snr_rows[max_snr_rows['bin_width_ms'] <= bw_star]
        
        t_true_final = t_true_anchor
        if not stable_rows.empty:
            t_true_final = stable_rows['median_mvt_ms'].mean()
        
        param_lookup[name] = {'T_true': t_true_final, 'BW_star': bw_star}

    print("--- Calculated FINAL T_true and BW* for all groups.")
    
    # --- Part 2: Map values and apply new statistical filters ---
    df['group_key'] = list(zip(*[df[col] for col in group_by_cols]))
    df['T_true'] = df['group_key'].map(lambda x: param_lookup.get(x, {}).get('T_true'))
    df['BW_star'] = df['group_key'].map(lambda x: param_lookup.get(x, {}).get('BW_star'))
    df = df.dropna(subset=['T_true', 'BW_star'])

    # --- Step 4: Find "Valid Points" ---
    is_in_envelope = df['bin_width_ms'] <= df['BW_star']
    
    err_lower_mag = df['mvt_err_lower'].clip(lower=0)
    err_upper_mag = df['mvt_err_upper'].clip(lower=0)
    point_lower_bound = df['median_mvt_ms'] - err_lower_mag
    point_upper_bound = df['median_mvt_ms'] + err_upper_mag
    
    is_stat_consistent = (df['T_true'] >= point_lower_bound) & (df['T_true'] <= point_upper_bound)
    
    df['rel_err'] = (err_lower_mag + err_upper_mag) / df['median_mvt_ms']
    is_low_rel_err = df['rel_err'] <= RELERR_LIMIT
    
    df['is_in_envelope'] = is_in_envelope
    df['is_error_aware'] = is_low_rel_err
    
    df['filter_no_error'] = is_in_envelope & is_stat_consistent
    df['filter_error_aware'] = is_in_envelope & is_stat_consistent & is_low_rel_err

    # --- Step 5: Find "Min SNR" ---
    df_no_error = df[df['filter_no_error']]
    df_error_aware = df[df['filter_error_aware']]

    # --- [NEW] Create envelope DFs that include T_true ---
    if df_no_error.empty:
        print("--- WARNING: No data survived the 'no-error' (stat. consistent) filters.")
        no_error_envelope = pd.DataFrame(columns=group_by_cols + ['min_MVT_snr', 'T_true'])
    else:
        # Get min_MVT_snr and the corresponding T_true (which is constant)
        no_error_min_snr = df_no_error.groupby(group_by_cols)['MVT_snr'].min()
        no_error_t_true = df_no_error.groupby(group_by_cols)['T_true'].first()
        no_error_envelope = pd.DataFrame({'min_MVT_snr': no_error_min_snr, 'T_true': no_error_t_true}).reset_index()

    if df_error_aware.empty:
        print(f"--- WARNING: No data survived the 'error-aware' (Rel. Err < {RELERR_LIMIT}) filters.")
        error_aware_envelope = pd.DataFrame(columns=group_by_cols + ['min_MVT_snr', 'T_true'])
    else:
        error_aware_min_snr = df_error_aware.groupby(group_by_cols)['MVT_snr'].min()
        error_aware_t_true = df_error_aware.groupby(group_by_cols)['T_true'].first()
        error_aware_envelope = pd.DataFrame({'min_MVT_snr': error_aware_min_snr, 'T_true': error_aware_t_true}).reset_index()

    # Return the annotated DF and the final envelope DFs
    return df, no_error_envelope, error_aware_envelope



def plot_envelope_publication(df_annotated, no_err_points, err_aware_points, pulse_shape, output_filename):
    """
    Generates and saves the single-panel "Publication" plot.
    [V25 - CLEANUP]
    
    - Fits SNR = f(MVT) with a 2nd-order poly.
    - Uses bootstrap for a horizontal 95% CI.
    - Background points are colored by their 'T_true' value.
    - Adds a HORIZONTAL colorbar to the TOP of the plot (80% width).
    - [REMOVED] Connecting lines have been removed for clarity.
    - Saves the final fit model to 'mvt_snr_fit_model.npz'.
    """
    print(f"--- Generating publication-ready plot (Bootstrap CI + Colormap) for {pulse_shape}...")
    
    # --- 1. Define font sizes ---
    FONT_TITLE = 22
    FONT_LABEL = 20
    FONT_TICK = 18
    FONT_LEGEND = 14
    alpha_bg = 0.5  # Transparency for background points
    color_fill = 'orange'  # Color for CI shading
    color_min_snr = 'red'  # Color for Min SNR point

    # --- 2. Get background points ---
    bg_no_error = df_annotated[df_annotated['is_in_envelope']]

    # --- 3. Set up continuous colormap based on T_true ---
    if bg_no_error.empty:
        print("--- No background points to plot. Skipping plot.")
        return

    # Get the log of T_true for color mapping
    log_t_true = np.log10(bg_no_error['T_true'])
    log_t_true = log_t_true.replace([np.inf, -np.inf], np.nan).dropna()
    
    if log_t_true.empty:
        print("--- [WARNING] No valid T_true values. Defaulting to gray.")
        vmin = 0
        vmax = 1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mcolors.ListedColormap(['gray'])
        use_colorbar = False
    else:
        vmin = log_t_true.min()
        vmax = log_t_true.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis # A good perceptually-uniform colormap
        use_colorbar = True

    # --- 4. Create Plot (Single Panel) ---
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 10)) # Single panel
    # We create 'handles' for a custom legend
    handles = []
    
    # --- 5. Plot Background Points with Colormap ---
    
    print("--- Plotting background points with T_true colormap...")
    for group_key, group_data in bg_no_error.groupby('group_key'):
        if group_data.empty:
            continue
            
        # Get the T_true for this group and find its color
        t_true = group_data['T_true'].iloc[0]
        color = cmap(norm(np.log10(t_true)))
        
        # Get marker based on pulse shape
        shape = group_data['pulse_shape'].iloc[0]
        if shape == 'gaussian': marker = 'o'
        elif shape == 'norris': marker = 's'
        elif shape == 'triangular': marker = '^'
        else: marker = 'x' # Fallback
        
        y_vals = group_data['median_mvt_ms']
        y_err_lower_mag = group_data['mvt_err_lower'].clip(lower=0)
        y_err_upper_mag = group_data['mvt_err_upper'].clip(lower=0)
        
        # Plot the error bars
        ax1.errorbar(
            group_data['MVT_snr'], y_vals, yerr=[y_err_lower_mag, y_err_upper_mag],
            fmt=marker, color=color, ecolor=color, alpha=alpha_bg, elinewidth=0.5,
            capsize=0, zorder=2, mfc=color, mec='none', ms=5
        )
        
        # --- [REMOVED] Connecting lines block ---
            
    # --- Add dummy markers for the legend ---
    h_g = ax1.scatter([], [], marker='o', color='gray', label='Gaussian')
    h_n = ax1.scatter([], [], marker='s', color='gray', label='Norris')
    h_t = ax1.scatter([], [], marker='^', color='gray', label='Triangular')
    handles.extend([h_t, h_n, h_g]) # Order them as you like

    # --- 6. Highlight Min SNR & Fit (Unchanged) ---
    if not no_err_points.empty:
        h = ax1.scatter(
            no_err_points['min_MVT_snr'], no_err_points['T_true'],
            marker='D', color='red', edgecolor='k', s=50,
            label='Min. Valid SNR', zorder=10
        )
        handles.append(h)
        
        # --- Bootstrap Resampling for Confidence Interval ---
        if len(no_err_points) > 2: 
            
            print("--- Starting Bootstrap resampling for fit CI (10000 iterations)...")
            
            x_fit_data_log = np.log10(no_err_points['T_true'])
            y_fit_data_log = np.log10(no_err_points['min_MVT_snr'])

            fit_data_clean = pd.DataFrame({
                'x': x_fit_data_log, 'y': y_fit_data_log
            }).replace([np.inf, -np.inf], np.nan).dropna()
            
            x_clean = fit_data_clean['x'] # log(MVT)
            y_clean = fit_data_clean['y'] # log(SNR)
            n_points = len(x_clean)
            
            if n_points <= 2:
                print("--- [WARNING] Not enough valid points to perform bootstrap fit. Skipping fit.")
            else:
                x_grid_log = np.linspace(x_clean.min(), x_clean.max(), 100)
                n_bootstraps = 10000
                bootstrap_y_preds = np.zeros((n_bootstraps, len(x_grid_log)))
                
                for i in range(n_bootstraps):
                    indices = np.random.choice(range(n_points), size=n_points, replace=True)
                    x_boot = x_clean.iloc[indices]
                    y_boot = y_clean.iloc[indices]
                    try:
                        coeffs_boot = np.polyfit(x_boot, y_boot, 2)
                        poly_boot = np.poly1d(coeffs_boot)
                        bootstrap_y_preds[i] = poly_boot(x_grid_log)
                    except np.linalg.LinAlgError:
                        bootstrap_y_preds[i] = np.nan
                
                lower_bound_log = np.nanpercentile(bootstrap_y_preds, 2.5, axis=0)
                upper_bound_log = np.nanpercentile(bootstrap_y_preds, 97.5, axis=0)
                median_fit_log = np.nanpercentile(bootstrap_y_preds, 50.0, axis=0)
                
                print("--- Bootstrap complete.")
                
                # --- Save fit model to NPZ file (Unchanged) ---
                fit_output_filename = "mvt_snr_fit_model.npz"
                np.savez(
                    fit_output_filename,
                    mvt_grid_log=x_grid_log,     # log10(MVT) grid
                    snr_median_log=median_fit_log, # log10(SNR) median fit
                    snr_lower_log=lower_bound_log, # log10(SNR) 2.5th percentile
                    snr_upper_log=upper_bound_log  # log10(SNR) 97.5th percentile
                )
                print(f"--- Saved fit model to {fit_output_filename}")

                # --- Plot shaded region (Unchanged) ---
                h_fill = ax1.fill_betweenx(
                    10**x_grid_log, 10**lower_bound_log, 10**upper_bound_log,
                    color=color_fill, alpha=0.2, zorder=1, label='95% CI (Bootstrap)'
                )
                handles.append(h_fill)
                
                # --- Plot best-fit line (Unchanged) ---
                h_line, = ax1.plot(
                    10**median_fit_log, 10**x_grid_log,
                    linestyle='-', color=color_min_snr, lw=3, zorder=11, label='2nd Order Fit (Median)'
                )
                handles.append(h_line)
    
    # --- 7. Plot Formatting (Unchanged) ---
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel('MVT (ms)', fontsize=FONT_LABEL)
    ax1.set_xlabel(r'SNR$_{{\mathrm{MVT}}}$', fontsize=FONT_LABEL)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    
    # --- 8. Create Legend and Colorbar ---
    # Create the main legend
    ax1.legend(handles=handles[::-1], fontsize=FONT_LEGEND, loc='lower right', ncol=3) 

    # --- [MODIFIED] Add Horizontal Colorbar ---
    if use_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Add the colorbar, placing it horizontally on top
        cbar = fig.colorbar(sm, ax=ax1, 
                            orientation='horizontal', 
                            location='top',
                            pad=0.01, # Padding
                            shrink=0.95, # [MODIFIED] Make it 80% of the plot width
                            aspect=30)  # Aspect ratio (long and thin)
        cbar.set_label(r'Log$_{10}$(MVT$_{\mathrm{0}}$ [ms])', size=20, labelpad=10)
        cbar.ax.tick_params(labelsize=FONT_TICK)

    # --- 9. Save Plot (Unchanged) ---
    plt.ylim(bottom=0.1)
    # Use standard tight_layout, it works better with location='top'
    plt.tight_layout() 
    plt.savefig(output_filename, dpi=300)
    print(f"--- Publication plot saved to {output_filename}")
    plt.close(fig)


    
# --- Main Execution ---
def main():
    
    # --- Define all 4 valid file paths ---
    GAUSS_FILE_PATH = '03_BW_AMP_Gauss_ALL.csv'
    NORRIS_FILE_PATH = '07_Norris_summary_results.csv'
    TRI_FILE_PATH = '06_plotted_data_MVT_snr_vs_median_mvt_ms-2.csv'
    
    GAUSS_NEW_FILE_PATH = '08_Gauss_hi_results.csv' # <-- NEW
    TRI_NEW_FILE_PATH = '08_Tri_hi_results.csv'     # <-- NEW

    # GAUSS_FILE_PATH2 ('02_...') has been REMOVED as it is incompatible
    
    PARAMETER_MAP = {
        'gaussian': ['sigma'],
        'norris': ['rise_time', 'decay_time'],
        'triangular': ['width']
    }

    # --- Load and Combine Gaussian Data (File 1 + New File) ---
    print("\n" + "="*30)
    print("  Loading & Combining Gaussian Data")
    print("="*30)
    df_gauss = load_data(GAUSS_FILE_PATH)
    df_gauss_new = load_data(GAUSS_NEW_FILE_PATH)
    
    if df_gauss_new is not None:
        if 'sigma' in df_gauss_new.columns:
            print("--- Converting new Gaussian 'sigma' from s to ms...")
            df_gauss_new['sigma'] = df_gauss_new['sigma'] * 1000
            df_gauss_new = df_gauss_new[df_gauss_new['bin_width_ms'] < 10.0].copy()
        else:
            print("--- WARNING: New Gaussian file missing 'sigma' column.")
    
    # Combine old and new Gaussian data
    df_gauss_combined = pd.concat([df_gauss, df_gauss_new], ignore_index=True)

    # --- Load and Combine Triangular Data (File 1 + New File) ---
    print("\n" + "="*30)
    print("  Loading & Combining Triangular Data")
    print("="*30)
    df_tri = load_data(TRI_FILE_PATH)
    df_tri_new = load_data(TRI_NEW_FILE_PATH)

    if df_tri is not None and 'pulse_shape' not in df_tri.columns:
        print("--- Manually adding 'pulse_shape' = 'triangular' to OLD Tri data")
        df_tri['pulse_shape'] = 'triangular'
        
    if df_tri_new is not None:
        if 'pulse_shape' not in df_tri_new.columns:
            print("--- Manually adding 'pulse_shape' = 'triangular' to NEW Tri data")
            df_tri_new['pulse_shape'] = 'triangular'
            
        if 'width' in df_tri_new.columns:
            print("--- Converting new Triangular 'width' from s to ms...")
            df_tri_new['width'] = df_tri_new['width'] * 1000
        else:
            print("--- WARNING: New Triangular file missing 'width' column.")
    df_tri_new = df_tri_new[df_tri_new['bin_width_ms'] < 10.0].copy()
    # Combine old and new Triangular data
    df_tri_combined = pd.concat([df_tri, df_tri_new], ignore_index=True)

    # --- Load Norris Data ---
    print("\n" + "="*30)
    print("  Loading Norris Data")
    print("="*30)
    df_norris = load_data(NORRIS_FILE_PATH)

    # --- Run Analysis on Combined Gaussian ---
    print("\n" + "="*30)
    print("  Analyzing ALL Gaussian Data")
    print("="*30)
    df_g_annotated, g_env_no_err, g_env_err_aware = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if df_gauss_combined is not None and not df_gauss_combined.empty:
        df_g_annotated, g_env_no_err, g_env_err_aware = analyze_envelope(
            df_gauss_combined[df_gauss_combined['pulse_shape'] == 'gaussian'], 
            PARAMETER_MAP['gaussian']
        )

    # --- Run Analysis on Norris ---
    print("\n" + "="*30)
    print("  Analyzing Norris Data")
    print("="*30)
    df_n_annotated, n_env_no_err, n_env_err_aware = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if df_norris is not None and not df_norris.empty:
        if all(col in df_norris.columns for col in PARAMETER_MAP['norris']):
            df_n_annotated, n_env_no_err, n_env_err_aware = analyze_envelope(
                df_norris[df_norris['pulse_shape'] == 'norris'], 
                PARAMETER_MAP['norris']
            )
        else: print(f"--- SKIPPING Norris analysis (missing columns).")

    # --- Run Analysis on Combined Triangular ---
    print("\n" + "="*30)
    print("  Analyzing ALL Triangular Data")
    print("="*30)
    df_t_annotated, t_env_no_err, t_env_err_aware = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if df_tri_combined is not None and not df_tri_combined.empty:
        if all(col in df_tri_combined.columns for col in PARAMETER_MAP['triangular']):
            df_t_annotated, t_env_no_err, t_env_err_aware = analyze_envelope(
                df_tri_combined[df_tri_combined['pulse_shape'] == 'triangular'], 
                PARAMETER_MAP['triangular']
            )
        else: print(f"--- SKIPPING Triangular analysis (missing columns).")

    # --- Process Special Gaussian File 2 has been REMOVED ---

    # --- Combine ALL Results ---
    print("\n" + "="*30)
    print("  Combining All Results for Plotting")
    print("="*30)
    
    # We now only combine the 3 valid, annotated datasets
    combined_df = pd.concat(
        [df_g_annotated, df_n_annotated, df_t_annotated], 
        ignore_index=True
    )
    # We combine the 3 valid sets of envelope points
    combined_no_err = pd.concat(
        [g_env_no_err, n_env_no_err, t_env_no_err], 
        ignore_index=True
    )
    combined_err_aware = pd.concat(
        [g_env_err_aware, n_env_err_aware, t_env_err_aware], 
        ignore_index=True
    )

    print(f"Total background points: {len(combined_df)}")
    print(f"Total 'No-Error' Min SNR points: {len(combined_no_err)}")
    print(f"Total 'Error-Aware' Min SNR points: {len(combined_err_aware)}")

    # --- Plot BOTH versions ---
    if not combined_df.empty:
        
        
        # 2. Generate the "Publication" plot
        plot_envelope_publication(
            combined_df, 
            combined_no_err, 
            combined_err_aware, 
            "Combined (Gauss + Norris + Tri)",
            "08_combined_mvt_snr_envelope_PAPER.png" # Note the filename
        )
        
    else:
        print("--- No data to plot.")


if __name__ == "__main__":
    # To run, you will need:
    # pip install pandas matplotlib numpy adjustText
    main()
