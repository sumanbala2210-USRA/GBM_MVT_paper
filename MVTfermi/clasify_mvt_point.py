#clasify_mvt_point.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

# --- Define Global Constants ---
MODEL_FILE = 'mvt_snr_fit_model.npz'
FONT_LABEL = 14
FONT_TICK = 12
FONT_LEGEND = 12

def load_model():
    """
    Loads the saved NPZ fit model.
    Returns a dictionary of interpolation functions.
    """
    if not os.path.exists(MODEL_FILE):
        print(f"--- ERROR: Model file not found! ---")
        print(f"Please make sure '{MODEL_FILE}' is in the same directory.")
        return None
        
    try:
        model = np.load(MODEL_FILE)
        mvt_grid_log = model['mvt_grid_log']
        
        # Create interpolation functions for all 3 lines
        interpolators = {
            'median': interp1d(mvt_grid_log, model['snr_median_log'],
                               kind='linear', bounds_error=False, fill_value='extrapolate'),
            'lower': interp1d(mvt_grid_log, model['snr_lower_log'],
                              kind='linear', bounds_error=False, fill_value='extrapolate'),
            'upper': interp1d(mvt_grid_log, model['snr_upper_log'],
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        }
        return interpolators, (mvt_grid_log.min(), mvt_grid_log.max())
        
    except Exception as e:
        print(f"--- ERROR: Could not load fit model. Reason: {e} ---")
        return None

def classify_mvt_point(mvt_value, snr_value, interpolators, mvt_range_log):
    """
    Classifies a new (MVT, SNR) point against the loaded model.
    """
    mvt_log = np.log10(mvt_value)
    snr_log = np.log10(snr_value)
    
    # Check if MVT is outside the fit range
    if mvt_log < mvt_range_log[0] or mvt_log > mvt_range_log[1]:
        print(f"--- WARNING: MVT value {mvt_value} ms is outside the model's fitted range. ---")
        print("    Classification may be unreliable (extrapolated).")
    
    # Find the boundaries at the new MVT value
    snr_median_bound = interpolators['median'](mvt_log)
    snr_lower_bound = interpolators['lower'](mvt_log)
    snr_upper_bound = interpolators['upper'](mvt_log)
    
    # Perform Classification
    if snr_log < snr_lower_bound:
        classification = "Below 95% CI (Upper Limit)"
    elif snr_log > snr_upper_bound:
        classification = "Above 95% CI (Robast Measurement)"
    else:
        classification = "Within 95% CI (Likely Upper Limit)"
        
    return classification, (snr_lower_bound, snr_median_bound, snr_upper_bound)

def plot_classification(mvt_value, snr_value, classification, bounds_log, 
                        interpolators, mvt_range_log, output_filename):
    """
    Generates a simplified plot showing the fit and the new point.
    """
    print(f"--- Generating plot: {output_filename} ---")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # --- 1. Plot the saved model (Median line and CI band) ---
    mvt_grid_log_plot = np.linspace(mvt_range_log[0], mvt_range_log[1], 100)
    
    snr_median_log = interpolators['median'](mvt_grid_log_plot)
    snr_lower_log = interpolators['lower'](mvt_grid_log_plot)
    snr_upper_log = interpolators['upper'](mvt_grid_log_plot)

    # Plot the shaded region (CI)
    ax.fill_betweenx(
        10**mvt_grid_log_plot, 10**snr_lower_log, 10**snr_upper_log,
        color='orange', alpha=0.2, zorder=1, label='95% CI (Bootstrap)'
    )
    
    # Plot the best-fit line (Median)
    ax.plot(
        10**snr_median_log, 10**mvt_grid_log_plot,
        linestyle='-', color='red', lw=3, zorder=11, label='2nd Order Fit (Median)'
    )
    
    # --- 2. Plot the new highlight point ---
    ax.plot(
        snr_value, mvt_value,
        marker='*', ms=20, mec='cyan', mfc='#000000AA', mew=2,
        zorder=20, label='Your Point'
    )
    
    # --- 3. Format the plot ---
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylabel('MVT (ms)', fontsize=FONT_LABEL)
    ax.set_xlabel(r'SNR$_{{\mathrm{MVT}}}$', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(fontsize=FONT_LEGEND, loc='upper right')
    
    # Set a reasonable view range
    ax.set_ylim(bottom=0.3, top=3000)
    ax.set_xlim(left=10, right=2000)
    
    # Add classification text
    plt.title(f"Classification: {classification}", fontsize=FONT_LABEL)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"--- Plot saved to {output_filename} ---")


def main():
    """
    Main entry point for the command-line tool.
    
    Usage:
    python classify_mvt_point.py --mvt <value> --snr <value> [--mode <plot|classify>]
    
    Example:
    python classify_mvt_point.py --mvt 100 --snr 80 --mode plot
    """
    
    # --- 1. Parse Command-Line Arguments ---
    args = sys.argv[1:]
    if len(args) < 4 or '--mvt' not in args or '--snr' not in args:
        print(__doc__) # Print the docstring as help
        return

    try:
        mvt = float(args[args.index('--mvt') + 1])
        snr = float(args[args.index('--snr') + 1])
    except (ValueError, IndexError):
        print("--- ERROR: --mvt and --snr must be followed by numbers. ---")
        print(__doc__)
        return
        
    mode = 'classify' # Default
    if '--mode' in args:
        try:
            mode_arg = args[args.index('--mode') + 1].lower()
            if mode_arg in ['plot', 'classify']:
                mode = mode_arg
            else:
                print(f"--- WARNING: Unknown mode '{mode_arg}'. Defaulting to 'classify'. ---")
        except IndexError:
            pass # Use default

    print(f"--- Classifying point: MVT={mvt} ms, SNR={snr} (Mode: {mode}) ---")

    # --- 2. Load Model ---
    model_data = load_model()
    if model_data is None:
        return
    interpolators, mvt_range_log = model_data

    # --- 3. Run Classification ---
    classification, bounds_log = classify_mvt_point(mvt, snr, interpolators, mvt_range_log)
    
    print(f"    Result: {classification}")
    
    # --- 4. Run Plot (if requested) ---
    if mode == 'plot':
        output_filename = f"classification_MVT_{mvt}_SNR_{snr}.png"
        plot_classification(mvt, snr, classification, bounds_log, 
                            interpolators, mvt_range_log, output_filename)

if __name__ == "__main__":
    main()