# run_haar_power.py
# This script should be placed in the other Python environment.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import json
from haar_power_mod import haar_power_mod # This environment has the tool installed


# In run_haar_power_mod.py

# (Your existing haar_power_mod function stays here, unchanged)

def time_resolved_mvt(
    counts: np.ndarray, 
    errors: np.ndarray, 
    min_dt: float, 
    window_size_s: float, 
    step_size_s: float,
    t_start: float = 0.0, # <<< NEW ARGUMENT
    **kwargs
) -> list:
    """
    Calculates the Minimum Variability Timescale (MVT) using a sliding time window,
    correctly handling an absolute start time.

    Args:
        counts (np.ndarray): The full binned light curve data.
        errors (np.ndarray): The errors for the counts.
        min_dt (float): The bin width in seconds.
        window_size_s (float): The total width of the sliding window in seconds.
        step_size_s (float): The amount to slide the window forward in seconds.
        t_start (float): The absolute start time of the counts array. Defaults to 0.0.
        **kwargs: Additional arguments to pass to haar_power_mod.

    Returns:
        list: A list of dictionaries containing the results for each time window.
    """
    window_size_bins = int(window_size_s / min_dt)
    step_size_bins = int(step_size_s / min_dt)

    if window_size_bins > len(counts):
        print("Warning: Window size is larger than the light curve. No analysis performed.")
        return []
        
    all_results = []
    
    start_bin = 0
    while start_bin + window_size_bins <= len(counts):
        end_bin = start_bin + window_size_bins
        
        window_counts = counts[start_bin:end_bin]
        window_errors = errors[start_bin:end_bin]
        
        # <<< CHANGE: Add the t_start offset to all time calculations >>>
        window_center_time = t_start + (start_bin + window_size_bins / 2.0) * min_dt
        window_start_time = t_start + start_bin * min_dt
        window_end_time = t_start + end_bin * min_dt
        
        try:
            result = haar_power_mod(window_counts, window_errors, min_dt=min_dt, **kwargs)
            mvt = round(result[2]*1000, 3)
            mvt_err = round(result[3]*1000, 3)
        except Exception as e:
            mvt = 0.0
            mvt_err = 0.0
            logging.warning(f"Error during MVT calculation for window at {window_center_time}s: {e}")
            
        all_results.append({
            'center_time_s': round(window_center_time, 4),
            'start_time_s': round(window_start_time, 4),
            'end_time_s': round(window_end_time, 4),
            'mvt_ms': round(mvt, 4),
            'mvt_err_ms': round(mvt_err, 4),
        })
            
        start_bin += step_size_bins
        
    return all_results


def main():
    parser = argparse.ArgumentParser(description="A wrapper to run MVT analysis.")
    # --- Existing arguments ---
    parser.add_argument("--input", required=True, help="Path to the input .npy file for counts.")
    parser.add_argument("--output", required=True, help="Path to the output .json file for results.")
    parser.add_argument("--min_dt", required=True, type=float, help="Minimum timescale (bin width).")
    parser.add_argument("--doplot", default='0', help="Whether to generate plots (only for non-time-resolved).")
    parser.add_argument("--file", default="test", help="Base name for output files.")

    # --- NEW arguments for time-resolved analysis ---
    parser.add_argument("--time-resolved", action='store_true', help="Perform time-resolved MVT analysis.")
    parser.add_argument("--window-size", type=float, default=10.0, help="Window size in seconds for time-resolved analysis.")
    parser.add_argument("--step-size", type=float, default=1.0, help="Step size in seconds for time-resolved analysis.")
    parser.add_argument("--tstart", type=float, default=0.0, help="Absolute start time of the counts array in seconds.")
    
    args = parser.parse_args()

    # 1. Load the input data
    counts = np.load(args.input)
    errors = np.sqrt(np.abs(counts))

    if args.doplot == '1':
        doplot_flag = True
    else:
        doplot_flag = False

    # 2. Run the correct analysis based on the flag
    if args.time_resolved:
        print(f"--- Running Time-Resolved MVT (Window: {args.window_size}s, Step: {args.step_size}s) ---")
        results = time_resolved_mvt(
            counts, 
            errors, 
            min_dt=args.min_dt, 
            window_size_s=args.window_size, 
            step_size_s=args.step_size,
            # Pass other relevant args through
            afactor=-1.0, 
            verbose=False, 
            weight=True, 
            file=args.file,
            t_start=args.tstart  # <<< NEW ARGUMENT PASSING
        )
    else:
        print("--- Running Standard MVT ---")
        mvt_results = haar_power_mod(
            counts, 
            errors, 
            min_dt=args.min_dt, 
            doplot=doplot_flag, 
            afactor=-1.0, 
            verbose=False, 
            weight=True, 
            file=args.file
        )
        mvt_val = round(mvt_results[2] * 1000, 3)  
        mvt_err = round(mvt_results[3] * 1000, 3)  
        results = {'mvt_ms': mvt_val, 'mvt_err_ms': mvt_err}

    plt.close('all')

    # 3. Save the results
    with open(args.output, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
