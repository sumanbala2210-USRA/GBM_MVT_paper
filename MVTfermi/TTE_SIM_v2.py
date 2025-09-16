"""
# TTE_SIM_v2.py
Suman Bala
Old: This script simulates light curves using Gaussian and triangular profiles.
7th June 2025: Including Fermi GBM simulation of same functions.
14th August 2025: Added GBM and normal functions together.

"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import concurrent.futures
import scipy.integrate as spi
import gdt.core
import yaml  # Import the JSON library for parameter logging
import warnings
from astropy.io.fits.verify import VerifyWarning
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
import pandas as pd

import re
from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- GDT Core Imports ---
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.plot.lightcurve import Lightcurve
#from gdt.core.simulate.profiles import tophat, constant, norris, quadratic, linear, gaussian
from gdt.core.simulate.tte import TteBackgroundSimulator, TteSourceSimulator
from gdt.core.spectra.functions import Band
from gdt.missions.fermi.gbm.response import GbmRsp2
from gdt.missions.fermi.gbm.tte import GbmTte
#from lib_sim import write_yaml
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
import matplotlib.pyplot as plt
from gdt.core.plot.lightcurve import Lightcurve
from gdt.core.tte import PhotonList
from gdt.core.data_primitives import EventList, Ebounds

from SIM_lib import run_mvt_in_subprocess, convert_det_to_list, _create_param_directory_name
# Suppress a common FITS warning

from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)

DEFAULT_PARAM_VALUE = -999

#warnings.simplefilter('ignore', VerifyWarning)

def sim_gbm_name_format(trigger_number, det, name_key, r_seed):
    """
    Formats the simulation name for GBM files.
    """
    if r_seed is None:
        src_name = f"TTE_bn{trigger_number}_{det}_{name_key}_src.fits"
        bkgd_name = f"TTE_bn{trigger_number}_{det}_{name_key}_bkgd.fits"
    else:
        src_name = f"TTE_bn{trigger_number}_{det}_{name_key}_r_seed_{r_seed}_src.fits"
        bkgd_name = f"TTE_bn{trigger_number}_{det}_{name_key}_r_seed_{r_seed}_bkgd.fits"
        
    #print("Source Name:", src_name)
    #print("Background Name:", bkgd_name)
    return src_name, bkgd_name



def print_nested_dict(d, indent=0):
    """
    Recursively prints a nested dictionary with simple values (int, str, list of primitives)
    printed on the same line as their key.
    """
    spacing = "  " * indent

    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value)
            ):
                print(f"{spacing}{repr(key)}: {value}")
            else:
                print(f"{spacing}{repr(key)}:")
                print_nested_dict(value, indent + 1)

    elif isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, (str, int, float, bool)) or (
                isinstance(item, list) and all(isinstance(v, (str, int, float, bool)) for v in item)
            ):
                print(f"{spacing}- [Index {i}]: {item}")
            else:
                print(f"{spacing}- [Index {i}]")
                print_nested_dict(item, indent + 1)
    else:
        print(f"{spacing}{repr(d)}")





def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a dictionary by one level.

    It takes a dictionary like {'a': 1, 'b': {'c': 2, 'd': 3}} and
    returns a flat dictionary {'a': 1, 'c': 2, 'd': 3}.

    Args:
        d (Dict): The dictionary to flatten.

    Returns:
        Dict: A new, flattened dictionary.
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, unpack its items
            flat_dict.update(value)
        else:
            # Otherwise, just add the key-value pair
            flat_dict[key] = value
    return flat_dict


def check_param_consistency(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    pulse_shape: str,
    dict1_name: str = 'sim_params',
    dict2_name: str = 'base_params'
) -> List[str]:
    """
    Finds all keys common to two dictionaries and checks if their values are equal.
    Handles standard types, floats (with tolerance), and NumPy arrays.

    Args:
        dict1 (Dict): The first dictionary.
        dict2 (Dict): The second dictionary.
        dict1_name (str): A descriptive name for the first dictionary for logging.
        dict2_name (str): A descriptive name for the second dictionary for logging.

    Returns:
        List[str]: A list of strings describing any discrepancies found. An
                   empty list means no discrepancies were found.
    """
    discrepancies = []
    
    # Find the set of keys that exist in both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if pulse_shape == 'complex_pulse':
        avoid_keys = {'det', 'peak_amplitude', 'position'}
        #print("Excluding keys for complex pulse:", avoid_keys)
    else:
        avoid_keys = {'det'}
    common_keys = common_keys - avoid_keys  # Exclude these keys from comparison

    for key in sorted(list(common_keys)):
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Use different comparison methods based on the data type
        are_different = False
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                are_different = True
        elif isinstance(val1, float) or isinstance(val2, float):
            if not np.isclose(val1, val2):
                are_different = True
        elif val1 != val2:
            are_different = True
            
        if are_different:
            msg = (
                f"Discrepancy found for key '{key}':\n"
                f"  - {dict1_name}['{key}'] = {val1} (type: {type(val1).__name__})\n"
                f"  - {dict2_name}['{key}'] = {val2} (type: {type(val2).__name__})"
            )
            discrepancies.append(msg)
            print("########### ERROR ##########")
            print(msg)
            #exit()

def TTE_shift(final_tte, shift):
    data = EventList(final_tte.data.times+shift, final_tte.data.channels, ebounds=final_tte.data.ebounds)
    tte = PhotonList.from_data(data, gti=final_tte.gti, trigger_time=final_tte.trigtime,
                           event_deadtime=final_tte.event_deadtime, overflow_deadtime=final_tte.overflow_deadtime)
    return tte


def calculate_src_interval(params: Dict) -> Tuple[float, float]:
    """
    Calculates the 'true' source interval directly from the pulse model parameters.
    """
    pulse_shape = params['pulse_shape'] #, pulse_shape)
    if pulse_shape == 'gaussian':
        # For a Gaussian, the interval containing >99.7% of the flux is +/- 3-sigma
        sigma = params['sigma']
        center = params['center_time']
        return center - 3 * sigma, center + 3 * sigma

    elif pulse_shape == 'triangular':
        # For a triangular pulse, the start and stop are explicitly defined
        width = params['width']
        center = params['center_time']
        peak_ratio = params['peak_time_ratio']
        t_start = center - (width * peak_ratio)
        t_stop = t_start + width
        return t_start, t_stop

    elif pulse_shape in ['norris', 'fred']:
        # For pulses with long tails, we can define a practical window,
        # e.g., from the start time to where the pulse has significantly decayed.
        # Here, we approximate this as the peak + 7 decay timescales.
        t_rise = params['rise_time']
        t_decay = params['decay_time']
        t_start = params['start_time']
        # Rough peak time is a few rise times after the start
        peak_time_approx = t_start + 2 * t_rise 
        t_stop = peak_time_approx + 7 * t_decay
        return t_start, t_stop

    elif pulse_shape == 'complex_pulse':
        # For complex pulses, we can define a practical window based on the parameters
        t_start = 4.4
        t_stop = 14
        return t_start, t_stop

    # Add other pulse shapes as needed...

    # Fallback if shape is unknown
    return params.get('src_start'), params.get('src_stop')



def parse_intervals_from_csv(value_from_csv: str) -> List[List[float]]:
    """
    Parses a string like '-20.0, -5.0, 75.0, 200.0' from a CSV cell
    into a list of pairs, e.g., [[-20.0, -5.0], [75.0, 200.0]].
    """
    # Ensure the input is treated as a string
    s = str(value_from_csv).strip()
    
    # Use a regular expression to split by comma and/or any amount of space
    # This is very robust against formatting like "-20, -5" or "-20 -5"
    parts = re.split(r'[,\s]+', s)
    
    # Convert non-empty parts to floats
    vals = [float(p) for p in parts if p]
    
    # Check for an even number of values
    if len(vals) % 2 != 0:
        raise ValueError(
            f"background_intervals must contain an even number of values, but got {len(vals)} from '{value_from_csv}'"
        )
        
    # Group the flat list into a list of [start, end] pairs
    return [vals[i:i+2] for i in range(0, len(vals), 2)]



def calculate_adaptive_simulation_params(pulse_shape: str, params: Dict) -> Dict:
    """
    Calculates optimal t_start, t_stop, and grid_resolution based on pulse parameters.
    """
    t_start, t_stop, grid_res = None, None, None
    padding = 10.0  # Seconds of padding before and after the pulse

    if pulse_shape == 'gaussian':
        sigma = params['sigma']
        center = params['center_time']
        grid_res = sigma / 10.0
        # The pulse is significant within ~5-sigma of the center
        t_start = center - 5 * sigma - padding * grid_res * 20
        t_stop = center + 5 * sigma + padding * grid_res * 20
        # Rule of Thumb: Grid must be ~10x finer than the narrowest feature

    elif pulse_shape == 'lognormal':
        sigma = params['sigma']
        center = params['center_time']
        timescale = min(center, sigma * center)
        grid_res = timescale / 10.0
        # Use a similar 5-sigma rule, but in log-space
        t_start = center * np.exp(-5 * sigma - padding * grid_res * 20)
        t_stop = center * np.exp(5 * sigma + padding * grid_res * 20)
        # A rough characteristic timescale for lognormal
        
    elif pulse_shape in ['norris', 'fred']:
        t_rise = params['rise_time']
        t_decay = params['decay_time']
        start = params['start_time']
        # For FRED/Norris, the rise time is the narrowest feature
        grid_res =  min(t_rise, t_decay) / 10.0
        # Determine the peak time to set a reasonable end point
        # A simple approximation for the peak time
        peak_time = np.sqrt(t_rise * t_decay)
        # End the simulation after the pulse has decayed significantly (~10x decay time)
        t_start = start - 2 * t_rise 
        t_stop = start + 6 * t_decay 

    elif pulse_shape == 'triangular':
        width = params['width']
        center = params['center_time']
        peak_ratio = params['peak_time_ratio']
        rise_duration = width * peak_ratio
        fall_duration = width * (1.0 - peak_ratio)
        grid_res = min(rise_duration, fall_duration) / 10.0
        # Start and stop are explicitly defined by the parameters
        t_start = center - (width * peak_ratio) - padding * grid_res * 5
        #t_stop = t_start + width + padding + (padding*2)
        t_stop = center + width * (1.0 - peak_ratio) + padding * grid_res * 5
        # Narrowest feature is the shorter of the rise or fall time
    
    elif pulse_shape == 'complex_pulse':
        t_start = params.get('t_start', -5.0) 
        t_stop = params.get('t_stop', 5.0)
        grid_res = params.get('grid_resolution', 0.001)

    # --- Safety Net ---
    # Ensure grid resolution is within reasonable bounds to prevent memory errors
    # or inaccurate simulations.
    if grid_res is not None:
        grid_res = np.clip(grid_res, a_min=1e-7, a_max=0.001) # e.g., 1µs to 1ms
    else:
        # Fallback for unknown shapes
        t_start, t_stop, grid_res = -5.0, 5.0, 0.0001
        
    return {'t_start': t_start, 't_stop': t_stop, 'grid_resolution': grid_res}







def _format_params_for_annotation(func: Callable, func_par: Tuple) -> str:
    """Formats function and parameters into a concise string for a plot title."""
    if not func or not func_par:
        return "No Source Model"

    # This handles our specific case where func is generate_pulse_function
    # and func_par is a tuple containing a single parameter dictionary.
    if func.__name__ == 'generate_pulse_function' and isinstance(func_par[0], dict):
        params_dict = func_par[0]
        pulse_strings = []
        
        # Loop through the pulse definitions in the dictionary
        for pulse_def in params_dict.get('pulse_list', []):
            p_type, p_params = pulse_def
            # Format numbers to 3 decimal places to keep the title clean
            param_str = ", ".join([f"{round(p,3)}" for p in p_params])
            pulse_strings.append(f"{p_type}({param_str})\n")
        
        if not pulse_strings:
            return "Empty Pulse List"
        # Join multiple pulses with a plus sign
        return " ".join(pulse_strings)

    # This is a generic fallback for other function types
    else:
        try:
            func_name = func.__name__
            param_str = ", ".join([f"{round(p,3)}" for p in func_par])
            return f"{func_name}({param_str})"
        except TypeError:
            # Fallback for non-numeric or complex parameters
            return f"{func.__name__}{func_par}"




def _calculate_multi_timescale_snr(
    total_counts: np.ndarray,
    sim_bin_width: float,
    back_avg_cps: float,
    search_timescales: List[float]
) -> Dict[str, float]:
    """
    Calculates SNR by finding the peak in the total light curve and
    subtracting the expected background.

    Args:
        total_counts (np.ndarray): High-resolution binned light curve of TOTAL (source + bkg) events.
        sim_bin_width (float): The bin width of the high-resolution light curve (in seconds).
        back_avg_cps (float): The average background rate in counts per second.
        search_timescales (List[float]): A list of timescales (in seconds) to search.

    Returns:
        A dictionary of SNR values for each timescale.
    """
    snr_results = {}

    for timescale in search_timescales:
        try:
            factor = max(1, int(round(timescale / sim_bin_width)))
            end = (len(total_counts) // factor) * factor
            if end == 0:
                snr_results[f'S{int(timescale*1000)}'] = 0.0
                continue
            
            # Re-bin the TOTAL counts
            rebinned_total_counts = np.sum(total_counts[:end].reshape(-1, factor), axis=1)
            
            # Find the total number of counts in the brightest time window
            counts_in_peak_bin = np.max(rebinned_total_counts)
            
            # <<< NEW: Calculate Signal and Noise via background subtraction >>>
            expected_bkg_in_bin = back_avg_cps * timescale
            
            # The signal is the excess counts above the background
            signal = counts_in_peak_bin - expected_bkg_in_bin
            
            # The noise is the Poisson error on the total counts in that bin
            noise = np.sqrt(counts_in_peak_bin)
            
            snr = signal / noise if noise > 0 else 0
            snr_results[f'S{int(timescale*1000)}'] = round(snr, 2)
            
        except Exception:
            snr_results[f'S{int(timescale*1000)}'] = -1.0
            continue

    return snr_results


def create_final_plot(
    source_events: np.ndarray,
    background_events: np.ndarray,
    model_info: Dict,
    output_info: Dict
):
    """
    A self-contained function that takes raw event data and creates a
    final, styled, and richly annotated representative plot.
    """
    try:
        # --- 1. Unpack all necessary data and parameters ---
        params = model_info['base_params']
        #print_nested_dict(params)
        func_to_use = model_info['func']
        func_par = model_info['func_par']
        fig_name = output_info['file_path'] / f"LC_{output_info['file_info']}.png"
        base_title = f" LC {output_info['file_info']}"
        t_start, t_stop = params['t_start'], params['t_stop']
        background_level_cps = params['background_level']* params.get('scale_factor', 1.0)
        
        # --- 2. Prepare Data for Plotting (Binning) ---
        total_events = np.sort(np.concatenate([source_events, background_events]))
        bin_width = params.get('bin_width_for_plot', 0.01)
        bins = np.arange(t_start, t_stop + bin_width, bin_width)
        times = bins[:-1] + bin_width / 2.0
        total_counts, _ = np.histogram(total_events, bins=bins)
        source_only_counts, _ = np.histogram(source_events, bins=bins)
        
        # <<< NEW: Calculate multi-timescale SNR >>>
        duration = t_stop - t_start
        total_counts_fine, _ = np.histogram(total_events, bins=int(duration / 0.001))
        snr_results_dict = _calculate_multi_timescale_snr(
            total_counts=total_counts_fine,
            sim_bin_width=0.001,
            back_avg_cps=background_level_cps,
            search_timescales= model_info['snr_analysis']
        )
        
        # <<< NEW: Format SNR results for the title >>>
        snr_annotation_parts = []
        for ts, snr_val in snr_results_dict.items():
            label = f"S$_{{{ts[1:]}}}$" # Use LaTeX for subscript
            value = f"{snr_val:.1f}"
            snr_annotation_parts.append(f"{label}={value}")
        SNR_text = "; ".join(snr_annotation_parts)
        final_title = f"{base_title}\n{SNR_text}"

        # <<< NEW: Format model parameters for the annotation box >>>
        annotation_text = _format_params_for_annotation(func_to_use, func_par)

        # --- 3. Define the plot data (for 'decomposed' plot type) ---
        ideal_background_counts = background_level_cps * bin_width
        plot_data = [
            {'x': times, 'y': total_counts, 'label': 'Total Signal (Simulated)', 'color': 'rosybrown', 'fill_alpha': 0.6},
            {'x': times, 'y': source_only_counts, 'label': 'Source Signal (Simulated)', 'color': 'darkgreen', 'fill_alpha': 0.4}
        ]
        h_lines = [{'y': ideal_background_counts, 'label': f'Ideal Background ({background_level_cps:.1f} cps)', 'color': 'k'}]
        ylabel = f"Counts per {bin_width*1000:.1f} ms Bin"
            
        # --- 4. Create the Plot (Core Matplotlib Logic) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        for data in plot_data:
            ax.step(data['x'], data['y'], where='mid', label=data.get('label'), color=data.get('color'), lw=data.get('lw', 1.5))
            if 'fill_alpha' in data:
                ax.fill_between(data['x'], data['y'], step="mid", color=data.get('color'), alpha=data.get('fill_alpha'))
        for line in h_lines:
            ax.axhline(y=line['y'], color=line.get('color'), linestyle='--', label=line.get('label'))

        ax.set_title(final_title, fontsize=12) # Use new dynamic title
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper right')
        ax.set_xlim(t_start, t_stop)
        ax.set_ylim(bottom=0)

        # Use the new dynamic annotation text
        if annotation_text:
            props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
            ax.text(0.03, 0.97, annotation_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
        
        fig.tight_layout()
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)
        #logging.info(f"Representative plot saved to {fig_name}")

    except Exception as e:
        print(f"Failed to generate representative plot. Error: {e}")
        pass
        #logging.error(f"Failed to generate representative plot. Error: {e}")



def create_final_plot_with_MVT_number(
    source_events: np.ndarray,
    background_events: np.ndarray,
    model_info: Dict,
    output_info: Dict,
    mvt_summary_df: pd.DataFrame = None
):
    """
    Creates a final plot showing a colorful, decomposed light curve and
    overlays the time-resolved MVT results with horizontal error bars
    representing the calculation window.
    """
    try:
        # --- 1. Unpack data and parameters ---
        params = model_info['base_params']
        # <<< CHANGE: Get the MVT window size from model_info >>>
        mvt_window_size_s = model_info.get('mvt_window_size_s', 1.0) # Default to 1.0s if not provided
        
        fig_name = output_info['file_path'] / f"LC_with_MVT_{output_info['file_info']}.png"
        base_title = f"LC & MVT: {output_info['file_info']}"
        t_start, t_stop = params['t_start'], params['t_stop']
        
        # ... (Light curve data preparation is unchanged) ...
        total_events = np.sort(np.concatenate([source_events, background_events]))
        bin_width = params.get('bin_width_for_plot', 0.01)
        bins = np.arange(t_start, t_stop + bin_width, bin_width)
        times = bins[:-1] + bin_width / 2.0
        total_counts, _ = np.histogram(total_events, bins=bins)
        source_only_counts, _ = np.histogram(source_events, bins=bins)
        
        # --- 3. Create the Plot ---
        plt.style.use('seaborn-v0_8-ticks')
        fig, ax = plt.subplots(figsize=(12, 7))

        # ... (Plotting the light curve is unchanged) ...
        plot_data = [
            {'x': times, 'y': total_counts, 'label': 'Total Signal', 'color': 'rosybrown', 'fill_alpha': 0.7},
            {'x': times, 'y': source_only_counts, 'label': 'Source Signal', 'color': 'darkgreen', 'fill_alpha': 0.5}
        ]
        for data in plot_data:
            ax.step(data['x'], data['y'], where='mid', label=data.get('label'), color=data.get('color'), lw=1.5)
            if 'fill_alpha' in data:
                ax.fill_between(data['x'], data['y'], step="mid", color=data.get('color'), alpha=data.get('fill_alpha'))
        ax.set_title(base_title, fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel(f"Counts per {bin_width*1000:.1f} ms Bin", fontsize=12, color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_xlim(t_start, t_stop)
        ax.set_ylim(bottom=0)

        # --- 4. Overlay MVT data ---
        if mvt_summary_df is not None and not mvt_summary_df.empty:
            ax2 = ax.twinx()
            
            filtered_mvt_df = mvt_summary_df[mvt_summary_df['median_mvt_ms'] > 0]
            x_mvt = filtered_mvt_df['center_time_s']
            y_mvt = filtered_mvt_df['median_mvt_ms']
            y_err = [
                filtered_mvt_df['mvt_err_lower_ms'],
                filtered_mvt_df['mvt_err_upper_ms']
            ]
            # <<< CHANGE: Define the horizontal error bar width >>>
            # The error bar radius is half the total window size.
            x_err = mvt_window_size_s / 2.0
            
            # <<< CHANGE: Add 'xerr' to the errorbar call >>>
            ax2.errorbar(x_mvt, y_mvt, yerr=y_err, xerr=x_err,
                         fmt='o', color='firebrick', label='Median MVT (68% C.I.)',
                         capsize=4, markersize=6, lw=1.5, zorder=10)
            
            ax2.set_ylabel("MVT (ms)", fontsize=12, color='firebrick')
            ax2.tick_params(axis='y', labelcolor='firebrick')
            ax2.set_yscale("log")
            
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper right')

        else:
            ax.legend(loc='upper right')
        
        fig.tight_layout()
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)

    except Exception as e:
        logging.error(f"Failed to generate final plot with MVT. Error: {e}")
        pass


def create_final_plot_with_MVT(
    source_events: np.ndarray,
    background_events: np.ndarray,
    model_info: Dict,
    output_info: Dict,
    mvt_summary_df: pd.DataFrame = None
):
    """
    Creates a final plot showing a colorful, decomposed light curve and
    overlays the time-resolved MVT results, color-coded by the
    success percentage of simulation runs.
    """
    try:
        # --- Unpack data and parameters ---
        params = model_info['base_params']
        mvt_window_size_s = model_info.get('mvt_window_size_s', 1.0)
        # Use the passed-in output_info to name the new plot
        #fig_name = "test_mvt_LC.png"
        fig_name = Path(output_info['file_path']) / f"LC_with_MVT_{output_info['file_info']}.png"
        base_title = f"LC & MVT: {output_info['file_info']}"
        t_start, t_stop = params['t_start'], params['t_stop']

        # --- Prepare Light Curve Data ---
        total_events = np.sort(np.concatenate([source_events, background_events]))
        bin_width = params.get('bin_width_for_plot', 0.01)
        bins = np.arange(t_start, t_stop + bin_width, bin_width)
        times = bins[:-1] + bin_width / 2.0
        total_counts, _ = np.histogram(total_events, bins=bins)
        source_only_counts, _ = np.histogram(source_events, bins=bins)

        # --- Create the Plot ---
        plt.style.use('seaborn-v0_8-ticks')
        fig, ax = plt.subplots(figsize=(13, 7))

        plot_data = [
            {'x': times, 'y': total_counts, 'label': 'Total Signal', 'color': 'rosybrown', 'fill_alpha': 0.2},
            {'x': times, 'y': source_only_counts, 'label': 'Source Signal', 'color': 'green', 'fill_alpha': 0.3}
        ]
        for data in plot_data:
            ax.step(data['x'], data['y'], where='mid', label=data.get('label'), color=data.get('color'), lw=0.1, zorder=1)
            if 'fill_alpha' in data:
                ax.fill_between(data['x'], data['y'], step="mid", color=data.get('color'), alpha=data.get('fill_alpha'))
        ax.set_title(base_title, fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel(f"Counts per {bin_width*1000:.1f} ms Bin", fontsize=12)
        ax.set_xlim(t_start, t_stop)
        ax.set_ylim(bottom=0)

        # --- Overlay MVT data ---
        if mvt_summary_df is not None and not mvt_summary_df.empty:
            ax2 = ax.twinx()
            filtered_mvt_df = mvt_summary_df[mvt_summary_df['median_mvt_ms'] > 0].copy()
            x_err = mvt_window_size_s / 2.0
            filtered_mvt_df['success_percent'] = (filtered_mvt_df['successful_runs'] / filtered_mvt_df['total_runs_at_step']) * 100

            if not filtered_mvt_df.empty:
                norm = mcolors.Normalize(vmin=0, vmax=100)
                
                # ***** THIS IS THE LINE TO EXPERIMENT WITH *****
                cmap = cm.get_cmap('binary') # Try 'plasma', 'cividis', 'gray_r', etc.'OrRd'
                #mcmap = cm.get_cmap('binary') #  cm.get_cmap('OrRd')

                for index, row in filtered_mvt_df.iterrows():
                    if index == 0:
                        x_err = row['center_time_s'] - row['start_time_s']
                    color = cmap(norm(row['success_percent']))
                    #mcolor = mcmap(norm(row['success_percent']))

                    ax2.errorbar(x=row['center_time_s'], y=row['median_mvt_ms'],
                                 yerr=[[row['mvt_err_lower_ms']], [row['mvt_err_upper_ms']]], xerr=x_err,
                                 fmt='o', color=color, capsize=4, markersize=6, lw=1.5, zorder=10)

                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(mappable, ax=ax2, pad=0.08, aspect=30)
                cbar.set_label(f'Success Rate [{filtered_mvt_df["total_runs_at_step"][1]}] (%)', fontsize=12)

            ax2.set_ylabel("MVT (ms)", fontsize=12)
            ax2.tick_params(axis='y')
            ax2.set_yscale("log")
            ax.legend(loc='upper left')
        else:
            ax.legend(loc='upper right')
        
        fig.tight_layout()
        
        plt.savefig(fig_name, dpi=300)
        #plt.show()
        plt.close(fig)
        
        print(f"Plot saved to: {fig_name}")

    except Exception as e:
        logging.error(f"Failed to generate plot. Error: {e}", exc_info=True)
        pass



def plot_gbm_lc(tte_total, tte_src, tte_bkgd,
                bkgd_cps,
                model_info: Dict,
                output_info: Dict):
    params = model_info['base_params']
    #params = model_info.get('params', {})
    #print_nested_dict(params)
    
    
    en_lo = params.get('en_lo', 8.0)
    en_hi = params.get('en_hi', 900.0)

    #print(params)
    trigger_number = params['trigger_number'] if 'trigger_number' in params else 0
    det = params['det'] if 'det' in params else 'nn'
    angle = params['angle'] if 'angle' in params else 0

    #analysis_settings = model_info['snr_analysis']

    func_to_use = model_info.get('func', None)
    func_par = model_info.get('func_par', {})
    fig_name = output_info['file_path'] / f"LC_{output_info['file_name']}.png"
    

    energy_range_nai = (en_lo, en_hi)

    fine_bw = 0.001
    phaii = tte_total.to_phaii(bin_by_time, fine_bw)

    phaii = tte_total.to_phaii(bin_by_time, fine_bw)
    phii_src = tte_src.to_phaii(bin_by_time, fine_bw)
    phii_bkgd = tte_bkgd.to_phaii(bin_by_time, fine_bw)
    lc_tot = phaii.to_lightcurve(energy_range=energy_range_nai)
    lc_src = phii_src.to_lightcurve(energy_range=energy_range_nai)
    lc_bkgd = phii_bkgd.to_lightcurve(energy_range=energy_range_nai)

    try:
        #lcplot = Lightcurve(data=phaii.to_lightcurve(energy_range=energy_range_nai))
        lcplot = Lightcurve(data=lc_tot)
        lcplot.add_selection(lc_src)
        lcplot.add_selection(lc_bkgd)
        lcplot.selections[1].color = 'pink'
        lcplot.selections[0].color = 'green'
        lcplot.selections[0].alpha = 1
        lcplot.selections[1].alpha = 0.5

        #x_low = func_par[1] - func_par[1]
        #x_high = func_par[1] + func_par[1]
        #plt.xlim(x_low, x_high)
        lcplot.errorbars.hide()


        ######### SNR Calculation #########
    except Exception as e:
        print(f"Error during plotting: {e}")
        lcplot = None

    snr_results_dict = _calculate_multi_timescale_snr(
                total_counts=lc_tot.counts, sim_bin_width=0.001,
                back_avg_cps= bkgd_cps,
                search_timescales=model_info['snr_analysis']
            )
    
    # <<< NEW: Format SNR results for the title >>>
    snr_annotation_parts = []
    for ts, snr_val in snr_results_dict.items():
        label = f"S$_{{{ts[1:]}}}$" # Use LaTeX for subscript
        value = f"{snr_val:.1f}"
        snr_annotation_parts.append(f"{label}={value}")
    SNR_text = "; ".join(snr_annotation_parts)
    if output_info["combine_flag"]:
        base_title = f" LC {output_info['file_info']}"
        det_string = output_info.get('det_string')
        #print("Det in output_info:", det_string)
        final_title = f'Bn{trigger_number}, {det_string}, {angle} det,' + f"{base_title}\n{SNR_text}"
    else:
        base_title = f" LC {output_info['file_name']}"
        final_title = f"{base_title}\n{SNR_text}"

    # <<< NEW: Format model parameters for the annotation box >>>
    if func_to_use is not None:
        annotation_text = _format_params_for_annotation(func_to_use, func_par)
        if annotation_text:
            props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
            plt.text(0.03, 0.97, annotation_text, transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left', bbox=props)

    fig_name = output_info['file_path'] / f"LC_{output_info['file_name']}.png"

    if fig_name is None:
        fig_name = f'lc_{trigger_number}_n{det}_{angle}deg.png'

    plt.title(final_title, fontsize=10)

    plt.savefig(fig_name, dpi=300)
    plt.close()



def create_final_gbm_plot(
                src_event_list,
                back_event_list,
                model_info: Dict,
                output_info: Dict):
    params = model_info['base_params']
    #params = model_info.get('params', {})
    #print_nested_dict(params)

    #fig_name = params.get('fig_name', None)
    t_start = params['t_start'] if 't_start' in params else -5.0
    t_stop = params['t_stop'] if 't_stop' in params else 5.0
    trange = [t_start, t_stop]


    for src_event_file, back_event_file in zip(src_event_list, back_event_list):
        #print(type(src_event_file), type(back_event_file))
        # Open the files
        tte_src = GbmTte.open(src_event_file).slice_time(trange)
        tte_bkgd = GbmTte.open(back_event_file).slice_time(trange)

        #tte_src = tte_src_all.slice_time([t_start, t_stop])
        #tte_bkgd = tte_bkgd_all.slice_time([t_start, t_stop])
        total_bkgd_counts = tte_bkgd.data.size
        #print("Tstart:", t_start, "Tstop:", t_stop)
        bkgd_cps = total_bkgd_counts/(t_stop - t_start)
        #print(f"Background counts: {total_bkgd_counts}, Background CPS: {bkgd_cps}")

        # merge the background and source
        tte_total = GbmTte.merge([tte_src, tte_bkgd])

        output_info["file_name"] = src_event_file.stem
        output_info["combine_flag"] = False

        try:
            plot_gbm_lc(tte_total, tte_src, tte_bkgd,
                        bkgd_cps,
                        model_info=model_info,
                        output_info=output_info
                        )
        except Exception as e:
            print(f"Failed to generate representative GBM plot. Error: {e}")
    
    if len(src_event_list) > 1:
        src_tte_list = []
        bkgd_tte_list = []
        for i, src_file_path in enumerate(src_event_list):
            src_tte_list.append(GbmTte.open(src_file_path).slice_time(trange))
            bkgd_tte_list.append(GbmTte.open(back_event_list[i]).slice_time(trange))

        tte_src = GbmTte.merge(src_tte_list)
        tte_bkgd = GbmTte.merge(bkgd_tte_list)
        total_bkgd_counts = tte_bkgd.data.size
        #print("Tstart:", t_start, "Tstop:", t_stop)
        bkgd_cps = total_bkgd_counts/(t_stop - t_start)
        #print(f"Background counts: {total_bkgd_counts}, Background CPS: {bkgd_cps}")

        # merge the background and source
        tte_total = GbmTte.merge([tte_src, tte_bkgd])
        output_info["file_name"] = 'combined_' + src_event_file.stem
        output_info["combine_flag"] = True
        try:
            plot_gbm_lc(tte_total, tte_src, tte_bkgd,
                        bkgd_cps,
                        model_info=model_info,
                        output_info=output_info
                        )
        except Exception as e:
            print(f"Failed to generate representative GBM plot. Error: {e}")


def analysis_mvt_results_to_dataframe(
    mvt_results: List[List[Dict]],
    output_info: Path,
    bin_width_ms: float,
    total_runs: int,
):
    """
    Analyzes a collection of time-resolved MVT results from multiple simulations.

    This function flattens the nested data, groups it by time, calculates
    MVT statistics (median, C.I.) for each time step, generates distribution
    plots for each step, and returns a final summary DataFrame.

    Args:
        mvt_results (List[List[Dict]]): The nested list of MVT results.
        output_info (Dict[str, any]): A dictionary containing output metadata:
            - 'output_path' (Path): The directory to save results.
            - 'param_dir_name' (str): The name of the simulation parameter set.
            - 'selection_str' (str): The detector selection string for file naming.
        bin_width_ms (float): The bin width used in the analysis in milliseconds.
        total_runs (int): The total number of simulation iterations (NN_analysis).

    Returns:
        pd.DataFrame: A DataFrame summarizing the MVT statistics for each time step.
    """
    output_path = output_info['file_path']
    try:
        name = output_info['file_info']
    except:
        name = output_info['trigger_number']
    
    selection_str = output_info['selection_str']

    detailed_df = pd.DataFrame(mvt_results)
    detailed_df.to_csv(output_path / f"Detailed_{name}_{selection_str}s_{bin_width_ms}ms.csv", index=False)

    # Loop through the DataFrame grouped by the analysis bin width
    valid_mvts = detailed_df[detailed_df['mvt_err_ms'] > 0]
    #print(len(valid_runs), "valid runs out of", NN, "for", param_dir.name, "at bin width", bin_width, "ms")
            # All runs where MVT produced a positive timescale
    all_dist_flag = True
    try:
        all_positive_runs = detailed_df[(detailed_df['mvt_ms'] > 0) & (detailed_df['mvt_ms'] < 1e5)]
        all_p16, all_median_mvt, all_p84 = np.percentile(all_positive_runs['mvt_ms'], [16, 50, 84])
    except:
        all_dist_flag = False
        all_p16, all_median_mvt, all_p84 = (-100, -100, -100)

    valid_flag = False
    if len(valid_mvts) >= 2:
        valid_flag = True
        # Statistics are calculated ONLY on the valid runs
        p16, median_mvt, p84 = np.percentile(valid_mvts['mvt_ms'], [16, 50, 84])
        # Use the 68% confidence interval width as a robust measure of "sigma"
        ci_width = p84 - p16
        # Set plot limits to be wide enough to see the distribution, but not the extreme outliers
        data_min = max(0, p16 - 3 * ci_width)
        data_max = p84 + 10 * ci_width #
        #data_max = np.percentile(all_positive_runs['mvt_ms'], 99.5) if not all_positive_runs.empty else p84 + 3 * ci_width
    else:
        p16, median_mvt, p84 = (-100, -100, -100)

    fig, ax = plt.subplots(figsize=(10, 6))
        # --- Create the Enhanced MVT Distribution Plot ---
    if all_dist_flag:
        try:
            # <<< 2. Plot the background histogram of ALL non-failed runs in gray >>>
            if not all_positive_runs.empty and all_dist_flag:
                ax.hist(all_positive_runs['mvt_ms'], bins=30, density=True, 
                            label=f'All Runs w/ MVT > 0 ({len(all_positive_runs)}/{total_runs})',
                            color='gray', alpha=0.5, histtype='stepfilled', edgecolor='none', zorder=1)
                # Overlay the statistics from the valid runs
                ax.axvline(all_median_mvt, color='k', linestyle='-', lw=1.0,
                        label=f"Median = {all_median_mvt:.3f} ms")
                #ax.axvspan(all_p16, all_p84, color='k', alpha=0.1, hatch='///',
            #label=f"68% C.I. [{all_p16:.3f}, {all_p84:.3f}]")

                ax.axvspan(all_p16, all_p84, color='gray', alpha=0.1,
                        label=f"68% C.I. [{all_p16:.3f}, {all_p84:.3f}]")
        except Exception as e:
            logging.error(f"Error creating background MVT distribution plot for {name} at bin width {bin_width_ms}ms: {e}")
            all_dist_flag = False

    # <<< 3. Plot the main histogram of VALID runs (err > 0) on top >>>
    if valid_flag:
        try:
            ax.hist(valid_mvts['mvt_ms'], bins=30, density=True, 
                    label=f'Valid Runs w/ Err > 0 ({len(valid_mvts)}/{total_runs})',
                    color='steelblue', histtype='stepfilled', edgecolor='black', zorder=2) 

            # Overlay the statistics from the valid runs
            ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2.5,
                    label=f"Median = {median_mvt:.3f} ms")
            #ax.axvspan(p16, p84, color='darkorange', alpha=0.3,
            #        label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")
            ax.axvline(p16, color='orange', linestyle='--', lw=1)
            ax.axvline(p84, color='orange', linestyle='--', lw=1)
            ax.axvspan(p16, p84, color='darkorange', alpha=0.1, hatch='///',
                    label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")
                
            auto_min, auto_max = ax.get_xlim()
            final_min = max(auto_min, data_min)
            final_max = min(auto_max, data_max)

            # Formatting
            ax.set_xlim(final_min, final_max)
        except Exception as e:
            logging.error(f"Error creating MVT distribution plot for {name} at bin width {bin_width_ms}ms: {e}")
    
    if valid_flag or all_dist_flag:
        ax.set_ylim(bottom=0)
        ax.set_title(f"MVT: {name} {selection_str}s\nBin Width: {bin_width_ms} ms", fontsize=12)
        ax.set_xlabel("Minimum Variability Timescale (ms)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        fig.tight_layout()
        plt.savefig(output_path / f"MVT_dis_{name}_{selection_str}s_{round(bin_width_ms, 2)}ms.png", dpi=300)
    plt.close(fig)
 


    MVT_summary = {
        'median_mvt_ms': round(median_mvt, 4),
        'mvt_err_lower': round(median_mvt - p16, 4),
        'mvt_err_upper': round(p84 - median_mvt, 4),
        'all_median_mvt_ms': round(all_median_mvt, 4),
        'all_mvt_err_lower': round(all_median_mvt - all_p16, 4),
        'all_mvt_err_upper': round(all_p84 - all_median_mvt, 4),
        'successful_runs': len(valid_mvts),
        'total_sim': total_runs,
        'failed_runs': len(detailed_df) - len(valid_mvts),
    }

    snr_keys = {}

    if valid_flag:
        for col in valid_mvts.columns:
            if col.startswith(('S_flu', 'S1', 's2', 'S3', 'S6', 'bkgd_counts', 'src_counts', 'back_avg_cps')):
                new_key = f'mean_{col}' if 'counts' in col or 'cps' in col else col
                snr_keys[new_key] = round(valid_mvts[col].mean(), 2)
    else:
        for col in all_positive_runs.columns:
            if col.startswith(('S_flu', 'S1', 's2', 'S3', 'S6', 'bkgd_counts', 'src_counts', 'back_avg_cps')):
                new_key = f'mean_{col}' if 'counts' in col or 'cps' in col else col
                snr_keys[new_key] = round(all_positive_runs[col].mean(), 2)

    #print(MVT_summary)
    #final_MVT_csv_path = output_path / f"MVT_{trigger_number}_default_{bin_width_ms}ms.csv"
    #MVT_summary_df = pd.DataFrame(MVT_summary)
    #MVT_summary_df.to_csv(final_MVT_csv_path, index=False)

    return MVT_summary, snr_keys


def create_final_result_dict(input_info: Dict,
                             MVT_summary: Dict,
                             snr_keys: dict,
                             snr_MVT: float,
                             snr_mvt_position: float) -> Dict:

    base_params = input_info['base_params']
    sim_params_file = input_info['sim_par_file']

    if type(sim_params_file) is dict:
        sim_params = sim_params_file
    else:
        sim_params = yaml.safe_load(open(sim_params_file, 'r'))
    bin_width = input_info['bin_width_ms']
    base_dets = input_info['base_det']
    analysis_det = input_info['analysis_det']
    analysis_settings = input_info.get('analysis_settings', {})

    # <<< Define standard keys and defaults from your original script >>>
    ALL_PULSE_PARAMS = ['sigma', 'center_time', 'width', 'peak_time_ratio', 'start_time', 'rise_time', 'decay_time']

    STANDARD_KEYS = [
        # Core Parameters
        'sim_type', 'pulse_shape', 'bin_width_ms', 'peak_amplitude', 'position', 'angle', 'trigger',
        'sim_det', 'base_det', 'analysis_det', 'num_analysis_det', 'background_level',
        # Run Summary
        'total_sim', 'successful_runs', 'failed_runs',
        # MVT Stats
        'median_mvt_ms', 'mvt_err_lower', 'mvt_err_upper', 'all_median_mvt_ms', 'all_mvt_err_lower', 'all_mvt_err_upper',
        # Mean Counts (from any analysis type)
        'mean_src_counts', 'mean_bkgd_counts', 'mean_src_counts_total', 'mean_src_counts_template', 
        'mean_src_counts_feature', 'mean_bkgd_counts_feature_local', 'mean_back_avg_cps', 
        # Fluence SNR (from any analysis type)
        'MVT_snr', 'MVT_snr_position',
        'S_flu', 'S_flu_total', 'S_flu_template', 'S_flu_feature_avg', 'S_flu_feature_local',
        # Multi-timescale SNR (Total Pulse - simple runs will use this)
        'S16', 'S32', 'S64', 'S128',
        # Multi-timescale SNR (Complex Pulse Breakdown)
        'S16_total', 'S32_total', 'S64_total', 'S128_total',
        'S16_template', 'S32_template', 'S64_template', 'S128_template', 
        'S16_feature', 'S32_feature', 'S64_feature', 'S128_feature', 
    ]


    # ==================== START CHANGE 1 ====================
    # If it's a complex pulse, dynamically add keys for the feature pulse parameters.
    # This ensures they have a column in the final summary CSV.
    if base_params['pulse_shape'] == 'complex_pulse':
        extra_pulse_config = analysis_settings.get('extra_pulse', {})
        # Create descriptive keys like 'sigma_feature', 'peak_amplitude_feature', etc.
        feature_param_keys = [f"{key}_feature" for key in extra_pulse_config if key != 'pulse_shape']
        STANDARD_KEYS.extend(feature_param_keys)
    # ===================== END CHANGE 1 =====================

    result_data = {**base_params, **MVT_summary,
                    'MVT_snr': snr_MVT,
                    'MVT_snr_position': snr_mvt_position,
                    'bin_width_ms': bin_width,
                    'sim_det': sim_params['det'],
                    'base_det': base_dets,
                    'analysis_det': analysis_det,
                    'num_analysis_det': len(analysis_det) if isinstance(analysis_det, list) else 1,
                    'trigger': sim_params.get('trigger_number', 9999999),
                    'angle': sim_params.get('angle', 0),
                    'background_level': sim_params.get('background_level'),
                    'position': base_params.get('position', 0),
                    **snr_keys
                    }
    
    if result_data.get('pulse_shape') == 'complex_pulse':
            extra_pulse_config = analysis_settings.get('extra_pulse', {})
            feature_params_for_summary = {f"{key}_feature": val for key, val in extra_pulse_config.items() if key != 'pulse_shape'}
            result_data.update(feature_params_for_summary)


 
    final_dict = {}
    for key in STANDARD_KEYS:
        final_dict[key] = result_data.get(key, DEFAULT_PARAM_VALUE)
    
    # Add all possible pulse parameter keys, using the default value if a key is not in this run's result_data
    for key in ALL_PULSE_PARAMS:
        final_dict[key] = result_data.get(key, DEFAULT_PARAM_VALUE)
    
    return final_dict


def compute_snr_for_mvt(input_info: Dict,
                           output_info: Dict,
                           mvt_ms: float,
                           position = None):
    
    output_path = output_info['file_path']
    try:
        name = output_info['file_info']
    except:
        name = output_info['trigger_number']
    
    selection_str = output_info['selection_str']

    sim_data_path = input_info['sim_data_path']
    analysis_settings = input_info['analysis_settings']
    src_event_files = sorted(sim_data_path.glob('*_src.npz'))
    back_event_files = sorted(sim_data_path.glob('*_bkgd.npz'))
    if not src_event_files or not back_event_files:
        logging.error(f"No source or background event files found in {sim_data_path}")

    sim_params_file = input_info['sim_par_file']

    if type(sim_params_file) is dict:
        sim_params = sim_params_file
    else:
        sim_params = yaml.safe_load(open(sim_params_file, 'r'))

    data_src = np.load(src_event_files[0], allow_pickle=True)
    data_back = np.load(back_event_files[0], allow_pickle=True)
    #sim_params = data_src['params'].item()
    
    background_level = sim_params['background_level']
    scale_factor = sim_params['scale_factor']
  

    t_start = sim_params['t_start']
    t_stop = sim_params['t_stop']
    det = sim_params.get('det', 'nn')
    angle = sim_params.get('angle', 0)
   
    base_params = input_info['base_params']
    NN_analysis = base_params['num_analysis']

    source_event_realizations_all = data_src['realizations']
    background_event_realizations = data_back['realizations']
    #sim_params = data_src['params'].item()
    
    duration = sim_params['t_stop'] - sim_params['t_start']

    iteration_results = []
    iteration_position_results = []    

    NN = len(source_event_realizations_all)
    NN_back = len(background_event_realizations)
    if NN != NN_back:
        logging.warning(f"Mismatch in realization counts: {NN} (source) vs {NN_back} (background)")
        return []
    if NN < NN_analysis:
        logging.warning(f"Insufficient realizations: {NN} (source) < {NN_analysis} (required)")
        return []
    
    if NN_analysis < NN:
        source_event_realizations = source_event_realizations_all[:NN_analysis]
        NN = NN_analysis
    else:
        source_event_realizations = source_event_realizations_all

    for i, source_events in enumerate(source_event_realizations):
        try:
            background_events = background_event_realizations[i]
            total_events = np.sort(np.concatenate([source_events, background_events]))

            total_bkgd_counts = len(background_events)

            background_level_cps = total_bkgd_counts / duration
            bin_width_s = mvt_ms / 1000.0
            background_counts = background_level_cps * bin_width_s

            # Loop through analysis bin width
            
            bins = np.arange(sim_params['t_start'], sim_params['t_stop'] + bin_width_s, bin_width_s)
            counts, _ = np.histogram(total_events, bins=bins)
            SNR_MVT = max(counts) / np.sqrt(background_counts) if background_counts > 0 else DEFAULT_PARAM_VALUE 

            if position is not None:
                try:
                    # First, check if the requested position is within the valid time range of the bins
                    if sim_params['t_start'] <= position < sim_params['t_stop']:
                        
                        # np.digitize finds the index of the bin that 'position' belongs to.
                        # We subtract 1 because digitize gives an index for the 'bins' array (N+1 edges),
                        # and we need the corresponding index for the 'counts' array (N values).
                        bin_index = np.digitize(position, bins) - 1

                        # A final safety check to ensure the index is valid for the counts array
                        if 0 <= bin_index < len(counts):
                            count_at_position = counts[bin_index]
                            
                            # Now, calculate SNR using the count from that specific bin
                            SNR_MVT_position = count_at_position / np.sqrt(background_counts) if background_counts > 0 else DEFAULT_PARAM_VALUE 
                        else:
                            # This handles edge cases where the position might be exactly t_stop
                            SNR_MVT_position = DEFAULT_PARAM_VALUE
                    else:
                        # If the position is outside the light curve's range, SNR is 0
                        SNR_MVT_position = DEFAULT_PARAM_VALUE
                        logging.warning(f"Requested position {position} is outside the analysis time range.")

                except Exception as e:
                    SNR_MVT_position = DEFAULT_PARAM_VALUE
                    logging.error(f"Error finding count at position {position}: {e}", exc_info=True)
            else:
                SNR_MVT_position = DEFAULT_PARAM_VALUE 

            iteration_position_results.append((SNR_MVT_position))
            iteration_results.append(SNR_MVT)
        except Exception as e:
            logging.error(f"Error computing SNR for MVT in iteration {i}: {e}", exc_info=True)
            continue
    try:
        mvt_snr_output = output_path / f"SNR_MVT_{name}_{selection_str}_{round(mvt_ms, 2)}.csv"
        pd.DataFrame({'SNR_MVT': iteration_results, 'SNR_MVT_position': iteration_position_results}).to_csv(mvt_snr_output, index=False)
    except Exception as e:
        logging.error(f"Error saving SNR_MVT results to CSV: {e}", exc_info=True)
    if iteration_results:
        iteration_results_filtered = [snr for snr in iteration_results if snr > 0]
        mean_snr_mvt = round(np.mean(iteration_results_filtered), 2)
    else:
        mean_snr_mvt = DEFAULT_PARAM_VALUE 

    if position is not None and iteration_position_results:
        iteration_position_results_filtered = [snr for snr in iteration_position_results if snr > 0]
        mean_snr_mvt_position = round(np.mean(iteration_position_results_filtered), 2)
    else:
        mean_snr_mvt_position = DEFAULT_PARAM_VALUE
    
    return mean_snr_mvt, mean_snr_mvt_position



def Function_MVT_analysis(input_info: Dict,
                           output_info: Dict):
    #params = input_info['base_params']

    sim_data_path = input_info['sim_data_path']
    haar_python_path = input_info['haar_python_path']
    analysis_settings = input_info['analysis_settings']
    src_event_files = sorted(sim_data_path.glob('*_src.npz'))
    back_event_files = sorted(sim_data_path.glob('*_bkgd.npz'))
    if not src_event_files or not back_event_files:
        logging.error(f"No source or background event files found in {sim_data_path}")

    sim_params_file = input_info['sim_par_file']

    if type(sim_params_file) is dict:
        sim_params = sim_params_file
    else:
        sim_params = yaml.safe_load(open(sim_params_file, 'r'))

    data_src = np.load(src_event_files[0], allow_pickle=True)
    data_back = np.load(back_event_files[0], allow_pickle=True)
    #sim_params = data_src['params'].item()
    
    background_level = sim_params['background_level']
    scale_factor = sim_params['scale_factor']
  

    t_start = sim_params['t_start']
    t_stop = sim_params['t_stop']
    det = sim_params.get('det', 'nn')
    angle = sim_params.get('angle', 0)
   
    base_params = input_info['base_params']
    NN_analysis = base_params['num_analysis']
    snr_timescales = analysis_settings.get('snr_timescales', [0.010, 0.016, 0.032, 0.064, 0.128])
    bin_width_ms = input_info['bin_width_ms']


    source_event_realizations_all = data_src['realizations']
    background_event_realizations = data_back['realizations']
    #sim_params = data_src['params'].item()
    
    #background_level = sim_params['background_level']
    #scale_factor = sim_params['scale_factor']

    #background_level_cps = background_level * scale_factor
    src_start, src_stop = calculate_src_interval(sim_params)
    src_duration = src_stop - src_start

    duration = sim_params['t_stop'] - sim_params['t_start']

    iteration_results = []
    

    NN = len(source_event_realizations_all)
    NN_back = len(background_event_realizations)
    if NN != NN_back:
        logging.warning(f"Mismatch in realization counts: {NN} (source) vs {NN_back} (background)")
        return []
    if NN < NN_analysis:
        logging.warning(f"Insufficient realizations: {NN} (source) < {NN_analysis} (required)")
        return []
    
    if NN_analysis < NN:
        source_event_realizations = source_event_realizations_all[:NN_analysis]
        NN = NN_analysis
    else:
        source_event_realizations = source_event_realizations_all

    for i, source_events in enumerate(source_event_realizations):
        try:
            background_events = background_event_realizations[i]
            total_events = np.sort(np.concatenate([source_events, background_events]))
            iteration_seed = sim_params['random_seed'] + i

            total_src_counts = len(source_events)
            total_bkgd_counts = len(background_events)

            background_level_cps = total_bkgd_counts / duration
            background_counts = background_level_cps * src_duration
            try:
                snr_fluence = total_src_counts / np.sqrt(background_counts)
            except ZeroDivisionError:
                snr_fluence = 0
            #snr_fluence = total_src_counts / sigma_bkgd_counts
            # Calculate per-realization metrics that are independent of bin width
            total_counts_fine, _ = np.histogram(total_events, bins=int(duration / 0.001))
            snr_dict = _calculate_multi_timescale_snr(
                total_counts=total_counts_fine, sim_bin_width=0.001,
                back_avg_cps=background_level_cps,
                search_timescales=snr_timescales
            )

            base_iter_detail = {
                'iteration': i + 1,
                'random_seed': iteration_seed,
                'back_avg_cps': round(background_level_cps, 2),
                'bkgd_counts': int(background_counts),
                'src_counts': total_src_counts,
                'S_flu': round(snr_fluence, 2),
                **snr_dict,
            }

            if i == 1:
                create_final_plot(source_events=source_events,
                                  background_events=background_events,
                                    model_info={
                                        'func': None,
                                        'func_par': None,
                                        'base_params': sim_params,
                                        'snr_analysis': snr_timescales
                                    },
                                    output_info= output_info
                                )

            # Loop through analysis bin width
            bin_width_s = bin_width_ms / 1000.0
            bins = np.arange(sim_params['t_start'], sim_params['t_stop'] + bin_width_s, bin_width_s)
            counts, _ = np.histogram(total_events, bins=bins)

            mvt_res = run_mvt_in_subprocess(
                            counts=counts,
                            bin_width_s=bin_width_s,
                            haar_python_path=haar_python_path
                        )
            plt.close('all')
            mvt_val = mvt_res['mvt_ms']
            mvt_err = mvt_res['mvt_err_ms']


            iter_detail = {**base_iter_detail,
                            'analysis_bin_width_ms': bin_width_ms,
                            'mvt_ms': round(mvt_val, 4),
                            'mvt_err_ms': round(mvt_err, 4),
                            **base_params}
            iteration_results.append(iter_detail)

        except Exception as e:
            logging.warning(f"Failed analysis on realization {i} in {src_event_files[0].name}. Error: {e}")
            iteration_results.append({'iteration': i + 1,
                                            'random_seed': sim_params['random_seed'] + i,
                                            'analysis_bin_width_ms': bin_width_ms,
                                            'mvt_ms': DEFAULT_PARAM_VALUE,
                                            'mvt_err_ms': DEFAULT_PARAM_VALUE,
                                            'back_avg_cps': DEFAULT_PARAM_VALUE,
                                            'bkgd_counts': DEFAULT_PARAM_VALUE,
                                            'src_counts': DEFAULT_PARAM_VALUE,
                                            'S_flu': DEFAULT_PARAM_VALUE,
                                            **base_params,
                                            **snr_dict})
    final_summary_list = []
    MVT_summary, snr_keys = analysis_mvt_results_to_dataframe(
        mvt_results=iteration_results,
        output_info=output_info,
        bin_width_ms=bin_width_ms,
        total_runs=NN
    )

    mvt_ms = MVT_summary['median_mvt_ms']
    try:
        if mvt_ms > 0:
            snr_MVT, snr_mvt_position = compute_snr_for_mvt(input_info=input_info,
                                output_info=output_info,
                                mvt_ms=mvt_ms)
        else:
            snr_MVT = DEFAULT_PARAM_VALUE
            snr_mvt_position = DEFAULT_PARAM_VALUE
    except Exception as e:
        logging.error(f"Error computing SNR at MVT timescale: {e}")
        snr_MVT = DEFAULT_PARAM_VALUE
        snr_mvt_position = DEFAULT_PARAM_VALUE

    final_result = create_final_result_dict(input_info,
                             MVT_summary,
                             snr_keys,
                             snr_MVT=snr_MVT,
                             snr_mvt_position=snr_mvt_position)
    final_summary_list.append(final_result)
    return final_summary_list



def Function_MVT_analysis_complex(input_info: Dict, output_info: Dict):
    """
    Performs MVT analysis by assembling a pre-generated template and feature,
    and calculates a detailed breakdown of source counts and SNR for each component.
    """
    base_params = input_info['base_params']
    analysis_settings = input_info['analysis_settings']
    sim_params = input_info['sim_par_file']
    template_dir_path = input_info['sim_data_path']
    data_path_root = template_dir_path.parents[2]
    bin_width_ms = input_info['bin_width_ms']
    haar_python_path = input_info['haar_python_path']
    feature_amplitude = base_params['peak_amplitude']
    position_shift = base_params['position']
    extra_pulse_config = analysis_settings.get('extra_pulse')
    feature_shape = extra_pulse_config.get('pulse_shape')
    feature_params_for_naming = { 'peak_amplitude': feature_amplitude, **{k: v for k, v in extra_pulse_config.items() if k != 'pulse_shape'} }
    feature_dir_name = _create_param_directory_name('function', feature_shape, feature_params_for_naming)
    feature_dir_path = data_path_root / 'function' / feature_shape / feature_dir_name

    # --- 3. Load event data (Unchanged) ---
    try:
        template_src_data = np.load(list(template_dir_path.glob('*_src.npz'))[0], allow_pickle=True)
        template_bkgd_data = np.load(list(template_dir_path.glob('*_bkgd.npz'))[0], allow_pickle=True)
        feature_src_data = np.load(list(feature_dir_path.glob('*_src.npz'))[0], allow_pickle=True)
        template_realizations = template_src_data['realizations']
        background_realizations = template_bkgd_data['realizations']
        feature_realizations = feature_src_data['realizations']
    except (FileNotFoundError, IndexError) as e:
        logging.error(f"Error loading source/background files for assembly: {e}")
        return [], 0

    # --- 4. Main Analysis Loop ---
    NN_analysis = base_params.get('num_analysis', len(template_realizations))
    duration = sim_params['t_stop'] - sim_params['t_start']
    snr_timescales = analysis_settings.get('snr_timescales', [])
    iteration_results = []
    
    for i in range(NN_analysis):
        try:
            # --- 5. Assemble the pulse for this realization ---
            template_events = template_realizations[i]
            feature_events = feature_realizations[i]
            background_events = background_realizations[i]
            shifted_feature_events = feature_events + position_shift
            complete_src_events = np.sort(np.concatenate([template_events, shifted_feature_events]))
            total_events = np.sort(np.concatenate([complete_src_events, background_events]))
            
            background_level_cps = len(background_events) / duration

            ## ================== FULL METRIC CALCULATIONS ==================
            # 1. Fluence SNR for each component
            src_counts_total = len(complete_src_events)
            bkgd_counts_total_window = background_level_cps * duration
            S_flu_total = src_counts_total / np.sqrt(bkgd_counts_total_window) if bkgd_counts_total_window > 0 else 0
            src_counts_template = len(template_events)
            S_flu_template = src_counts_template / np.sqrt(bkgd_counts_total_window) if bkgd_counts_total_window > 0 else 0
            src_counts_feature = len(shifted_feature_events)
            
            # --- 1a. SNR of Feature (using AVERAGE background+template rate) ---
            combined_bkgd_avg_cps = (len(background_events) + len(template_events)) / duration
            feature_params_for_interval = {**feature_params_for_naming, 'pulse_shape': feature_shape}
            feat_t_start_relative, feat_t_stop_relative = calculate_src_interval(feature_params_for_interval)
            feature_duration = feat_t_stop_relative - feat_t_start_relative
            bkgd_in_feature_window_avg = combined_bkgd_avg_cps * feature_duration
            S_flu_feature_avg = src_counts_feature / np.sqrt(bkgd_in_feature_window_avg) if bkgd_in_feature_window_avg > 0 else 0

            # --- 1b. SNR of Feature (using LOCAL background+template counts) ---
            feature_t_start = feat_t_start_relative + position_shift
            feature_t_stop = feat_t_stop_relative + position_shift
            template_counts_in_window = np.sum((template_events >= feature_t_start) & (template_events <= feature_t_stop))
            bkgd_counts_in_window = np.sum((background_events >= feature_t_start) & (background_events <= feature_t_stop))
            bkgd_counts_feature_local = template_counts_in_window + bkgd_counts_in_window
            S_flu_feature_local = src_counts_feature / np.sqrt(bkgd_counts_feature_local) if bkgd_counts_feature_local > 0 else 0
            
            # 2. Multi-timescale SNR for each component
            # --- 2a. Multi-timescale SNR for TOTAL pulse ---
            total_counts_fine, _ = np.histogram(total_events, bins=int(duration / 0.001))
            snr_dict_total = _calculate_multi_timescale_snr(total_counts=total_counts_fine, sim_bin_width=0.001, back_avg_cps=background_level_cps, search_timescales=snr_timescales)
            
            # --- 2b. Multi-timescale SNR for TEMPLATE pulse ---
            template_plus_bkgd_events = np.sort(np.concatenate([template_events, background_events]))
            template_counts_fine, _ = np.histogram(template_plus_bkgd_events, bins=int(duration / 0.001))
            snr_dict_template = _calculate_multi_timescale_snr(total_counts=template_counts_fine, sim_bin_width=0.001, back_avg_cps=background_level_cps, search_timescales=snr_timescales)

            # --- 2c. Multi-timescale SNR for FEATURE pulse ---
            # For this, the "signal" is the feature, and the "background" is the template+sky.
            # We use total_counts_fine (which includes all 3 components) and the combined average background rate.
            snr_dict_feature = _calculate_multi_timescale_snr(total_counts=total_counts_fine, sim_bin_width=0.001, back_avg_cps=combined_bkgd_avg_cps, search_timescales=snr_timescales)
            
            ## =============================================================
            
            # Rename SNR dictionary keys to be unique before merging
            final_snr_dict = {f'{k}_total': v for k, v in snr_dict_total.items()}
            final_snr_dict.update({f'{k}_template': v for k, v in snr_dict_template.items()})
            final_snr_dict.update({f'{k}_feature': v for k, v in snr_dict_feature.items()})

            base_iter_detail = {
                'iteration': i + 1, 'random_seed': sim_params['random_seed'] + i, 'back_avg_cps': round(background_level_cps, 2),
                'src_counts_total': src_counts_total, 'bkgd_counts': len(background_events),
                'src_counts_template': src_counts_template, 'src_counts_feature': src_counts_feature,
                'bkgd_counts_feature_local': bkgd_counts_feature_local, # <-- New Metric
                'S_flu_total': round(S_flu_total, 2), 'S_flu_template': round(S_flu_template, 2),
                'S_flu_feature_avg': round(S_flu_feature_avg, 2), 'S_flu_feature_local': round(S_flu_feature_local, 2),
                **final_snr_dict
            }

            
            if i == 1:
                create_final_plot(source_events=complete_src_events, background_events=background_events, model_info={ 'func': None, 'func_par': None, 'base_params': sim_params, 'snr_analysis': snr_timescales }, output_info=output_info)

            # --- MVT on the Assembled Pulse ---
            bin_width_s = bin_width_ms / 1000.0
            bins = np.arange(sim_params['t_start'], sim_params['t_stop'] + bin_width_s, bin_width_s)
            counts, _ = np.histogram(total_events, bins=bins)
            mvt_res = run_mvt_in_subprocess(counts=counts, bin_width_s=bin_width_s, haar_python_path=haar_python_path)

            mvt_val = mvt_res['mvt_ms'] if mvt_res else DEFAULT_PARAM_VALUE
            mvt_err = mvt_res['mvt_err_ms'] if mvt_res else DEFAULT_PARAM_VALUE

            iteration_results.append({ **base_iter_detail, 'analysis_bin_width_ms': bin_width_ms, 'mvt_ms': round(mvt_val, 4), 'mvt_err_ms': round(mvt_err, 4), **base_params })

        except Exception as e:
            logging.warning(f"Failed analysis on realization {i}. Error: {e}")
            iteration_results.append({'iteration': i + 1, 'mvt_ms': DEFAULT_PARAM_VALUE, 'mvt_err_ms': DEFAULT_PARAM_VALUE, **base_params})

    final_summary_list = []
    MVT_summary, snr_keys = analysis_mvt_results_to_dataframe(
        mvt_results=iteration_results,
        output_info=output_info,
        bin_width_ms=bin_width_ms,
        total_runs=NN_analysis
    )

    mvt_ms = MVT_summary['median_mvt_ms']
    try:
        snr_MVT, snr_mvt_position = compute_snr_for_mvt(input_info=input_info,
                               output_info=output_info,
                               mvt_ms=mvt_ms,
                               position=position_shift)
    except Exception as e:
        logging.error(f"Error computing SNR at MVT timescale: {e}")
        snr_MVT = DEFAULT_PARAM_VALUE
        snr_mvt_position = DEFAULT_PARAM_VALUE

    final_result = create_final_result_dict(input_info,
                             MVT_summary,
                             snr_keys,
                             snr_MVT=snr_MVT,
                             snr_mvt_position=snr_mvt_position)
    final_summary_list.append(final_result)

    return final_summary_list






def Function_MVT_analysis_complex_time_resolved(input_info: Dict, output_info: Dict):
    """
    Performs MVT analysis by assembling a pre-generated template and feature,
    and calculates a detailed breakdown of source counts and SNR for each component.
    """
    base_params = input_info['base_params']
    analysis_settings = input_info['analysis_settings']
    sim_params = input_info['sim_par_file']
    template_dir_path = input_info['sim_data_path']
    data_path_root = template_dir_path.parents[2]
    bin_width_ms = input_info['bin_width_ms']
    haar_python_path = input_info['haar_python_path']
    feature_amplitude = base_params['peak_amplitude']
    position_shift = base_params['position']
    extra_pulse_config = analysis_settings.get('extra_pulse')
    feature_shape = extra_pulse_config.get('pulse_shape')
    feature_params_for_naming = { 'peak_amplitude': feature_amplitude, **{k: v for k, v in extra_pulse_config.items() if k != 'pulse_shape'} }
    feature_dir_name = _create_param_directory_name('function', feature_shape, feature_params_for_naming)
    feature_dir_path = data_path_root / 'function' / feature_shape / feature_dir_name

    # --- 2. Extract MVT time window and step size from analysis settings ---
    mvt_time_window = analysis_settings.get('mvt_time_window', 1.0)
    mvt_step_size = analysis_settings.get('mvt_step_size', 0.5)

    # --- 3. Load event data (Unchanged) ---
    try:
        template_src_data = np.load(list(template_dir_path.glob('*_src.npz'))[0], allow_pickle=True)
        template_bkgd_data = np.load(list(template_dir_path.glob('*_bkgd.npz'))[0], allow_pickle=True)
        feature_src_data = np.load(list(feature_dir_path.glob('*_src.npz'))[0], allow_pickle=True)
        template_realizations = template_src_data['realizations']
        background_realizations = template_bkgd_data['realizations']
        feature_realizations = feature_src_data['realizations']
    except (FileNotFoundError, IndexError) as e:
        logging.error(f"Error loading source/background files for assembly: {e}")
        return [], 0

    # --- 4. Main Analysis Loop ---
    NN_analysis = base_params.get('num_analysis', len(template_realizations))
    duration = sim_params['t_stop'] - sim_params['t_start']
    snr_timescales = analysis_settings.get('snr_timescales', [])
    iteration_results = []
    MVT_time_resolved_results = []
    first_realization_events = {}
    
    for i in range(NN_analysis):
        try:
            # --- 5. Assemble the pulse for this realization ---
            template_events = template_realizations[i]
            feature_events = feature_realizations[i]
            background_events = background_realizations[i]
            shifted_feature_events = feature_events + position_shift
            complete_src_events = np.sort(np.concatenate([template_events, shifted_feature_events]))
            total_events = np.sort(np.concatenate([complete_src_events, background_events]))
            # --- NEW: Save the event data from the first run ---
            if i == 0:
                first_realization_events['source'] = complete_src_events
                first_realization_events['background'] = background_events
            
            background_level_cps = len(background_events) / duration

            ## ================== FULL METRIC CALCULATIONS ==================
            # 1. Fluence SNR for each component
            src_counts_total = len(complete_src_events)
            bkgd_counts_total_window = background_level_cps * duration
            S_flu_total = src_counts_total / np.sqrt(bkgd_counts_total_window) if bkgd_counts_total_window > 0 else 0
            src_counts_template = len(template_events)
            S_flu_template = src_counts_template / np.sqrt(bkgd_counts_total_window) if bkgd_counts_total_window > 0 else 0
            src_counts_feature = len(shifted_feature_events)
            
            # --- 1a. SNR of Feature (using AVERAGE background+template rate) ---
            combined_bkgd_avg_cps = (len(background_events) + len(template_events)) / duration
            feature_params_for_interval = {**feature_params_for_naming, 'pulse_shape': feature_shape}
            feat_t_start_relative, feat_t_stop_relative = calculate_src_interval(feature_params_for_interval)
            feature_duration = feat_t_stop_relative - feat_t_start_relative
            bkgd_in_feature_window_avg = combined_bkgd_avg_cps * feature_duration
            S_flu_feature_avg = src_counts_feature / np.sqrt(bkgd_in_feature_window_avg) if bkgd_in_feature_window_avg > 0 else 0

            # --- 1b. SNR of Feature (using LOCAL background+template counts) ---
            feature_t_start = feat_t_start_relative + position_shift
            feature_t_stop = feat_t_stop_relative + position_shift
            template_counts_in_window = np.sum((template_events >= feature_t_start) & (template_events <= feature_t_stop))
            bkgd_counts_in_window = np.sum((background_events >= feature_t_start) & (background_events <= feature_t_stop))
            bkgd_counts_feature_local = template_counts_in_window + bkgd_counts_in_window
            S_flu_feature_local = src_counts_feature / np.sqrt(bkgd_counts_feature_local) if bkgd_counts_feature_local > 0 else 0
            
            # 2. Multi-timescale SNR for each component
            # --- 2a. Multi-timescale SNR for TOTAL pulse ---
            total_counts_fine, _ = np.histogram(total_events, bins=int(duration / 0.001))
            snr_dict_total = _calculate_multi_timescale_snr(total_counts=total_counts_fine, sim_bin_width=0.001, back_avg_cps=background_level_cps, search_timescales=snr_timescales)
            
            # --- 2b. Multi-timescale SNR for TEMPLATE pulse ---
            template_plus_bkgd_events = np.sort(np.concatenate([template_events, background_events]))
            template_counts_fine, _ = np.histogram(template_plus_bkgd_events, bins=int(duration / 0.001))
            snr_dict_template = _calculate_multi_timescale_snr(total_counts=template_counts_fine, sim_bin_width=0.001, back_avg_cps=background_level_cps, search_timescales=snr_timescales)

            # --- 2c. Multi-timescale SNR for FEATURE pulse ---
            # For this, the "signal" is the feature, and the "background" is the template+sky.
            # We use total_counts_fine (which includes all 3 components) and the combined average background rate.
            snr_dict_feature = _calculate_multi_timescale_snr(total_counts=total_counts_fine, sim_bin_width=0.001, back_avg_cps=combined_bkgd_avg_cps, search_timescales=snr_timescales)
            
            ## =============================================================
            
            # Rename SNR dictionary keys to be unique before merging
            final_snr_dict = {f'{k}_total': v for k, v in snr_dict_total.items()}
            final_snr_dict.update({f'{k}_template': v for k, v in snr_dict_template.items()})
            final_snr_dict.update({f'{k}_feature': v for k, v in snr_dict_feature.items()})

            base_iter_detail = {
                'iteration': i + 1, 'random_seed': sim_params['random_seed'] + i, 'back_avg_cps': round(background_level_cps, 2),
                'src_counts_total': src_counts_total, 'bkgd_counts': len(background_events),
                'src_counts_template': src_counts_template, 'src_counts_feature': src_counts_feature,
                'bkgd_counts_feature_local': bkgd_counts_feature_local, # <-- New Metric
                'S_flu_total': round(S_flu_total, 2), 'S_flu_template': round(S_flu_template, 2),
                'S_flu_feature_avg': round(S_flu_feature_avg, 2), 'S_flu_feature_local': round(S_flu_feature_local, 2),
                **final_snr_dict
            }

            
            if i == 1:
                create_final_plot(source_events=complete_src_events, background_events=background_events, model_info={ 'func': None, 'func_par': None, 'base_params': sim_params, 'snr_analysis': snr_timescales }, output_info=output_info)

            # --- MVT on the Assembled Pulse ---
            bin_width_s = bin_width_ms / 1000.0
            bins = np.arange(sim_params['t_start'], sim_params['t_stop'] + bin_width_s, bin_width_s)
            counts, _ = np.histogram(total_events, bins=bins)
            mvt_res = run_mvt_in_subprocess(counts=counts, bin_width_s=bin_width_s, haar_python_path=haar_python_path, time_resolved=True, window_size_s=mvt_time_window, step_size_s=mvt_step_size)
            
            #mvt_val = float(mvt_res[2]) * 1000 if mvt_res else -100
            #mvt_err = float(mvt_res[3]) * 1000 if mvt_res else -200

            iteration_results.append({ **base_iter_detail, 'analysis_bin_width_ms': bin_width_ms, **base_params})
            MVT_time_resolved_results.append(mvt_res)

        except Exception as e:
            logging.warning(f"Failed analysis on realization {i}. Error: {e}")
            iteration_results.append({'iteration': i + 1, **base_params})
            #MVT_time_resolved_results.append(None)
    
    # --- NEW: Call the dedicated analysis function ---
    mvt_summary_df = analysis_mvt_time_resolved_results_to_dataframe(
        mvt_results=MVT_time_resolved_results,
        output_info=output_info, # Pass the dict containing path, name, etc.
        bin_width_ms=bin_width_ms,
        total_runs=NN_analysis
    )

    # --- NEW: Call the final plotting function AFTER the loop ---
    if first_realization_events:
        # Define the output filename for the plot data
        plot_data_filename = output_info['file_path'] / f"plot_data_{output_info['file_info']}.npz"
        
        # Prepare the model_info dictionary with all necessary info
        model_info_for_plot = {
            'base_params': sim_params,
            'mvt_window_size_s': analysis_settings.get('mvt_window_size_s', 1.0)
        }

        # Save all components to a single compressed .npz file
        np.savez_compressed(
            plot_data_filename,
            source_events=first_realization_events['source'],
            background_events=first_realization_events['background'],
            model_info=model_info_for_plot,
            output_info=output_info,
            # Convert DataFrame to a dict for robust saving in .npz
            mvt_summary_df=mvt_summary_df.to_dict(orient='list'),
        )
        logging.info(f"Plotting data saved to {plot_data_filename}")
        create_final_plot_with_MVT(
            source_events=first_realization_events['source'],
            background_events=first_realization_events['background'],
            model_info={'base_params': sim_params}, # Pass sim_params for plot settings
            output_info=output_info,
            mvt_summary_df=mvt_summary_df # Pass the final MVT summary
        )
        
    

    return iteration_results, mvt_summary_df, NN_analysis



def analysis_mvt_time_resolved_results_to_dataframe(
    mvt_results: List[List[Dict]],
    output_info: Dict[str, any],
    bin_width_ms: float,
    total_runs: int
) -> pd.DataFrame:
    """
    Analyzes a collection of time-resolved MVT results from multiple simulations.

    This function flattens the nested data, groups it by time, calculates
    MVT statistics (median, C.I.) for each time step, generates distribution
    plots for each step, and returns a final summary DataFrame.

    Args:
        mvt_results (List[List[Dict]]): The nested list of MVT results.
        output_info (Dict[str, any]): A dictionary containing output metadata:
            - 'output_path' (Path): The directory to save results.
            - 'param_dir_name' (str): The name of the simulation parameter set.
            - 'selection_str' (str): The detector selection string for file naming.
        bin_width_ms (float): The bin width used in the analysis in milliseconds.
        total_runs (int): The total number of simulation iterations (NN_analysis).

    Returns:
        pd.DataFrame: A DataFrame summarizing the MVT statistics for each time step.
    """
    # --- Step 1: Flatten the nested MVT data into a single list ---
    flat_mvt_data = []
    for run_index, run_results in enumerate(mvt_results):
        if run_results:  # Check if the run was successful
            for time_window_result in run_results:
                time_window_result['run_index'] = run_index
                flat_mvt_data.append(time_window_result)

    if not flat_mvt_data:
        logging.warning(f"MVT analysis produced no valid data points for {output_info['file_info']}.")
        return pd.DataFrame() # Return an empty DataFrame

    mvt_df = pd.DataFrame(flat_mvt_data)
    
    # Save the raw, flattened data for detailed inspection
    flat_csv_path = output_info['file_path'] / f"Detailed_MVT_flat_{output_info['file_info']}_default_{bin_width_ms}ms.csv"
    mvt_df.to_csv(flat_csv_path, index=False)

    # --- Step 2: Group by each time step and calculate statistics ---
    time_resolved_summary = []
    for center_time, group in mvt_df.groupby('center_time_s'):
        valid_mvts = group[group['mvt_err_ms'] > 0]

        if len(valid_mvts) < 2:
            continue # Skip if there are not enough valid runs for statistics

        # --- BUG FIX ---
        # The MVT is in the 'mvt_s' column. Convert to 'ms' for stats and plotting.
        start_time = valid_mvts['start_time_s'].mean()
        end_time = valid_mvts['end_time_s'].mean()

        mvt_values_ms = valid_mvts['mvt_ms'] #* 1000
        p16, median_mvt, p84 = np.percentile(mvt_values_ms, [16, 50, 84])
        
        # --- Create a distribution plot for THIS time step ---
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use all positive MVT values for the background distribution
            all_positive_runs_ms = group[group['mvt_ms'] > 0]['mvt_ms']

            if not all_positive_runs_ms.empty:
                 ax.hist(all_positive_runs_ms, bins=30, density=True,
                         label=f'All Runs w/ MVT > 0 ({len(all_positive_runs_ms)}/{total_runs})',
                         color='gray', alpha=0.5)

            ax.hist(mvt_values_ms, bins=30, density=True,
                    label=f'Valid Runs w/ Err > 0 ({len(valid_mvts)}/{total_runs})',
                    color='steelblue', edgecolor='black')

            ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2,
                       label=f"Median = {median_mvt:.3f} ms")
            ax.axvspan(p16, p84, color='darkorange', alpha=0.2,
                       label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")

            ax.set_title(f"MVT Distribution for {output_info['file_info']}\nTime Step: {center_time:.2f} s | Bin Width: {bin_width_ms} ms")
            ax.set_xlabel("Minimum Variability Timescale (ms)")
            ax.set_ylabel("Probability Density")
            ax.legend()
            fig.tight_layout()

            plot_filename = f"MVT_dist_{output_info['file_info']}_default_{bin_width_ms}ms_T{center_time:.2f}s.png"
            plt.savefig(output_info['file_path'] / plot_filename, dpi=150)
            plt.close(fig)

        except Exception as e:
            logging.error(f"Error creating MVT distribution plot for time {center_time}s: {e}")

        time_resolved_summary.append({
            'center_time_s': center_time,
            'start_time_s': start_time,
            'end_time_s': end_time,
            'median_mvt_ms': round(median_mvt, 4),
            'mvt_err_lower_ms': round(median_mvt - p16, 4),
            'mvt_err_upper_ms': round(p84 - median_mvt, 4),
            'successful_runs': len(valid_mvts),
            'total_runs_at_step': len(group),
            'failed_runs': len(group) - len(valid_mvts),
        })

    final_MVT_csv_path = output_info['file_path'] / f"MVT_{output_info['file_info']}_default_{bin_width_ms}ms.csv"
    time_resolved_summary_df = pd.DataFrame(time_resolved_summary)
    time_resolved_summary_df.to_csv(final_MVT_csv_path, index=False)

    return time_resolved_summary_df



def GBM_MVT_analysis_det(input_info: Dict,
                output_info: Dict):
    #params = input_info['base_params']
    sim_data_path = input_info['sim_data_path']
    haar_python_path = input_info['haar_python_path']
    analysis_settings = input_info['analysis_settings']
    snr_timescales = analysis_settings.get('snr_timescales', [0.010, 0.016, 0.032, 0.064, 0.128, 0.256])
    bin_width_ms = input_info['bin_width_ms']
    sim_params_file = input_info['sim_par_file']
    dets = input_info['analysis_det']

    if type(sim_params_file) is dict:
        sim_params = sim_params_file
    else:
        sim_params = yaml.safe_load(open(sim_params_file, 'r'))
    #sim_params = yaml.safe_load(open(sim_params_file[0], 'r'))
    base_params = input_info['base_params']
    NN_analysis = base_params['num_analysis']
    #print_nested_dict(base_params)

    en_lo = sim_params.get('en_lo', 8.0)
    en_hi = sim_params.get('en_hi', 900.0)
  
   
    t_start = sim_params['t_start']
    t_stop = sim_params['t_stop']
    trange = [t_start, t_stop]
    #print("---------------")
    #dets = convert_det_to_list(sim_params.get('det', 'nn'))
    #print("sim_params.get('det', 'nn'):", sim_params.get('det', 'nn'))
    #print("Detector:", dets)
    #print("---------------")
    angle = sim_params.get('angle', 0)
    trigger_number = sim_params.get('trigger_number', 0)
    name_key = sim_params.get('name_key', 'test')

    energy_range_nai = (en_lo, en_hi)
    src_start, src_stop = calculate_src_interval(sim_params)
    src_duration = src_stop - src_start
    duration = sim_params['t_stop'] - sim_params['t_start']



    for det in dets:
        src_filename, bkgd_filename = sim_gbm_name_format(
                trigger_number=trigger_number,
                det=det,
                name_key=name_key,
                r_seed='*'
            )
        
        src_event_file_list = list(sim_data_path.glob(src_filename))
        bkgd_event_file_list = list(sim_data_path.glob(bkgd_filename))
        #print(f"Source files: {src_event_file_list[0]}")
        #print(f"Background files: {bkgd_event_file_list[0]}")
        if len(src_event_file_list) != len(bkgd_event_file_list):
            logging.error(f"GBM analysis for {output_info['file_info']} has mismatched source and background files.")
        if len(src_event_file_list) < NN_analysis:
            print("####################################################")
            print(f"GBM analysis for {output_info['file_info']} has insufficient files ({len(src_event_file_list)} < {NN_analysis}).")
            print("####################################################")

    iteration_results = []
    #NN = len(src_event_files)
    for i in range(NN_analysis):
        iteration_seed = sim_params['random_seed'] + i

        src_tte_list = []
        bkgd_tte_list = []
        for det in dets:
            src_filename, bkgd_filename = sim_gbm_name_format(
                trigger_number=trigger_number,
                det=det,
                name_key=name_key,
                r_seed=iteration_seed
            )
            src_file_path = sim_data_path / src_filename
            bkgd_file_path = sim_data_path / bkgd_filename
            #src_tte_list.append(str(src_file_path))
            #bkgd_tte_list.append(str(bkgd_file_path))
            src_tte_list.append(GbmTte.open(str(src_file_path)).slice_time(trange))
            bkgd_tte_list.append(GbmTte.open(str(bkgd_file_path)).slice_time(trange))

        tte_src = GbmTte.merge(src_tte_list)
        tte_bkgd = GbmTte.merge(bkgd_tte_list)

        # Open the files
        #tte_src_all = GbmTte.open(src_file)
        #tte_bkgd_all = GbmTte.open(bkgd_file)

        #tte_src = tte_src_all.slice_time([t_start, t_stop])
        #tte_bkgd = tte_bkgd_all.slice_time([t_start, t_stop])

        total_src_counts = tte_src.data.size
        total_bkgd_counts = tte_bkgd.data.size
        background_level_cps = total_bkgd_counts / duration
        background_counts = background_level_cps * src_duration
        snr_fluence = total_src_counts / np.sqrt(background_counts)

        # merge the background and source
        tte_total = GbmTte.merge([tte_src, tte_bkgd])

        #try:
        fine_bw = 0.001
        phaii = tte_total.to_phaii(bin_by_time, fine_bw)
        lc_total = phaii.to_lightcurve(energy_range=energy_range_nai)

        snr_results_dict = _calculate_multi_timescale_snr(
                    total_counts=lc_total.counts, sim_bin_width=0.001,
                    back_avg_cps=total_bkgd_counts/(t_stop - t_start),
                    search_timescales=snr_timescales
                )
        
        #except Exception as e:
        #    print(f"Error during SNR computing: {e}")

        base_iter_detail = {
                    'iteration': i + 1,
                    'random_seed': iteration_seed,
                    'back_avg_cps': round(background_level_cps, 2),
                    'bkgd_counts': int(background_counts),
                    'src_counts': total_src_counts,
                    'S_flu': round(snr_fluence, 2),
                    **snr_results_dict,
                }

        if i == 1:
            output_info["file_name"] = 'combined_' + src_file_path.stem
            output_info["combine_flag"] = True
            output_info["dets"] = dets
            try:
                plot_gbm_lc(tte_total, tte_src, tte_bkgd,
                            bkgd_cps=background_counts,
                            model_info={
                                    'func': None,
                                    'func_par': None,
                                    'base_params': sim_params,
                                    'snr_analysis': snr_timescales
                                },
                                output_info=output_info
                        )
            except Exception as e:
                print(f"Failed to generate representative GBM plot. Error: {e}")

        # Loop through analysis bin widths

        try:
            bin_width_s = bin_width_ms / 1000.0
            phaii_hi = tte_total.to_phaii(bin_by_time, bin_width_s)
            phaii = phaii_hi.slice_energy(energy_range_nai)
            data = phaii.to_lightcurve()

            mvt_res = run_mvt_in_subprocess(data.counts,
                                            bin_width_s=bin_width_s,
                                            haar_python_path=haar_python_path)
            plt.close('all')
            mvt_val = mvt_res['mvt_ms']
            mvt_err = mvt_res['mvt_err_ms']
        except Exception as e:
            print(f"Error during MVT calculation for bin width {bin_width_ms} ms: {e}")
            mvt_val = -100
            mvt_err = -100

        iter_detail = {**base_iter_detail,
                        'analysis_bin_width_ms': bin_width_ms,
                        'mvt_ms': round(mvt_val, 4),
                        'mvt_err_ms': round(mvt_err, 4),
                        **sim_params}
        iteration_results.append(iter_detail)


    return iteration_results, NN_analysis


def GBM_MVT_analysis_complex(input_info: Dict, output_info: Dict):
    # --- 1. Unpack parameters and find feature pulse directory ---
    base_params = input_info['base_params']
    analysis_settings = input_info['analysis_settings']
    template_dir_path = input_info['sim_data_path']
    data_path_root = template_dir_path.parents[2]
    bin_width_ms = input_info['bin_width_ms']
    haar_python_path = input_info['haar_python_path']
    feature_amplitude = base_params['peak_amplitude']
    position_shift = base_params['position']
    extra_pulse_config = analysis_settings.get('extra_pulse', {})
    sim_params = input_info['sim_par_file']
    dets = input_info.get('analysis_det', [])
    
    feature_shape = extra_pulse_config.get('pulse_shape')
    feature_params_for_naming = { 'peak_amplitude': feature_amplitude, **{k: v for k, v in extra_pulse_config.items() if k != 'pulse_shape'} }
    partial_name_key = _create_param_directory_name('gbm', feature_shape, feature_params_for_naming, extra_pulse=True)
    feature_name_key = 'det_all-' + partial_name_key # Assumes feature was generated with det='all'
    feature_dir_path = data_path_root / 'gbm' / feature_shape / feature_name_key

    if not feature_dir_path.exists():
        logging.error(f"Assembly Error: Feature directory not found at {feature_dir_path}")
        return [], 0
    
    feature_params_for_interval = {**feature_params_for_naming, 'pulse_shape': feature_shape}
    feat_t_start_relative, feat_t_stop_relative = calculate_src_interval(feature_params_for_interval)
 
    #print(feature_params_for_interval)
    # --- 2. Main Analysis Loop ---
    NN_analysis = base_params.get('num_analysis', 0)
    t_start, t_stop = sim_params['t_start'], sim_params['t_stop']
    trange = [t_start, t_stop]
    duration = t_stop - t_start
    trigger_number = sim_params.get('trigger_number', 0)
    name_key = sim_params.get('name_key', 'test')
    snr_timescales = analysis_settings.get('snr_timescales', [])
    energy_range_nai = (sim_params.get('en_lo', 8.0), sim_params.get('en_hi', 900.0))
    iteration_results = []

    for i in range(NN_analysis):
        try:
            iteration_seed = sim_params['random_seed'] + i
            
            # --- 3. Load and merge TTE files for this realization ---
            src_tte_list, bkgd_tte_list, feature_tte_list = [], [], []
            for det in dets: 
                src_filename, bkgd_filename = sim_gbm_name_format(trigger_number=trigger_number, det=det, name_key=name_key, r_seed=iteration_seed)
                feature_filename, _ = sim_gbm_name_format(trigger_number=trigger_number, det=det, name_key=feature_name_key, r_seed=iteration_seed)
                src_tte_list.append(GbmTte.open(str(template_dir_path / src_filename)).slice_time(trange))
                bkgd_tte_list.append(GbmTte.open(str(template_dir_path / bkgd_filename)).slice_time(trange))
                feature_tte_list.append(GbmTte.open(str(feature_dir_path / feature_filename)).slice_time(trange))

            tte_src = GbmTte.merge(src_tte_list) # This is the template
            tte_bkgd = GbmTte.merge(bkgd_tte_list)
            tte_feature_original = GbmTte.merge(feature_tte_list)
            tte_feature = TTE_shift(tte_feature_original, position_shift)

            try:
            # --- 4. Assemble the pulse ---
                #tte_feature.data['TIME'] += position_shift
                tte_complete_src = GbmTte.merge([tte_src, tte_feature])
                tte_total = GbmTte.merge([tte_complete_src, tte_bkgd])

                background_level_cps = tte_bkgd.data.size / duration

                ## ================== FULL METRIC CALCULATIONS ==================
                # 1. Fluence SNR for each component
                src_counts_total = tte_complete_src.data.size
                bkgd_counts_total_window = background_level_cps * duration
                S_flu_total = src_counts_total / np.sqrt(bkgd_counts_total_window) if bkgd_counts_total_window > 0 else 0
                src_counts_template = tte_src.data.size
                S_flu_template = src_counts_template / np.sqrt(bkgd_counts_total_window) if bkgd_counts_total_window > 0 else 0
                src_counts_feature = tte_feature.data.size
            
                

                # --- 1a. SNR of Feature (using AVERAGE background+template rate) ---
            
                combined_bkgd_avg_cps = (tte_bkgd.data.size + tte_src.data.size) / duration

                feature_duration = feat_t_stop_relative - feat_t_start_relative
                bkgd_in_feature_window_avg = combined_bkgd_avg_cps * feature_duration
                S_flu_feature_avg = src_counts_feature / np.sqrt(bkgd_in_feature_window_avg) if bkgd_in_feature_window_avg > 0 else 0
            except Exception as e:
                print(f"Error occurred while calculating SNR 0 metrics: {e}")
                exit(1)
            
            feature_t_start_unclipped = feat_t_start_relative + position_shift
            feature_t_stop_unclipped = feat_t_stop_relative + position_shift

            # --- 1b. SNR of Feature (using LOCAL background+template counts) ---
            #feature_t_start, feature_t_stop = feat_t_start_relative + position_shift, feat_t_stop_relative + position_shift
            #print(f"Feature time window: {feature_t_start} to {feature_t_stop}")
            try:
                # --- 1b. SNR of Feature (using LOCAL background+template counts) ---
                # Clip the feature's window to the valid simulation range (trange)
                feature_t_start_src = max(tte_src.time_range[0], feature_t_start_unclipped)
                feature_t_stop_src = min(tte_src.time_range[1], feature_t_stop_unclipped)

                # This is safe: if start > stop, slice_time returns an empty object, 
                # and .data.size will correctly be 0.
                #print(f"Time window: {feature_t_start_src} to {feature_t_stop_src}")
                #print(f"Trange: {tte_src.time_range[0]} to {tte_src.time_range[1]}")
                if feature_t_start_src >= feature_t_stop_src:
                    #print("Feature time window is invalid")
                    template_counts_in_window = 0
                else:
                    #print(f"Feature time window: {feature_t_start_src} to {feature_t_stop_src}")
                    template_counts_in_window = tte_src.slice_time([feature_t_start_src, feature_t_stop_src]).data.size
            except Exception as e:
                print(f"Error occurred while calculating SNR 1 metrics: {e}")
                exit(1)
            
            try:
                feature_t_start_bkgd = max(tte_bkgd.time_range[0], feature_t_start_unclipped)
                feature_t_stop_bkgd = min(tte_bkgd.time_range[1], feature_t_stop_unclipped)

                bkgd_counts_in_window = tte_bkgd.slice_time([feature_t_start_bkgd, feature_t_stop_bkgd]).data.size
                bkgd_counts_feature_local = template_counts_in_window + bkgd_counts_in_window
                S_flu_feature_local = src_counts_feature / np.sqrt(bkgd_counts_feature_local) if bkgd_counts_feature_local > 0 else 0
            except Exception as e:
                print(f"Error occurred while calculating SNR 2 metrics: {e}")
                exit(1)

            # 2. Multi-timescale SNR for each component
            fine_bw = 0.001
            # --- 2a. Multi-timescale SNR for TOTAL pulse ---
            lc_total = tte_total.to_phaii(bin_by_time, fine_bw).to_lightcurve(energy_range=energy_range_nai)
            snr_dict_total = _calculate_multi_timescale_snr(total_counts=lc_total.counts, sim_bin_width=fine_bw, back_avg_cps=background_level_cps, search_timescales=snr_timescales)
            
            # --- 2b. Multi-timescale SNR for TEMPLATE pulse ---
            lc_template_plus_bkgd = GbmTte.merge([tte_src, tte_bkgd]).to_phaii(bin_by_time, fine_bw).to_lightcurve(energy_range=energy_range_nai)
            snr_dict_template = _calculate_multi_timescale_snr(total_counts=lc_template_plus_bkgd.counts, sim_bin_width=fine_bw, back_avg_cps=background_level_cps, search_timescales=snr_timescales)

            # --- 2c. Multi-timescale SNR for FEATURE pulse ---
            snr_dict_feature = _calculate_multi_timescale_snr(total_counts=lc_total.counts, sim_bin_width=fine_bw, back_avg_cps=combined_bkgd_avg_cps, search_timescales=snr_timescales)
            
            ## =============================================================

            # Rename SNR dictionary keys to be unique
            #final_snr_dict = {f'{k}_total': v for k, v in snr_dict_total.items()}
            final_snr_dict = {f'{k}_total': v for k, v in snr_dict_total.items()}
            final_snr_dict.update({f'{k}_template': v for k, v in snr_dict_template.items()})
            final_snr_dict.update({f'{k}_feature': v for k, v in snr_dict_feature.items()})

            base_iter_detail = {
                'iteration': i + 1, 'random_seed': iteration_seed, 'back_avg_cps': round(background_level_cps, 2),
                'src_counts': src_counts_total,
                'src_counts_total': src_counts_total, 'bkgd_counts': tte_bkgd.data.size,
                'src_counts_template': src_counts_template, 'src_counts_feature': src_counts_feature,
                'bkgd_counts_feature_local': bkgd_counts_feature_local, 'S_flu': round(S_flu_total, 2),
                'S_flu_total': round(S_flu_total, 2), 'S_flu_template': round(S_flu_template, 2),
                'S_flu_feature_avg': round(S_flu_feature_avg, 2), 'S_flu_feature_local': round(S_flu_feature_local, 2),
                **final_snr_dict
            }


            #print(f"dets: {dets}")      
            if i == 1:
                output_info["file_name"] = 'combined_' + (template_dir_path / src_filename).stem
                output_info["combine_flag"] = True
                output_info["dets"] = dets

                det_string = "_".join([f"{det}" for det in dets])
                output_info["file_name"] = 'combined_' + (template_dir_path / src_filename).stem + f"_det_{det_string}" 
                output_info["det_string"] = det_string
                try:
                    plot_gbm_lc(tte_total, tte_src, tte_bkgd,
                                bkgd_cps=tte_bkgd.data.size,
                                model_info={
                                        'func': None,
                                        'func_par': None,
                                        'base_params': sim_params,
                                        'snr_analysis': snr_timescales,
                                    },
                                    output_info=output_info
                            )
                except Exception as e:
                    print(f"Failed to generate representative GBM plot. Error: {e}")
            
            # --- 5. MVT on the Assembled Pulse ---
            phaii = tte_total.to_phaii(bin_by_time, bin_width_ms / 1000.0)
            data = phaii.to_lightcurve(energy_range=energy_range_nai)
            mvt_res = run_mvt_in_subprocess(data.counts, bin_width_s=bin_width_ms / 1000.0, haar_python_path=haar_python_path)
            mvt_val = mvt_res['mvt_ms'] if mvt_res else -100
            mvt_err = mvt_res['mvt_err_ms'] if mvt_res else -200

            iteration_results.append({ **base_iter_detail, 'analysis_bin_width_ms': bin_width_ms, 'mvt_ms': round(mvt_val, 4), 'mvt_err_ms': round(mvt_err, 4), **base_params })

        except Exception as e:
            logging.warning(f"Failed analysis on realization {i}. Error: {e}")
            iteration_results.append({'iteration': i + 1, 'mvt_ms': -100, 'mvt_err_ms': -200, **base_params})
            
    return iteration_results, NN_analysis


def generate_function_events(
    func: Callable,
    func_par: Tuple,
    back_func: Callable,
    back_func_par: Tuple,
    params: Dict[str, Any],
    back_flag: bool = True,
    source_flag: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a TTE simulation and returns source and/or background events based on flags.

    Args:
        ...
        back_flag (bool): If True, simulate and return background events.
        source_flag (bool): If True, simulate and return source events.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing (source_events, background_events).
                                       An array will be empty if its flag was False.
    """
    # If neither is requested, return empty arrays immediately
    if not source_flag and not back_flag:
        return np.array([]), np.array([])

    # Unpack necessary parameters from the dictionary
    t_start = params['t_start']
    t_stop = params['t_stop']
    random_seed = params['random_seed']
    source_base_rate = params.get('scale_factor', 1.0)
    background_base_rate = params.get('scale_factor', 1.0)
    grid_resolution = params.get('grid_resolution', 0.0001)

    np.random.seed(random_seed)

    # 1. Efficiently define the total rate function based on flags
    def total_rate_func(t):
        source_rate = func(t, *func_par) * source_base_rate if source_flag and func else 0
        background_rate = back_func(t, *back_func_par) * background_base_rate if back_flag and back_func else 0
        return source_rate + background_rate

    # 2. Simulate the total required event stream
    grid_times = np.arange(t_start, t_stop, grid_resolution)
    rate_on_grid = total_rate_func(grid_times)
    cumulative_counts = np.cumsum(rate_on_grid) * grid_resolution
    total_expected_counts = cumulative_counts[-1] if len(cumulative_counts) > 0 else 0
    num_events = np.random.poisson(total_expected_counts)
    random_counts = np.random.uniform(0, total_expected_counts, num_events)
    total_event_times = np.interp(random_counts, cumulative_counts, grid_times)

    # 3. Conditionally assign or split events
    source_event_times = np.array([])
    background_event_times = np.array([])

    if source_flag and back_flag:
        # If we need both, we must do the probabilistic split
        source_rate_at_events = func(total_event_times, *func_par) * source_base_rate
        background_rate_at_events = back_func(total_event_times, *back_func_par) * background_base_rate
        total_rate_at_events = source_rate_at_events + background_rate_at_events

        p_source = np.divide(source_rate_at_events, total_rate_at_events,
                             out=np.zeros_like(total_rate_at_events),
                             where=total_rate_at_events > 0)
        
        is_source_event = np.random.rand(num_events) < p_source
        source_event_times = np.sort(total_event_times[is_source_event])
        background_event_times = np.sort(total_event_times[~is_source_event])

    elif source_flag:
        # If we only simulated the source, all events are source events
        source_event_times = np.sort(total_event_times)

    elif back_flag:
        # If we only simulated the background, all events are background events
        background_event_times = np.sort(total_event_times)

    return source_event_times, background_event_times



def generate_gbm_events_dets(
        event_file_path: Path,
        func: Callable,
        func_par: Tuple,
        back_func: Callable,
        back_func_par: Tuple,
        params: Dict[str, Any],
        back_flag: bool = True,
        source_flag: bool = True,
        det_flag: bool = False):

    #dets = params['det']
    trigger_number = params['trigger_number']
    #angle = params['angle']
    en_lo = params['en_lo']
    en_hi = params['en_hi']
    t_start = params['t_start']
    t_stop = params['t_stop']

    select_time = (t_start, t_stop)
    random_seed = params['random_seed']
    grid_resolution = params.get('grid_resolution', 0.0001) # Use a fixed, fine

    energy_range_nai = (en_lo, en_hi)
    #print(f"angle: {type(params['angle2'])}")
    raw_intervals = params['background_intervals']
    bkgd_times = raw_intervals# parse_intervals_from_csv(raw_intervals)  #[(-20.0, -5.0), (75.0, 200.0)]
    #print(f"Background intervals: {type(bkgd_times)}")
    #print(type(bkgd_times))

    # Fixed spectral Model
    band_params = (0.1, 300.0, -1.0, -2.5)

    #tte = GbmTte.open('glg_tte_n6_bn250612519_v00.fit')
    folder_path = os.path.join(os.getcwd(), f'bn{trigger_number}')
    dets = convert_det_to_list(params['det'])
    #print(f"Detectors: {dets}")

    src_file_list = []
    bkgd_file_list = []
    for det in dets:
        tte_pattern = f'{folder_path}/glg_tte_{det}_bn{trigger_number}_v*.fit'
        tte_files = glob.glob(tte_pattern)

        if not tte_files:
            raise FileNotFoundError(f"No TTE file found matching pattern: {tte_pattern}")
        tte_file = tte_files[0]  # Assuming only one file/version per det/trigger_number

        # Find the RSP2 file (e.g., glg_cspec_n3_bn230307053_v03.rsp2)
        rsp2_pattern = f'{folder_path}/glg_cspec_{det}_bn{trigger_number}_v*.rsp2'
        rsp2_files = glob.glob(rsp2_pattern)

        if not rsp2_files:
            raise FileNotFoundError(f"No RSP2 file found matching pattern: {rsp2_pattern}")
        rsp2_file = rsp2_files[0]  # Assuming only one file/version per det/trigger_number
        rsp2 = GbmRsp2.open(rsp2_file)
        rsp = rsp2.extract_drm(atime=np.average(select_time))

        # Use the .name property to get the descriptive base filename
        base_filename = event_file_path.name
        src_filename, bkgd_filename = sim_gbm_name_format(
            trigger_number=trigger_number,
            det=det,
            name_key=base_filename,
            r_seed=None
        )
        src_path = event_file_path.parent / src_filename
        bkgd_path = event_file_path.parent / bkgd_filename
        # Open the files
        tte = GbmTte.open(tte_file)
        tte_demo = tte.slice_time([-50,-49.99])


        if source_flag:
        # source simulation
            tte_sim = TteSourceSimulator(rsp, Band(), band_params, func, func_par, deadtime=1e-6, sample_period=grid_resolution, rng=np.random.default_rng(random_seed))
            tte_src = tte_sim.to_tte(t_start, t_stop)
            tte_gbm_src = GbmTte.merge([tte_demo, tte_src])
            # Construct the new FITS filenames
            
            # Save the files to the correct directory with the new names
            tte_gbm_src.write(filename=src_filename, directory=event_file_path.parent, overwrite=True)

        if back_flag:
            #bin to 1.024 s resolution, reference time is trigger time
            phaii = tte.to_phaii(bin_by_time, 1.024, time_ref=0.0)
            bkgd_times = bkgd_times
            backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)
            backfitter.fit(order=1)
            bkgd = backfitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
            
            # the background model integrated over the source selection time
            spec_bkgd = bkgd.integrate_time(*select_time)
            
            # background simulation
            tte_sim = TteBackgroundSimulator(spec_bkgd, 'Gaussian', back_func, back_func_par, deadtime=1e-6, sample_period=grid_resolution, rng=np.random.default_rng(random_seed))
            tte_bkgd = tte_sim.to_tte(t_start, t_stop)
            tte_gbm_bkgd = GbmTte.merge([tte_demo, tte_bkgd])
            tte_gbm_bkgd.write(filename=bkgd_filename, directory=event_file_path.parent, overwrite=True)
        
        src_file_list.append(src_path)
        bkgd_file_list.append(bkgd_path)

    return src_file_list, bkgd_file_list
