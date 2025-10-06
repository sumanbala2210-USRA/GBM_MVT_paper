import os
import sys
import yaml

from gdt.missions.fermi.gbm.tte import GbmTte

from gdt.missions.fermi.gbm.finders import TriggerFtp
from gdt.core.binning.unbinned import bin_by_time
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.missions.fermi.time import *
from gdt.core.phaii import Phaii
from datetime import datetime
import numpy as np 
from gdt.missions.fermi.gbm.trigdat import Trigdat
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np 
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from gdt.core.plot.lightcurve import Lightcurve
from gdt.missions.fermi.gbm.finders import TriggerFinder

import numpy as np
from pathlib import Path

import argparse
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import functools

from SIM_lib import run_mvt_in_subprocess
from poly_find import find_best_poly_order
from tqdm import tqdm

#from LIB_time import _calculate_multi_timescale_snr, analysis_mvt_time_resolved_results_to_dataframe, create_final_GBM_plot_with_MVT, analysis_mvt_results_to_dataframe

from TTE_SIM_v2 import _calculate_multi_timescale_snr, analysis_mvt_time_resolved_results_to_dataframe, create_final_GBM_plot_with_MVT, analysis_mvt_results_to_dataframe, write_yaml
from SIM_lib import send_email

CONFIG_FILE = 'config_MVT_fermi.yaml'
MVT_CONFIG_FILE = 'simulations_ALL.yaml'
MAX_WORKERS = os.cpu_count() - 2
#time_resolved = False # config_MVT['time_resolved']

def setup_publication_style():
    """
    Sets Matplotlib parameters for high-quality, publication-ready plots.
    This function uses a serif font, larger text sizes, and inward-pointing ticks.
    """
    # Using a style sheet is a good base. 'seaborn-v0_8-paper' is clean.
    plt.style.use('seaborn-v0_8-paper')

    params = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'], # A classic academic font
        'font.size': 14,               # Base font size
        'axes.labelsize': 16,          # X and Y labels
        'axes.titlesize': 16,          # Title for individual subplots
        'xtick.labelsize': 14,         # X-axis tick labels
        'ytick.labelsize': 14,         # Y-axis tick labels
        'legend.fontsize': 14,         # Legend font size
        'figure.titlesize': 18,        # Main figure title (suptitle)
        'lines.linewidth': 1.5,
        'xtick.direction': 'in',       # Ticks point inward
        'ytick.direction': 'in',
        'xtick.top': True,             # Display ticks on all 4 sides
        'ytick.right': True,
        'savefig.dpi': 300,            # High resolution for raster elements
        'savefig.bbox': 'tight',       # No wasted whitespace
    }
    plt.rcParams.update(params)
    print("Matplotlib style set for publication quality.")

def download_data(trigger_number, path, flags={'tte':False, 'rsp':False, 'cat':False, 'trigdat':False}):
    try:
        trigger_ftp = TriggerFinder(trigger_number, protocol='AWS')
    except Exception as e:
        print(f"Failed to initialize TriggerFinder for {trigger_number}: {e}")
        exit(1)
    if flags.get('tte', False):
        trigger_ftp.get_tte(download_dir=path)
    if flags.get('rsp', False):
        trigger_ftp.get_rsp2(download_dir=path)
    if flags.get('cat', False):
        trigger_ftp.get_cat_files(download_dir=path)
    if flags.get('trigdat', False):
        trigger_ftp.get_trigdat(download_dir=path)
    
def trigger_file_list(trigger_dir, file_type, trigger_number, nai_dets=None, bgo_flag=False):
    """
    Generate lists of file absolute paths and file names based on trigger directory, file type, and trigger number.

    Args:
    trigger_dir (str): The directory where the files are located.
    file_type (str): The type of files to include.
    trigger_number (str): The trigger number to filter files.

    Returns:
    tuple: A tuple containing two lists:
           - List of file absolute paths.
           - List of file names.
    """
    # Generate file absolute paths using glob based on file type and trigger number
    #file_abs_path_list = glob.glob(trigger_dir + "/glg_" + file_type + "_*_bn" + trigger_number + "*.fit")
    #det_list = ['n6', 'n7', 'n9']
    file_abs_path_list = []
    if file_type in ['trigdat', 'tcat', 'bcat']:
        bgo_flag = True
    if nai_dets == 'all':
        if bgo_flag:
            pattern = trigger_dir / f"glg_{file_type}_*_bn{trigger_number}*.fit"
        else:
            pattern = trigger_dir / f"glg_{file_type}_n*_bn{trigger_number}*.fit"
        file_abs_path_list = glob.glob(str(pattern))
    elif nai_dets is None:
        # If nai_dets is None, use all detectors
        if bgo_flag:
            pattern = trigger_dir / f"glg_{file_type}_*_bn{trigger_number}*.fit"
        else:
            pattern = trigger_dir / f"glg_{file_type}_n*_bn{trigger_number}*.fit"
        file_abs_path_list = glob.glob(str(pattern))
    else:
        for det in nai_dets:
            pattern = trigger_dir / f"glg_{file_type}_{det}_bn{trigger_number}*.fit"
            file_abs_path_list.extend(glob.glob(str(pattern)))

    file_name_list = [os.path.basename(path) for path in file_abs_path_list]

    return sorted(file_abs_path_list), sorted(file_name_list)


def get_GRB_par(trigger_number, trigger_directory):
    #print(f"Getting GRB parameters for {trigger_number} from BCAT...")
    try:
        bcat_down_list_path, _ = trigger_file_list(trigger_directory, "bcat", trigger_number)
        bcat_hdu = fits.open(bcat_down_list_path[-1])
        T0 = round(float(bcat_hdu[0].header['T90START']), 4)
        T90 = round(float(bcat_hdu[0].header['T90']), 4)
        T50 = round(float(bcat_hdu[0].header['T50']), 4)
        PF64 = round(float(bcat_hdu[0].header['PF64']), 4)
        PFLX = round(float(bcat_hdu[0].header['PFLX']), 4)
        FLU = round(float(bcat_hdu[0].header['FLU'])*1e6, 4)
        bcat_hdu.close()
        return T0,T90, T50, PF64, PFLX, FLU
    except Exception as e:
        print(f"Failed to get T90 for {trigger_number}: {e}")
        return None, None, None, None, None, None


def check_GBM_status(tig_time):
    problem_detectors = []
    file_path = 'gbm_status.csv'
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    for i in range(0,len(df.TSTART)):
        bad_time_start = Time(df.TSTART[i], format='iso')
        bad_time_stop = Time(df.TSTOP[i], format='iso')
        
        #print(tig_time.fermi,  bad_time_start.fermi)
        if tig_time.fermi > bad_time_start.fermi and tig_time.fermi < bad_time_stop.fermi:
            print(f"Trigger Within bad time {df.TSTART[i]} & {df.TSTOP[i]}")
            print(f"Problem with {df.Detectors[i]} detector(s): {df.Comment[i]}\n")
            
            if 'all' in df.Detectors[i]:
                problem_detectors = [det.name for det in GbmDetectors]
            else:
                problem_detectors = [word.lower() for word in df.Detectors[i].split()] 
    return problem_detectors

def get_dets_list(trigger_number, trigger_directory):
    try:
        #trigger_directory = os.path.join(data_path, "bn" + trigger_number)
        #print(f"trigger directory: {trigger_directory}")
        tcat_down_list_path, _ = trigger_file_list(trigger_directory, "tcat", trigger_number)
        tcat_hdu = fits.open(tcat_down_list_path[-1])
        tcat_header_hdu0 = tcat_hdu[0].header
        ra_source = tcat_header_hdu0['RA_OBJ']
        dec_source = tcat_header_hdu0['DEC_OBJ']
        error_source = tcat_header_hdu0['ERR_RAD']
        #source_position = {'ra':ra_source, 'dec': dec_source, 'error': error_source}
        source_coord = SkyCoord(ra_source, dec_source, frame='icrs', unit='deg')
        #time_scale = tcat_header_hdu0['TRIGSCAL']
        tcat_hdu.close()
        
    except Exception as e:
        print(f"Failed to get source coordinates for {trigger_number}: {e}")

    try:
        trigdat_down_list_path, _ = trigger_file_list(trigger_directory, "trigdat", trigger_number)
        trigdat = Trigdat.open(trigdat_down_list_path[-1])
        tig_time = Time(trigdat.trigtime, format='fermi')
        frame = trigdat.poshist.at(tig_time)
        #obj = trigdat.headers[0]['OBJECT'][:-9]
    except Exception as e:
        print(f"Failed to get trigdat for {trigger_number}: {e}")     

    try:        
        detectors_data = {}
        for det in GbmDetectors:
            angle = frame.detector_angle(det.name, source_coord).deg[0]
            detectors_data[det.name] = angle

        detectors_data_sorted = sorted(detectors_data.items(), key=lambda x: x[1])
        try:
            problem_detectors = check_GBM_status(tig_time)
        except Exception as e:
            print(f"Failed to check GBM status for {trigger_number}: {e}")
            problem_detectors = []

        selected_detectors_data = [(name, angle) for name, angle in detectors_data_sorted if angle <= 60.0 and name[0]!='b' and name not in problem_detectors]
        
        det_list_sorted = [item[0] for item in selected_detectors_data]
        det_list_sorted_full = [GbmDetectors.from_str(name).full_name for name in det_list_sorted]
    
        
        det_final = det_list_sorted_full#[:3]
            
        det_list = [GbmDetectors.from_full_name(name).name for name in det_final ]   

        if len(det_list)==0:
            nai_det_list = [(name, angle) for name, angle in detectors_data_sorted if name[0]!='b' and name not in problem_detectors]
            det_list.append(nai_det_list[0][0])
        
        return det_list
    except Exception as e:
        print(f"Failed to get detectors list for {trigger_number}: {e}")
        return 'all'
    
    
def normalize_det_list(det_list):
    if isinstance(det_list, str):
        return [d.strip() for d in det_list.split(",") if d.strip()]
    elif isinstance(det_list, list):
        return det_list
    else:
        return []

def config_to_det_list(config_dic, trigger_directory):
    det_list = config_dic.get("det_list", None)
    if det_list == 'None' or det_list is None or det_list == [] or det_list == 0:
        return None, None
    elif config_dic["det_list"] in ['best', 'one']:
        nai_dets = get_dets_list(config_dic['trigger_number'], trigger_directory)
        if config_dic["det_list"] == 'one':
            nai_dets = [nai_dets[0]]
            det_string = "_".join(nai_dets)
        else:
            nai_dets = nai_dets
            det_string = "_".join(nai_dets)
    elif config_dic["det_list"] != "all":
        config_dic["det_list"] = normalize_det_list(config_dic["det_list"])
        nai_dets = [d for d in config_dic['det_list'] if d.startswith('n')]
        det_string = "_".join(nai_dets)
    else:
        nai_dets = [f'n{i}' for i in range(10)] + ['na', 'nb']
        det_string = "all"
    return nai_dets, det_string



def calculate_snr(lc, back_intervals):
    """
    Calculates the signal-to-noise ratio (SNR) for a given light curve.
    
    Args:
        lc (Lightcurve): The light curve object.
        back_intervals (list of lists): A list of [start, stop] times for background.

    Returns:
        float: The calculated SNR.
    """
    times, counts = lc.centroids, lc.counts
    bkg_mask = np.zeros_like(times, dtype=bool)
    for t_start, t_stop in back_intervals:
        bkg_mask |= (times >= t_start) & (times < t_stop)
    
    bkg_counts = counts[bkg_mask]
    if len(bkg_counts) == 0:
        return np.nan
        
    avg_bkg_per_bin = np.mean(bkg_counts)
    src_counts = counts[~bkg_mask]
    if len(src_counts) == 0:
        return 0.0
        
    peak_counts = np.max(src_counts)
    net_signal = peak_counts - avg_bkg_per_bin
    
    if avg_bkg_per_bin > 0:
        snr = net_signal / np.sqrt(avg_bkg_per_bin)
    else:
        snr = np.inf if net_signal > 0 else 0.0
        
    return snr

def define_time_intervals(trigger_number, trigger_directory):
    """
    Defines source and background time intervals based on GRB properties.
    
    Args:
        trigger_number (str): The GRB trigger number.
        trigger_directory (str or Path): The directory containing trigger data.

    Returns:
        tuple: (src_range, back_intervals, trange_total)
    """
    try:
        print("Fetching GRB parameters for time windowing...")
        T0, T90, T50, PF64, PFLX, FLU = get_GRB_par(trigger_number, trigger_directory)
        source_par_dict = {
            "T0": T0,
            "T90": T90,
            "T50": T50,
            "PF64": PF64,
            "PFLX": PFLX,
            "FLU": FLU
        }
        if T90 < 2.0:
            padding = 0.256 * 10 + T90
            back_before = [T0 - padding - 50, T0 - padding]
            back_after = [T0 + T90 + padding * 2, T0 + T90 + padding * 2 + 50]
        elif 2.0 <= T90 < 6.0:
            padding = 1.024 * 10 + T90 * 0.2
            reference = min(110, max(50.0, T90)) + padding
            back_before = [T0 - padding - 50, T0 - padding]
            back_after = [T0 + T90 + padding, T0 + T90 + reference]
        else:
            padding = 1.024 * 10 + T90 * 0.2
            r_padding = 1.024 * 10 + T90 * 0.4
            reference = min(110, max(50.0, T90)) + padding
            r_reference = min(110, max(50.0, T90)) + r_padding
            back_before = [T0 - reference, T0 - padding]
            back_after = [T0 + T90 + r_padding, T0 + T90 + r_reference]
        
        src_range = (T0 - padding / 2, T0 + T90 + padding * 0.9)
        back_intervals = [back_before, back_after]
        
    except Exception as e:
        print(f"Failed to get GRB params: {e}. Using default window.")
        src_range = (-10, 120)
        back_intervals = [[-50, -10], [120, 150]]

    trange_total = (back_intervals[0][0], back_intervals[1][1])
    print(f"Source window: {src_range[0]:.2f} - {src_range[1]:.2f}")
    print(f"Background intervals: {back_intervals}")
    return src_range, back_intervals, trange_total, source_par_dict

def generate_lightcurves(tte_files, src_range, trange_total, bw=0.064, erange=(8.0, 900.0), combined=False, src_only=False):
    """
    Generates individual and combined light curves from TTE files.

    Args:
        tte_files (list): List of paths to TTE files.
        src_range (tuple): The (start, stop) time for the source plot.
        trange_total (tuple): The (start, stop) time for data extraction (incl. background).
        bw (float): The bin width in seconds.
        erange (tuple): The energy range for the light curves.
        combined (bool): Whether to generate a combined light curve.
        src_only (bool): If True, only the source interval is used for individual LCs.

    Returns:
        tuple: Contains lists of light curves and titles for plotting.
    """
    lcs_src, lcs_total, titles, tte_list = [], [], [], []
    for tte_file in tte_files:
        if not src_only:
            tte = GbmTte.open(tte_file).slice_time(trange_total)
        else:
            tte = GbmTte.open(tte_file).slice_time(src_range)
        tte_list.append(tte)

        if not combined:
            phaii = tte.to_phaii(bin_by_time, bw)
            
            # LC for plotting (source region only)
            lcs_src.append(phaii.to_lightcurve(time_range=src_range, energy_range=erange))
            # LC for SNR calculation (full range)
            lcs_total.append(phaii.to_lightcurve(energy_range=erange))
            titles.append(os.path.basename(tte_file))
        
    lc_combined_src, lc_combined_total = None, None
        # --- CHANGE IS HERE ---
    
    if len(tte_list) > 0:
        tte_combined = GbmTte.merge(tte_list)
        phaii_combined = tte_combined.to_phaii(bin_by_time, bw)
        lc_combined_src = phaii_combined.to_lightcurve(time_range=src_range, energy_range=erange)
        if not src_only:
            lc_combined_total = phaii_combined.to_lightcurve(energy_range=erange)

    return lcs_src, lcs_total, lc_combined_src, lc_combined_total, titles


# --- 2. Unified Plotting Function ---

def plot_gbm_lightcurves(trigger_directory, tte_files, src_range, back_intervals, bw=0.064, suffix=""):
    """
    Generates and saves two grid plots of GBM light curves (shared and independent y-axes).
    
    Args:
        trigger_directory (str or Path): Directory to save plots.
        tte_files (list): List of TTE file paths to plot.
        src_range (tuple): The (start, stop) time for the source plot.
        back_intervals (list of lists): Background time intervals for SNR calculation.
        bw (float): Bin width in seconds.
        suffix (str, optional): A suffix to add to the output plot filenames.
    """
    trange_total = (back_intervals[0][0], back_intervals[1][1])
    lcs_src, lcs_total, lc_comb_src, lc_comb_total, titles = generate_lightcurves(
        tte_files, src_range, trange_total, bw
    )

    num_plots = len(lcs_src) + (1 if lc_comb_src else 0)
    cols, rows = 4, int(np.ceil(num_plots / 4))

    # Internal helper to draw on axes
    def _draw_plots(axes, is_shared_y):
        ax_flat = axes.flatten()
        for i, (lc_s, lc_t, title) in enumerate(zip(lcs_src, lcs_total, titles)):
            ax = ax_flat[i]
            ax.step(lc_s.centroids, lc_s.counts, where='post')
            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            snr = calculate_snr(lc_t, back_intervals)
            ax.text(0.95, 0.95, f"SNR: {snr:.1f}", transform=ax.transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        if lc_comb_src:
            ax = ax_flat[len(lcs_src)]
            ax.step(lc_comb_src.centroids, lc_comb_src.counts, where='post', color='black', linewidth=1.5)
            ax.set_title('Combined', fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            snr = calculate_snr(lc_comb_total, back_intervals)
            ax.text(0.95, 0.95, f"SNR: {snr:.1f}", transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            if not is_shared_y:
                 ax.set_ylabel('Counts')


        for j in range(num_plots, len(ax_flat)):
            ax_flat[j].axis('off')

    # Plot 1: Shared Y-Axes
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
    _draw_plots(axes1, is_shared_y=True)
    fig1.supxlabel('Time (s)'); fig1.supylabel('Counts')
    fig1.suptitle(f'Detector Light Curves (Shared Y, {bw * 1000:.0f} ms){suffix}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path1 = Path(trigger_directory) / f"lc_grid_shared_y{suffix}_{bw * 1000:.0f}ms.png"
    plt.savefig(output_path1); plt.close(fig1)
    print(f"Saved shared-axis plot: \n{output_path1}")

    # Plot 2: Independent Y-Axes
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=False)
    _draw_plots(axes2, is_shared_y=False)
    fig2.supxlabel('Time (s)'); fig2.supylabel('Counts')
    fig2.suptitle(f'Detector Light Curves (Independent Y, {bw * 1000:.0f} ms){suffix}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path2 = Path(trigger_directory) / f"lc_grid_indep_y{suffix}_{bw * 1000:.0f}ms.png"
    plt.savefig(output_path2); plt.close(fig2)
    print(f"Saved independent-axis plot: \n{output_path2}")


# --- 3. Refactored Analysis Function ---

def find_optimal_detectors(trigger_number, trigger_directory, bw=0.064, config_dict=None, plot_flag=False):
    """
    Analyzes all detectors to find the combination that maximizes the combined SNR.
    This function is now much cleaner as it uses the core utility functions.
    """
    # 1. Setup time intervals using the dedicated function
    if config_dict is None:
        src_range, back_intervals, trange_total, source_par_dict = define_time_intervals(trigger_number, trigger_directory)
    else:
        src_range = config_dict["src_range"]
        back_intervals = config_dict["back_intervals"]
        trange_total = config_dict["trange_total"]
        source_par_dict = None

    # 2. Get list of all available TTE files
    all_tte_files, _ = trigger_file_list(trigger_directory, "tte", trigger_number)

    # 3. Rank individual detectors by SNR
    print("\n--- Ranking individual detectors ---")
    detector_snrs = []
    for tte_file in all_tte_files:
        # Generate a light curve just for SNR calculation
        _, lc_total, _, _, _ = generate_lightcurves([tte_file], src_range, trange_total, bw)
        snr = calculate_snr(lc_total[0], back_intervals)
        name = os.path.basename(tte_file)
        print(f"  {name}: SNR = {snr:.2f}")
        detector_snrs.append({'file': tte_file, 'name': name, 'snr': snr})
    
    ranked_detectors = sorted(detector_snrs, key=lambda x: x['snr'], reverse=True)

    # 4. Iteratively combine ranked detectors to find the peak SNR
    print("\n--- Finding optimal combination ---")
    snr_evolution = []
    for k in range(1, len(ranked_detectors) + 1):
        files_to_combine = [d['file'] for d in ranked_detectors[:k]]
        _, _, _, lc_combined_total, _ = generate_lightcurves(files_to_combine, src_range, trange_total, bw, combined=True)
        combined_snr = calculate_snr(lc_combined_total, back_intervals)
        snr_evolution.append({'k': k, 'snr': combined_snr})
        print(f"  Combined {k} detectors... New SNR = {combined_snr:.2f}")

    # 5. Identify the best set of detectors
    best_result = max(snr_evolution, key=lambda x: x['snr'])
    best_k = best_result['k']
    optimal_files = [d['file'] for d in ranked_detectors[:best_k]]
    optimal_names = [d['name'] for d in ranked_detectors[:best_k]]
    
    print(f"\nMaximum SNR of {best_result['snr']:.2f} found with the top {best_k} detectors.")
    
    # Return everything needed for the final report/plots
    if plot_flag:
        return optimal_files, snr_evolution, ranked_detectors, src_range, back_intervals, trange_total, source_par_dict
    else:
        return optimal_files, snr_evolution, src_range, back_intervals, trange_total, source_par_dict





def plot_gbm_lightcurves(trigger_directory, tte_files, src_range, back_intervals, bw=0.064, suffix=""):
    """
    Generates and saves two grid plots of GBM light curves (shared and independent y-axes).
    
    MODIFICATIONS:
    - Calls setup_publication_style() for professional aesthetics.
    - Saves plots in PDF format for lossless scaling.
    - Simplified layout handling.
    """
    # --- NEW: Set the style for this plot ---
    setup_publication_style()

    trange_total = (back_intervals[0][0], back_intervals[1][1])
    lcs_src, lcs_total, lc_comb_src, lc_comb_total, titles = generate_lightcurves(
        tte_files, src_range, trange_total, bw
    )

    num_plots = len(lcs_src) + (1 if lc_comb_src else 0)
    cols, rows = 4, int(np.ceil(num_plots / 4))

    # Internal helper to draw on axes
    def _draw_plots(axes, is_shared_y):
        ax_flat = axes.flatten()
        for i, (lc_s, lc_t, title) in enumerate(zip(lcs_src, lcs_total, titles)):
            ax = ax_flat[i]
            # Use detector name for title, e.g., 'n0' from 'glg_tte_n0_bn...'
            det_name = title.split('_')[2]
            ax.step(lc_s.centroids, lc_s.counts, where='post')
            #ax.set_title(f'Detector {det_name.upper()}')
            ax.grid(True, linestyle='--', alpha=0.5)
            snr = calculate_snr(lc_t, back_intervals)
            ax.text(0.95, 0.95, f"Det: {det_name.lower()}\nSNR: {snr:.1f}", transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8), fontsize=12)

        if lc_comb_src:
            ax = ax_flat[len(lcs_src)]
            ax.step(lc_comb_src.centroids, lc_comb_src.counts, where='post', color='black', linewidth=2.0)
            ax.set_title('Combined')
            ax.grid(True, linestyle='--', alpha=0.5)
            snr = calculate_snr(lc_comb_total, back_intervals)
            ax.text(0.95, 0.95, f"SNR: {snr:.1f}", transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8), fontsize=12)

        for j in range(num_plots, len(ax_flat)):
            ax_flat[j].axis('off')

    # Plot 1: Shared Y-Axes
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=True)
    _draw_plots(axes1, is_shared_y=True)
    fig1.supxlabel(f'Time since trigger (s)')
    fig1.supylabel('Counts / bin')
    fig1.suptitle(f'Detector Light Curves (Shared Y, {bw * 1000:.1f} ms){suffix}')
    # MODIFICATION: Save as PDF
    output_path1 = Path(trigger_directory) / f"lc_grid_shared_y{suffix}_{bw * 1000:.0f}ms.pdf"
    fig1.savefig(output_path1)
    plt.close(fig1)
    print(f"✅ Saved shared-axis plot: \n   {output_path1}")

    # Plot 2: Independent Y-Axes
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=False)
    _draw_plots(axes2, is_shared_y=False)
    fig2.supxlabel(f'Time since trigger (s)')
    fig2.supylabel('Counts / bin')
    fig2.suptitle(f'Detector Light Curves (Independent Y, {bw * 1000:.1f} ms){suffix}')
    # MODIFICATION: Save as PDF
    output_path2 = Path(trigger_directory) / f"lc_grid_indep_y{suffix}_{bw * 1000:.0f}ms.pdf"
    fig2.savefig(output_path2)
    plt.close(fig2)
    print(f"✅ Saved independent-axis plot: \n   {output_path2}")


def full_analysis_workflow(trigger_num, trigger_dir, bin_width=0.064, config_dict=None):
    """
    A single function to run the entire analysis and plotting workflow.
    """
    trigger_dir = Path(trigger_dir)

    best_detector_files, snr_results, src_range, back_intervals, trange_total, source_par_dict = find_optimal_detectors(
        trigger_num, trigger_dir, bw=bin_width, config_dict=config_dict
    )
    
    print("\n--- Final Result ---")
    print(f"Optimal detectors: {[os.path.basename(f)[8:10] for f in best_detector_files]}\n")

    plot_gbm_lightcurves(
        trigger_dir,
        best_detector_files,
        src_range,
        back_intervals,
        bw=bin_width,
        suffix="_optimal"
    )

    # --- MODIFICATION: SNR evolution plot ---
    # 1. Set the publication style
    setup_publication_style()
    
    k_values = [res['k'] for res in snr_results]
    snr_values = [res['snr'] for res in snr_results]

    # 2. Create figure and axes objects for more control
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, snr_values, 'o-', color='royalblue', label='Combined SNR')
    ax.set_title(f'SNR Evolution for GRB {trigger_num}')
    ax.set_xlabel('Number of Detectors (Ranked by Individual SNR)')
    ax.set_ylabel('Combined Signal-to-Noise Ratio (SNR)')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.5)
    
    best_k_result = max(snr_results, key=lambda x: x['snr'])
    best_k = best_k_result['k']
    best_snr = best_k_result['snr']
    
    # 3. Highlight the maximum point more clearly
    ax.axvline(best_k, color='crimson', linestyle='--', label=f'Optimal k={best_k} (SNR={best_snr:.2f})')
    ax.plot(best_k, best_snr, 'o', markersize=12, color='crimson', fillstyle='none', markeredgewidth=2)
    ax.legend()
    
    # 4. Save as PDF
    snr_evo_path = trigger_dir / f"snr_evolution_bn{trigger_num}_{int(bin_width*1000)}ms.pdf"
    fig.savefig(snr_evo_path)
    plt.close(fig)

    print(f"✅ SNR evolution plot saved to: \n   {snr_evo_path}")
    
    return best_detector_files, source_par_dict



def full_analysis_workflow(trigger_num, trigger_dir, bin_width=0.064, config_dict=None):
    """
    A single function to run the entire analysis and plotting workflow.
    """
    trigger_dir = Path(trigger_dir)

    # MODIFICATION: Capture the new 'ranked_detectors' output
    best_detector_files, snr_results, ranked_detectors, src_range, back_intervals, trange_total, source_par_dict = find_optimal_detectors(
        trigger_num, trigger_dir, bw=bin_width, config_dict=config_dict, plot_flag=True
    )
    
    print("\n--- Final Result ---")
    print(f"Optimal detectors: {[os.path.basename(f)[8:10] for f in best_detector_files]}\n")

    plot_gbm_lightcurves(
        trigger_dir,
        best_detector_files,
        src_range,
        back_intervals,
        bw=bin_width,
        suffix="_optimal"
    )

    # --- ENHANCED SNR EVOLUTION PLOT ---
    setup_publication_style()
    
    # --- Data Preparation ---
    k_values = [res['k'] for res in snr_results]
    combined_snr_values = [res['snr'] for res in snr_results]
    
    # NEW: Extract individual SNRs and detector names for plotting
    individual_snr_values = [d['snr'] for d in ranked_detectors]
    # Extract short names like 'n0', 'n1', etc.
    detector_names = [os.path.basename(d['file']).split('_')[2].upper() for d in ranked_detectors]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # NEW: Plot individual SNRs as a bar chart in the background
    ax.bar(k_values, individual_snr_values, color='gray', alpha=0.5, 
           label='Individual Detector SNR')

    # Plot the cumulative SNR as a line on top
    ax.plot(k_values, combined_snr_values, 'o-', color='royalblue', 
            linewidth=2.5, markersize=8, label='Cumulative Combined SNR')

    # --- Aesthetics and Highlighting ---
    #ax.set_title(f'SNR Evolution for GRB {trigger_num}')
    ax.set_xlabel('Detectors Added in Order of Rank')
    ax.set_ylabel('Signal-to-Noise Ratio (SNR)')
    
    # NEW: Use detector names for the x-axis labels
    ax.set_xticks(k_values)
    ax.set_xticklabels(detector_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.5, axis='y') # Grid on y-axis only for clarity

    # Highlight the maximum point
    best_k_result = max(snr_results, key=lambda x: x['snr'])
    best_k = best_k_result['k']
    best_snr = best_k_result['snr']
    
    ax.axvline(best_k, color='crimson', linestyle='--', 
                label=f'Optimal k={best_k} (SNR={best_snr:.2f})')
    ax.plot(best_k, best_snr, 'o', markersize=12, color='crimson', 
            fillstyle='none', markeredgewidth=2)
    
    ax.legend()
    
    # --- Saving ---
    snr_evo_path = trigger_dir / f"snr_evolution_bn{trigger_num}_{int(bin_width*1000)}ms.pdf"
    fig.savefig(snr_evo_path)
    plt.close(fig)

    print(f"✅ Enhanced SNR evolution plot saved to: \n   {snr_evo_path}")
    
    return best_detector_files, source_par_dict












def fit_background(phai_all, erange, src_interval, bkgd_intervals, outpath, trigger_number):

    lc_tot = phai_all.to_lightcurve(energy_range=erange)

    src_lc = phai_all.to_lightcurve(time_range=src_interval, energy_range=erange)

    bkg_lc_plot1 = phai_all.to_lightcurve(time_range=bkgd_intervals[0], energy_range=erange)
    bkg_lc_plot2 = phai_all.to_lightcurve(time_range=bkgd_intervals[1], energy_range=erange)

    backfitter = BackgroundFitter.from_phaii(phai_all, Polynomial, time_ranges=bkgd_intervals)
    backfitter.fit(order=1)

    best_order = find_best_poly_order(backfitter, energy_range=None, det='n1', max_order=4, outpath=outpath)
    

    backfitter.fit(order=best_order)
    #plt.show()

    bkgd_fit = backfitter.interpolate_bins(phai_all.data.tstart, phai_all.data.tstop)
    bkgd_fit_lc = bkgd_fit.integrate_energy(*erange)
    lcplot = Lightcurve(data=lc_tot, background=bkgd_fit_lc)
    lcplot.add_selection(src_lc)
    lcplot.add_selection(bkg_lc_plot1)
    lcplot.add_selection(bkg_lc_plot2)
    #lcplot.color = 'y'
    lcplot.selections[0].color = 'green'
    lcplot.selections[1].color = 'pink'
    lcplot.selections[2].color = 'pink'
    fig_name = outpath + f"selection_bn{trigger_number}.png"
    plt.savefig(fig_name)
    print(f"Selection plot saved to {fig_name}")
    #plt.show()
    plt.close()
    return backfitter






def run_single_simulation(sim_index, base_counts, bin_width_s, haar_python_path, time_resolved, t_start=0.0, window_size_s=None, step_size_s=None):
    """
    Runs a single MVT simulation instance. This function is designed to be
    run in a separate process.
    """
    try:
        # Ensure each process has a different random seed for reproducibility
        np.random.seed(sim_index + 1)
        
        # Generate counts for this specific simulation
        counts = np.random.poisson(base_counts)

        if not time_resolved:
            # print(f"Running MVT for sim {sim_index} (bin: {bin_width_ms} ms)") # Optional: for debugging
            mvt_res = run_mvt_in_subprocess(
                counts,
                bin_width_s=bin_width_s,
                haar_python_path=haar_python_path
            )
        else:
            # print(f"Running time-resolved MVT for sim {sim_index} (bin: {bin_width_ms} ms)") # Optional
            mvt_res = run_mvt_in_subprocess(
                counts=counts,
                bin_width_s=bin_width_s,
                haar_python_path=haar_python_path,
                time_resolved=True,
                window_size_s=window_size_s,
                step_size_s=step_size_s,
                tstart=t_start
            )
        
        # It's good practice to close plots inside the worker to prevent memory issues
        plt.close('all')
        
        return mvt_res
    except Exception as e:
        print(f"Error in simulation {sim_index} for bin width {bin_width_s*1000} ms: {e}")
        # Return a value indicating failure, which you can filter out later
        return None 


# This is a new helper function to create a config on the fly
def create_default_config(trigger_number):
    """Generates a default configuration dictionary for a given trigger number."""
    print(f"No config file provided. Generating a default configuration for bn{trigger_number}.")
    return {
        'trigger_number': trigger_number,
        # Most other parameters will be handled by .get() in main,
        # but you can set explicit defaults here if you want.
        'det_list': None, 
        'time_resolved': False,
        'total_sim': 30,
        'bin_width_ms': 1.0
    }



def main(config_dic, config_flag = False):

    with open(MVT_CONFIG_FILE, 'r') as f:
        MVT_config = yaml.safe_load(f)

    haar_python_path = MVT_config['project_settings']['haar_python_path']

    trigger_number = config_dic['trigger_number']
    en_lo = config_dic.get('en_lo', 8)
    en_hi = config_dic.get('en_hi', 900)

    outpath = config_dic.get('output_path', Path.cwd())
    data_path = config_dic.get('data_path', Path.cwd())
    T0 = config_dic.get('T0', 0)
    t_start = config_dic.get('tstart', 0)
    t_stop = config_dic.get('tstop', 0)
    T90 = config_dic.get('T90', 0)
    #config_dets = config_dic['det_list']
    back_intervals = config_dic.get('background_intervals', 0)
    print(f"Background intervals from config: {back_intervals}")
    time_resolved = config_dic.get('time_resolved', False)
    bin_width_ms = config_dic.get('bin_width_ms', 0.1)
    mvt_time_window = config_dic.get('mvt_time_window', 0.5)
    mvt_step_size = config_dic.get('mvt_step_size', 0.5)
    total_sim = config_dic.get('total_sim', 100)


    print(f"Trigger number: {trigger_number}")
    trigger_directory = Path(data_path) / f"bn{trigger_number}"
    det_list, _ = config_to_det_list(config_dic, trigger_directory)

    energy_range_nai = (en_lo, en_hi)
    
    if not trigger_directory.exists():
        print(f"Trigger directory {trigger_directory} does not exist.")
        print("Downloading necessary data files...")
        download_data(trigger_number, trigger_directory, flags={'tte':True, 'rsp':True, 'cat':True, 'trigdat':True})

    output_path = Path(outpath) / f"MVT_bn{trigger_number}/"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output path: {output_path}")
    #exit(1)

    
    print(f"Configured detectors: {det_list}")
    #exit(1)
    if det_list is None or det_list == []:
        print(f"No valid detectors found from config. Using 'best' detectors based on SNR.")
        tte_list_path, source_par_dict = full_analysis_workflow(trigger_number, trigger_directory, bin_width=0.064)
        det_list = [os.path.basename(f)[8:10] for f in tte_list_path]
    else:
        tte_list_path, _ = trigger_file_list(trigger_directory, "tte", trigger_number, det_list)
    
    #print(tte_list_path)

    print(f"\n\n--------- Starting MVT analysis for Trigger {trigger_number} ---------")
    if T90 is None or T90 <= 0:
        try:
            print("Fetching GRB parameters for T90...")
            T0, T90, T50, PF64, PFLX, FLU  = get_GRB_par(trigger_number, trigger_directory)
        #config_dic['T90'] = T90
        except:
            print("Failed to get T90 from BCAT.!!!")
            exit(1)
    else:
        print(f"Using T90: {T90} from config file")
 

    #t_start = -2 # min(config_dic['T0'], T0)-1
    if t_start == t_stop == 0:
        t_start = T0 - 0.5*min(5,T90)-1
        t_stop = T0 + T90*1.2+2
    else:
        bin_width_ms = config_dic['bin_width_ms']


    src_interval = [t_start, t_stop]
    print(f"T0: {T0}, T90: {T90}")
    print(f"Source interval: {src_interval}")
    det_string = "".join(det_list) if det_list else "not_specified"
    selection_str = f"{round(t_start, 2)}_{round(t_stop, 2)}s_{det_string}"

    

    if back_intervals in [0, [[0,0],[0,0]], None, 'none', 'None', 'NONE', 'no', 'No', 'NO']:
        print("Using default background intervals based on T90.")
        # Define parameters for the parameter file
        if T90 < 2.0:
            padding = 0.256*10+T90
            back_before = [t_start-padding-50, t_start-padding]
            back_after = [t_stop+padding*2, t_stop+padding*2+50]
        elif 2.0<=T90 < 6.0:
            padding = 1.024*10+T90*.2
            reference = min(110,max(50.0,T90))+padding
            back_before = [t_start-padding-50, t_start-padding]
            back_after = [t_stop+padding, t_stop+reference]
        else:
            padding = 1.024*10+T90*.2
            r_padding  = 1.024*10+T90*.4
            reference = min(110,max(50.0,T90))+padding
            r_reference = min(110,max(50.0,T90))+r_padding
            back_before = [t_start-reference, t_start-padding]
            back_after = [t_stop+r_padding, t_stop+r_reference]

        back_intervals = np.around(np.array([back_before, back_after]),4).tolist()

    trange = [back_intervals[0][0]-10, back_intervals[1][1]+10]
   


    try:
        print(f"Creating LC with bin width: {bin_width_ms} ms")
        bin_width_s = bin_width_ms / 1000.0
        _, _, lc_total, _, _ = generate_lightcurves(tte_list_path, src_interval, trange, bin_width_s, energy_range_nai, combined=True, src_only=True)
    
        print(f"Lightcurve generated with bin width {bin_width_ms} ms for detectors: {det_list}")

        task_function = functools.partial(
            run_single_simulation,
            base_counts=lc_total.counts,
            bin_width_s=bin_width_s,
            haar_python_path=haar_python_path,
            time_resolved=time_resolved,
            window_size_s=mvt_time_window if time_resolved else None,
            step_size_s=mvt_step_size if time_resolved else None,
            t_start=t_start
        )

        # --- STEP 3: Run the tasks in parallel using ProcessPoolExecutor ---
        if time_resolved:
            print(f"\n@@@@@@@@@@@ Starting {total_sim}, time-resolved simulations @@@@@@@@@@@\n \t window size: {mvt_time_window}s, step size: {mvt_step_size}s, bin width: {bin_width_ms}ms ")
        else:
            print(f"\n----- Starting {total_sim} parallel simulations bin width: {bin_width_ms}ms -----\n \t Time range: {t_start} to {t_stop}")
        MVT_time_resolved_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # executor.map applies 'task_function' to each item in 'range(total_sim)'
            # It distributes these tasks among the available CPU cores.
            results_iterator = executor.map(task_function, range(total_sim))
            
            # Collect the results as they complete
            MVT_time_resolved_results = [
            res for res in tqdm(
                results_iterator, 
                total=total_sim, 
                desc="Processing Simulations",
                unit="sim"
            ) if res is not None
        ]

        print("All parallel simulations completed.")
    except Exception as e:
        print(f"Error during MVT calculation for bin width {bin_width_ms} ms: {e}")
        # Reset results if the main setup fails
        MVT_time_resolved_results = []

    output_info = {'trigger_number': trigger_number, 'file_path': output_path, 'selection_str': selection_str}


    if time_resolved:
        mvt_summary_df = analysis_mvt_time_resolved_results_to_dataframe(MVT_time_resolved_results, output_info, bin_width_ms, total_sim)

        lc_bw = 0.001
        _, _, lc_plot, _, _ = generate_lightcurves(tte_list_path, src_interval, trange, lc_bw, energy_range_nai, combined=True, src_only=True)

        create_final_GBM_plot_with_MVT(
        lc_plot,
        t_start,
        t_stop,
        output_info,
        bin_width_ms,
        mvt_time_window,
        mvt_summary_df
        )
    else:
        mvt_res = run_mvt_in_subprocess(
                lc_total.counts,
                bin_width_s=bin_width_s,
                haar_python_path=haar_python_path,
                doplot=1,
                file_name = str(output_path) + f"/mvt_bn{trigger_number}.png",

            )
        print("\nOriginal MVT Results:--------")
        for key, value in mvt_res.items():
            print(f"{key}: {value}") 
        
        print("\nMVT Summary from all simulations:--------")
        mvt_summary, _ = analysis_mvt_results_to_dataframe(MVT_time_resolved_results, output_info, bin_width_ms, total_sim)
        for key, value in mvt_summary.items():
            print(f"{key}: {value}")
        print("\n--- Saving final configuration for reproducibility ---")

        mvt_s = mvt_summary['median_mvt_ms']/1000
        _, _, _, lc_snr, _ = generate_lightcurves(tte_list_path, src_interval, trange, mvt_s, energy_range_nai, combined=True)
        SNR_mvt = round(calculate_snr(lc_snr, back_intervals), 2)
        print(f"SNR_mvt: {SNR_mvt}")
        

        mvt_summary_all = {**mvt_res, 'SNR_mvt': SNR_mvt, **mvt_summary}

        mvt_all_summary_path = output_path / f"mvt_summary_bn{trigger_number}_{selection_str}_{(bin_width_ms)}ms.yaml"
        write_yaml(mvt_summary_all, mvt_all_summary_path)

        #full_analysis_workflow(trigger_number, trigger_directory, bin_width=mvt_s, config_dict={'src_range': src_interval, 'back_intervals': back_intervals, 'trange_total': trange})

    # Update the dictionary with any calculated values
    config_dic['tstart'] = t_start
    config_dic['tstop'] = t_stop
    config_dic['background_intervals'] = back_intervals
    config_dic['det_list'] = det_list # Saves the detectors that were actually used
    config_dic['T90'] = T90 # Ensure the fetched T90 is saved
    config_dic['T0'] = T0   # Ensure the fetched T0 is saved
    config_dic['en_lo'] = en_lo
    config_dic['en_hi'] = en_hi
    config_dic['total_sim'] = total_sim
    config_dic['time_resolved'] = time_resolved
    config_dic['bin_width_ms'] = bin_width_ms
    #if time_resolved:
    config_dic['mvt_time_window'] = mvt_time_window
    config_dic['mvt_step_size'] = mvt_step_size
    #config_dic.update(source_par_dict) # Add GRB parameters if available


    config_dic['det_string'] = det_string
    config_dic['source_interval'] = src_interval
    config_dic['energy_range'] = [en_lo, en_hi]
    config_dic['mvt_summary'] = mvt_summary_all if not time_resolved else mvt_summary_df.to_dict()



    # Define the output filename
    #final_config_path = output_path / f"config_MVT_{trigger_number}.yaml"
    #final_config_path = output_path / f"config_MVT_{trigger_number}_{det_string}_{(bin_width_ms)}ms.yaml"
    if config_flag:
        final_config_path = output_path / f"config_MVT_{trigger_number}_{det_string}_sim{total_sim}_{(bin_width_ms)}ms.yaml"
    else:
        config_dic['email_flag'] = False
        final_config_path = trigger_directory / f"config_MVT_{trigger_number}.yaml"
    # Write the dictionary to the YAML file
    write_yaml(config_dic, final_config_path)
        
    print(f"Final configuration saved to: {final_config_path}")
    return final_config_path




# This block is completely new and handles the command-line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run MVT analysis for a Fermi GBM trigger. "
                    "Provide either a trigger number or a config file."
    )
    # This group makes the arguments mutually exclusive (you can only use one)
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        '-c', '--config',
        type=str,
        help="Path to the simulation YAML configuration file."
    )
    group.add_argument(
        '-bn', '--trigger_number',
        type=str,
        help="Trigger number (e.g., '080916009'). A default config will be generated."
    )
    
    args = parser.parse_args()
    
    email_flag = False
    config = None
    if args.config:
        # If -c is used, load the specified file
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            email_flag = config.get('email_flag', False) # Default to False if not specified
        config_flag = True
        email_body = f"The MVT analysis for trigger {config['trigger_number']} is complete."
    elif args.trigger_number:
        # If -bn is used, create a default config in memory
        config = create_default_config(args.trigger_number)
        config_flag = False
        email_body = f"The MVT analysis for trigger {args.trigger_number} is complete."

    # Call the main function with the loaded or generated config dictionary
    if config:
        mvt_result_path = main(config, config_flag)
    else:
        print("Error: Could not create or load a configuration. Exiting.")

    #email_body = f"The MVT analysis for trigger {trigger_number} is complete."
    if email_flag:
        send_email(
            subject=f"Button!! Analysis Complete for {config['trigger_number']}:",
            body=email_body,
            attachment_path=mvt_result_path
     )



      
"""

if __name__ == '__main__':
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run MVT analysis based on a YAML configuration file."
    )
    # Add one argument: the path to the config file
    parser.add_argument(
        'config_file',
        type=str,
        # NEW: Make the argument optional and provide a default value
        nargs='?',
        default=CONFIG_FILE,
        help="Path to the simulation YAML configuration file. "
             f"Defaults to '{CONFIG_FILE}' if not provided."
    )
    
    # Parse the arguments provided by the user from the command line
    args = parser.parse_args()
    
    # Call the main function, passing in the path to the config file
    main(args.config_file)
"""

