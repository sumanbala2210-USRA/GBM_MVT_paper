import matplotlib.pyplot as plt
import os
import shutil
import yaml
import logging
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any
# ========= Import necessary libraries =========
import itertools
from matplotlib.lines import Line2D

# Assuming these are in a local library file as in the original script
from SIM_lib import _parse_param, e_n, _create_param_directory_name, PULSE_MODEL_MAP, SIMULATION_REGISTRY, write_yaml, convert_det_to_list, send_email, complex_pulse_list
from TTE_SIM_v2 import generate_function_events, calculate_adaptive_simulation_params, create_final_gbm_plot, generate_gbm_events_dets, sim_gbm_name_format, print_nested_dict
from TTE_SIM_v2 import Function_MVT_analysis, check_param_consistency, flatten_dict, GBM_MVT_analysis_det, GBM_MVT_analysis_complex, Function_MVT_analysis_complex
from sim_functions import constant

SIM_CONFIG_FILE = 'simulations_ALL.yaml'
# This is a placeholder as the file is not provided; the script uses hardcoded params instead.
# with open(SIM_CONFIG_FILE, 'r') as f:
#         config = yaml.safe_load(f)


def create_final_plot(
    source_events: np.ndarray,
    background_events: np.ndarray,
    model_info: Dict,
    output_info: Dict
):
    """
    A self-contained function that takes raw event data, creates a
    final, styled plot, and saves the plotting data to an NPZ file.
    """
    try:
        # --- 1. Unpack all necessary data and parameters ---
        params = model_info['base_params']
        fig_name = output_info['file_path'] / f"LC_{output_info['file_name']}.png"
        t_start, t_stop = params['t_start'], params['t_stop']
        background_level_cps = params['background_level'] * params.get('scale_factor', 1.0)
        
        # --- 2. Prepare Data for Plotting (Binning) ---
        total_events = np.sort(np.concatenate([source_events, background_events]))
        bin_width = params.get('bin_width_for_plot', 0.01)
        bins = np.arange(t_start, t_stop + bin_width, bin_width)
        times = bins[:-1] + bin_width / 2.0
        total_counts, _ = np.histogram(total_events, bins=bins)
        source_only_counts, _ = np.histogram(source_events, bins=bins)
        ideal_background_counts = background_level_cps * bin_width

        # --- [NEW] SAVE DATA TO NPZ FILE ---
        # Create a dedicated directory for the data files
        data_path = output_info['file_path'] / 'plot_data'
        data_path.mkdir(exist_ok=True, parents=True)
        data_file_path = data_path / f"DATA_{output_info['file_name']}.npz"

        #print(params)
        # Save all necessary arrays and the parameters dictionary
        np.savez_compressed(
            data_file_path,
            times=times,
            total_counts=total_counts,
            source_only_counts=source_only_counts,
            ideal_background_counts=np.array(ideal_background_counts),
            parameters=params  # Save the whole dictionary
        )
        print(f"Plotting data successfully saved to: {data_file_path}")
        print(f"DATA_{output_info['file_name']}.npz")
        # ------------------------------------

        pulse_shape_name_dict = {
            'complex_pulse_short': 'Short',
            'complex_pulse_long': 'Long',
            'complex_pulse_short_2p10ms': 'Short + Fixed 10ms (at 4.95s)',
            'complex_pulse_short_2p3ms': 'Short + Fixed 3ms (at 4.95s)',
        }
        pulse_shape = params['pulse_shape']
        pulse_shape_nice = pulse_shape_name_dict.get(pulse_shape, pulse_shape)

        plt.rcParams.update({
            'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14,
            'ytick.labelsize': 14, 'legend.fontsize': 12, 'legend.title_fontsize': 14,
            'font.family': 'serif'
        })
        sigma = params.get('sigma', 'N/A')
        position_shift = params.get('position', 'N/A')
        annotation_text = f"{pulse_shape_nice}\n" \
                          f"Varying {int(sigma*1000)} ms (at {position_shift}s)\n" \
                          f"Overall Amp: {params['overall_amplitude']}\n" \
                          f"Peak Amp Ratio: {params['peak_amp_ratio']}"

        # --- 3. Define the plot data ---
        plot_data = [
            {'x': times, 'y': total_counts, 'label': 'Total Signal (Simulated)', 'color': 'rosybrown', 'fill_alpha': 0.6},
            {'x': times, 'y': source_only_counts, 'label': 'Source Signal (Simulated)', 'color': 'darkgreen', 'fill_alpha': 0.4}
        ]
        h_lines = [{'y': ideal_background_counts, 'label': f'Ideal Background ({background_level_cps:.1f} cps)', 'color': 'k'}]
        ylabel = f"Counts per {bin_width*1000:.1f} ms Bin"
            
        # --- 4. Create the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        for data in plot_data:
            ax.step(data['x'], data['y'], where='mid', label=data.get('label'), color=data.get('color'), lw=data.get('lw', 1.5))
            if 'fill_alpha' in data:
                ax.fill_between(data['x'], data['y'], step="mid", color=data.get('color'), alpha=data.get('fill_alpha'))
        for line in h_lines:
            ax.axhline(y=line['y'], color=line.get('color'), linestyle='--', label=line.get('label'))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='lightgray', frameon=True)
        ax.set_xlim(t_start, t_stop)
        ax.set_ylim(bottom=0)

        if annotation_text:
            props = dict(boxstyle='round,pad=0.5', facecolor='lightgoldenrodyellow', alpha=0.7, edgecolor='lightgray')
            ax.text(0.4, 0.97, annotation_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left', bbox=props, fontfamily='serif', fontsize=16)
        
        fig.tight_layout()
        plt.savefig(fig_name, dpi=300)
        plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"Failed to generate representative plot. Error: {e}")
        pass

def Function_plot_complex(input_info: Dict, output_info: Dict):
    """
    Performs MVT analysis by assembling a pre-generated template and feature,
    and calculates a detailed breakdown of source counts and SNR for each component.
    """
    base_params = input_info['base_params']
    feature_amplitude = base_params['peak_amplitude']
    position_shift = base_params['position']
    t_start_analysis = base_params.get('t_start')
    t_stop_analysis = base_params.get('t_stop')

    try:
        template_events_all = input_info['src_events']
        background_events_all = input_info['bkgd_events']
        feature_events_all = input_info['pulse_events']
    except (FileNotFoundError, IndexError) as e:
        logging.error(f"Error loading source/background files for assembly: {e}")
        return [], 0

    snr_timescales = [1]
    template_events = template_events_all[(template_events_all >= t_start_analysis) & (template_events_all <= t_stop_analysis)]
    feature_events = feature_events_all
    background_events = background_events_all[(background_events_all >= t_start_analysis) & (background_events_all <= t_stop_analysis)]
    shifted_feature_events_all = feature_events + position_shift
    shifted_feature_events = shifted_feature_events_all[(shifted_feature_events_all >= t_start_analysis) & (shifted_feature_events_all <= t_stop_analysis)]
    complete_src_events = np.sort(np.concatenate([template_events, shifted_feature_events]))
    
    create_final_plot(source_events=complete_src_events, background_events=background_events, model_info={ 'func': None, 'func_par': None, 'base_params': base_params, 'snr_analysis': snr_timescales }, output_info=output_info)

def main(oa, pa, pos, sigma, shape):
    common_values = {
        'background_level': 1,
        'scale_factor': 1000,
        'random_seed': 37,
        'grid_resolution': 1e-05,
        'total_sim': 10,
        'center_time': 0.0,
    }
    tstop = 15
    if shape == 'complex_pulse_long':
        tstop = 15

    params = {
        'peak_amplitude': 1.0,
        'position': pos,
        'overall_amplitude': oa,
        't_start': 4,
        't_stop': tstop,
        'pulse_shape': shape,
        'sigma': sigma,
        'peak_amp_ratio': pa,
        **common_values
    }

    pulse_shape = params['pulse_shape']
    func_to_use, required_params = PULSE_MODEL_MAP[pulse_shape]
    adaptive_params = calculate_adaptive_simulation_params(pulse_shape, params)
    params.update(adaptive_params)
    func_par = tuple(params[key] for key in required_params)
    sim_params = {**params, 'random_seed': 37}

    output_path = Path.cwd() / 'Lc_collections'
    output_path.mkdir(exist_ok=True, parents=True)
    file_name = f"{oa}_{pa}_{pos}_{shape}"

    output_info = {'file_name': file_name, 'file_path': output_path, 'file_info': 'test2'}

    source_events, back_events = generate_function_events(
        func=func_to_use, func_par=func_par,
        back_func=constant, back_func_par=(params['background_level'],),
        params=sim_params
    )

    pulse_params = {
        'peak_amplitude': pa * params['overall_amplitude'],
        'sigma': sigma,
        't_start': 4,
        't_stop': 12,
        'pulse_shape': 'gaussian',
        **common_values
    }
    pulse_shape = pulse_params['pulse_shape']
    func_to_use, required_params = PULSE_MODEL_MAP[pulse_shape]
    adaptive_params = calculate_adaptive_simulation_params(pulse_shape, pulse_params)
    pulse_params.update(adaptive_params)
    pulse_func_par = tuple(pulse_params[key] for key in required_params)
    print("Pulse func par:", pulse_func_par)
    pulse_sim_params = {**pulse_params, 'random_seed': 37}
    pulse_source_events, _ = generate_function_events(
        func=func_to_use, func_par=pulse_func_par,
        back_func=constant, back_func_par=(pulse_params['background_level'],),
        params=pulse_sim_params
    )

    Function_plot_complex(
        input_info={'src_events': source_events, 'bkgd_events': back_events, 'pulse_events': pulse_source_events, 'base_params': params},
        output_info=output_info
    )
    file_saved = f"DATA_{output_info['file_name']}.npz"
    return file_saved
    # print("Length of pulse source events:", len(pulse_source_events))

if __name__ == "__main__":
    overall_amplitude = [5]  # [1, 5, 10, 50.0, 100.0]
    peak_amplitude = [3]  # [0.01, 1, 2, 3, 4, 5]
    position = [4.5]  # , 4.5]
    #shape_list = ['complex_pulse_long'] #, 'complex_pulse_short', 'complex_pulse_short_2p10ms', 'complex_pulse_short_2p3ms']
    shape_list = [ 'complex_pulse_short_2p10ms']#, 'complex_pulse_short_2p3ms']
    sigma_list = [0.003]  # , 30.0, 100.0]

    sim1 = ['complex_pulse_short_2p10ms', 0.03, 6.2, 10, 1]
    sim2 = ['complex_pulse_short_2p3ms', 0.01, 4.5, 100, 2]
    sim3 = ['complex_pulse_short', 0.003, 4.5, 5, 5]
    sim4 = ['complex_pulse_long', 0.01, 6.2, 1, 3]
    simulations_to_run = [sim4, sim3, sim2, sim1]

    file_list = []

    for i, sim in enumerate(simulations_to_run):
        print(f"Running simulation {i+1}/{len(simulations_to_run)}: {sim}")
        print(f"Running with: oa={sim[3]}, pa={sim[4]}, pos={sim[2]}, shape={sim[1]}, sigma={sim[0]}")
        file_saved = main(sim[3], sim[4], sim[2], sim[1], sim[0])
        file_list.append(file_saved)
    
    print("All simulations completed. Files saved:")
    print(file_list)

    #for oa, pa, pos, shape, sigma in itertools.product(overall_amplitude, peak_amplitude, position, shape_list, sigma_list):
    #    print(f"Running with: oa={oa}, pa={pa}, pos={pos}, shape={shape}, sigma={sigma}")
    #    main(oa, pa, pos, sigma, shape)