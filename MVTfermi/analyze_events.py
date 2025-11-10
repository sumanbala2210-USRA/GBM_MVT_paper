"""
# analyze_events.py
Suman Bala
17 Aug 2025: This script reads a directory of raw TTE event files,
            performs analysis (MVT, SNR) for various bin widths,
            and outputs a final summary CSV file.
"""

# ========= Import necessary libraries =========
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import yaml
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import sys

from SIM_lib import _parse_param, e_n, _create_param_directory_name, convert_det_to_list, write_yaml, complex_pulse_list


from TTE_SIM_v2 import Function_MVT_analysis, print_nested_dict, check_param_consistency, flatten_dict, GBM_MVT_analysis_det, GBM_MVT_analysis_complex, Function_MVT_analysis_complex



# ========= USER SETTINGS =========
#MAX_WORKERS = os.cpu_count() - 2
MAX_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() - 2))
SIM_CONFIG_FILE = 'simulations_ALL.yaml'
RESULTS_FILE_NAME = "final_summary_results.csv"





def _create_task(sim_type, base_path, params, settings, bin_width, haar_path, selections=None):
    """Helper function to generate analysis tasks for different detector selections."""
    if selections and sim_type == 'gbm':
        for selection in selections:
            yield {
                'param_dir_path': base_path,
                'base_params': params,
                'analysis_settings': settings,
                'bin_width_to_process_ms': bin_width,
                'dets_to_analyze': selection,
                'haar_python_path': haar_path
            }
    else:
        yield {
            'param_dir_path': base_path,
            'base_params': params,
            'analysis_settings': settings,
            'bin_width_to_process_ms': bin_width,
            'haar_python_path': haar_path
        }
                


def generate_analysis_tasks(config: Dict[str, Any]) -> 'Generator':
    """Generates all analysis tasks based on the YAML configuration."""
    data_path = Path(config['project_settings']['data_path'])
    haar_python_path = config['project_settings']['haar_python_path']
    analysis_settings = config['analysis_settings']
    bin_widths_to_analyze_ms = analysis_settings.get('bin_widths_to_analyze_ms', [])
    detector_selections = analysis_settings.get('detector_selections')
    pulse_definitions = config['pulse_definitions']

    for campaign in config.get('simulation_campaigns', []):
        if not campaign.get('enabled', False):
            continue 
        sim_type = campaign.get('type')
        pulses_to_run = campaign.get('pulses_to_run', [])
        
        #if 'complex_pulse' in pulses_to_run and isinstance(pulses_to_run, list):
        if any(p in complex_pulse_list for p in pulses_to_run):
            logging.warning("When 'complex_pulse' is included in 'pulses_to_run', it must be the only entry. Overriding to only run 'complex_pulses'.")
            pulses_to_run = [p for p in pulses_to_run if p in complex_pulse_list]
            # Ensure 'complex_pulse' is the only entry if present
            #pulses_to_run = ['complex_pulse']

        for pulse_config in pulses_to_run:
            pulse_shape = pulse_config if isinstance(pulse_config, str) else list(pulse_config.keys())[0]
            
            
            # --- TOP-LEVEL SWITCH for Assembly vs. Standard Mode ---
            if pulse_shape in complex_pulse_list:
                logging.info(f"  > Found '{pulse_shape}', preparing 'assemble-in-analysis' tasks...")
                
                # 1. Manually construct the path to the SINGLE pre-generated template directory.
                #    This is the critical fix: it uses ONLY the fixed canonical parameters.
                template_generation_params = {'peak_amplitude': 1.0, 'position': 0.0}
                if 'trigger_set' in campaign['parameters']:
                    template_generation_params['trigger_set'] = campaign['parameters']['trigger_set'][0]

                #template_dir_name = _create_param_directory_name(sim_type, pulse_shape, template_generation_params)
                #template_dir_path = data_path / sim_type / pulse_shape / template_dir_name

                #if not template_dir_path.exists():
                #    logging.error(f"Assembly Error: Template directory for complex_pulse not found at {template_dir_path}")
                #    continue

                # 2. Get the full grid of variable parameters for the feature from the YAML.
                base_pulse_config = pulse_definitions.get(pulse_shape, {})
                variable_params_config = campaign['parameters'].copy()
                variable_params_config.update(base_pulse_config.get('parameters', {}))
                
                param_names = list(variable_params_config.keys())
                param_values = [_parse_param(v) for v in variable_params_config.values()]
                total_sim_analysis = campaign['constants'].get('total_sim', 100)
                t_start = campaign['constants'].get('t_start', 0.0)
                t_stop = campaign['constants'].get('t_stop', 15.0)

                # 3. Create a task for every combination of the feature parameters.
                for combo in itertools.product(*param_values):
                    current_variable_params = dict(zip(param_names, combo))
                    base_params = {**current_variable_params, 'pulse_shape': pulse_shape, 'sim_type': sim_type, 'num_analysis': total_sim_analysis, 't_start_analysis': t_start, 't_stop_analysis': t_stop}
                    
                    # Safety checks for this workflow
                    base_dets = current_variable_params.get('trigger_set', {}).get('det')
                    if base_dets == 'all' and not detector_selections:
                        logging.error(f"Configuration Error: det='all' requires 'detector_selections' in the analysis YAML.")
                        sys.exit(1)
                    if detector_selections and base_dets != 'all' and sim_type == 'gbm':
                        logging.error(f"Configuration Error: 'detector_selections' is active, but simulation is for det='{base_dets}', not 'all'.")
                        sys.exit(1)
                    current_variable_params['peak_amplitude'] = 1.0
                    current_variable_params['position'] = 0.0
                    template_dir_name = _create_param_directory_name(sim_type, pulse_shape, current_variable_params)
                    template_dir_path = data_path / sim_type / pulse_shape / template_dir_name
                    #print("Template Dir Path:", template_dir_path)
                    #exit(1)

                    if not template_dir_path.exists():
                        logging.error(f"Assembly Error: Template directory for complex_pulse not found at {template_dir_path}")
                        continue

                    for bin_width_ms in bin_widths_to_analyze_ms:
                        yield from _create_task(sim_type, template_dir_path, base_params, analysis_settings, bin_width_ms, haar_python_path, detector_selections)
            
            else: # --- STANDARD MODE for all other simple pulses ---
                base_pulse_config = pulse_definitions.get(pulse_shape, {})
                variable_params_config = campaign['parameters'].copy()
                variable_params_config.update(base_pulse_config.get('parameters', {}))
                total_sim_analysis = campaign['constants'].get('total_sim', 100)
                param_names = list(variable_params_config.keys())
                param_values = [_parse_param(v) for v in variable_params_config.values()]

                for combo in itertools.product(*param_values):
                    current_variable_params = dict(zip(param_names, combo))
                    base_params = {**current_variable_params, 'pulse_shape': pulse_shape, 'sim_type': sim_type, 'num_analysis': total_sim_analysis}
                    
                    # Safety checks for this workflow
                    base_dets = current_variable_params.get('trigger_set', {}).get('det')
                    if base_dets == 'all' and not detector_selections:
                        logging.error(f"Configuration Error: det='all' requires 'detector_selections' in the analysis YAML.")
                        sys.exit(1)
                    if detector_selections and base_dets != 'all' and sim_type == 'gbm':
                        logging.error(f"Configuration Error: 'detector_selections' is active, but simulation is for det='{base_dets}', not 'all'.")
                        sys.exit(1)
                    
                    param_dir_name = _create_param_directory_name(sim_type, pulse_shape, current_variable_params)
                    param_dir_path = data_path / sim_type / pulse_shape / param_dir_name
                    
                    if param_dir_path.exists():
                        for bin_width_ms in bin_widths_to_analyze_ms:
                            yield from _create_task(sim_type, param_dir_path, base_params, analysis_settings, bin_width_ms, haar_python_path, detector_selections)




# ========= THE CORE WORKER FUNCTION =========
def analyze_one_group(task_info: Dict, data_path: Path, results_path: Path) -> List[Dict]:
    """
    Analyzes one group of NN event files, intelligently handling both 
    'function' (.npz) and 'gbm' (.fits) data products. For each group, it 
    creates a detailed per-realization CSV, a distribution plot, and 
    returns the final summary data.
    """
    # --- 1. Initial Setup ---
    param_dir = task_info['param_dir_path']
    base_params = task_info['base_params']
    analysis_settings = task_info['analysis_settings']
    haar_python_path = task_info['haar_python_path']
    bin_width = task_info['bin_width_to_process_ms']
    sim_type = base_params['sim_type']

    #print("Position:", )
    #exit()
    
    
    #print_nested_dict(base_params)


    pulse_shape = base_params['pulse_shape']

    ## ================== Directory Creation Logic ==================
    output_analysis_path = None
    if pulse_shape in complex_pulse_list:
        # Assembly Mode: Name the output directory after the variable feature parameters.
        extra_pulse_config = analysis_settings.get('extra_pulse', {})
        feature_pulse_shape = extra_pulse_config.get('pulse_shape', 'extra')
        feature_pulse_params = {k: v for k, v in extra_pulse_config.items() if k != 'pulse_shape'}
        base_params['peak_amp_relative'] = base_params['peak_amplitude']
        base_params['peak_amplitude'] = base_params['peak_amplitude']*base_params.get('overall_amplitude', 1.0)
        
        variable_params_for_name = {
            'peak_amplitude': base_params['peak_amplitude'],
            'position': base_params.get('position', 0.0),
            'overall_amplitude': base_params.get('overall_amplitude', 1.0)
        }
        # Use a placeholder shape like 'gaussian' for consistent naming
        results_dir_name = _create_param_directory_name(sim_type, pulse_shape, variable_params_for_name, extra_pulse=True)
        print("Results Dir Name:", results_dir_name)
        output_analysis_path = results_path / sim_type / pulse_shape / results_dir_name
    else:
        # Standard Mode: Mirror the input directory structure.
        relative_path = param_dir.relative_to(data_path)
        output_analysis_path = results_path / relative_path

    #print('Path:', output_analysis_path)

    output_analysis_path.mkdir(parents=True, exist_ok=True)
    ## =============================================================
    
    # Create a mirrored output directory in the results path
    #relative_path = param_dir.relative_to(data_path)
    #output_analysis_path = results_path / relative_path
    #output_analysis_path.mkdir(parents=True, exist_ok=True)

    sim_param_file = sorted(param_dir.glob('*.yaml'))
    sim_params = yaml.safe_load(open(sim_param_file[0], 'r'))

    #print("Base Parameters:---------")
    #print_nested_dict(base_params)
    #print("Type:", type(sim_params))
    #print("Simulation Parameters:---------")
    #print_nested_dict(sim_params)
    #print('\n')
    #exit(1)

    discrepancies = check_param_consistency(
            dict1=sim_params,
            dict2=flatten_dict(base_params),
            pulse_shape=base_params['pulse_shape']
    )
    if discrepancies:
        for msg in discrepancies:
            logging.error(f"Parameter mismatch for {param_dir.name}: {msg}")
        logging.warning(f"Skipping analysis for {param_dir.name} due to parameter mismatch.")
        return [] # Return empty list to skip
    

    det_selection = task_info.get('dets_to_analyze')
    #sim_det = sim_params.get('det')
    #base_det = base_params.get('trigger_set', {}).get('det')
    #analysis_det = convert_det_to_list(det_selection if det_selection and base_det == 'all' else base_det or sim_det)
    #selection_str = "-".join(analysis_det) if isinstance(analysis_det, list) else str(analysis_det or "default")



    #exit(0)
    selection_str = "default"

    analysis_input = {
        'sim_data_path': param_dir,
        'sim_par_file': sim_params,
        'base_params': base_params,
        'analysis_settings': analysis_settings,
        'bin_width_ms': bin_width,
        'haar_python_path': haar_python_path
    }

    # --- 2. Load Data and Perform Per-Realization Analysis ---
    iteration_results, NN = (None, 0)
    if sim_type == 'gbm':
        #det_analysis = convert_det_to_list(det_selection)

        base_dets = base_params['trigger_set']['det']
        if base_dets in ['all', ['all']]:
            analysis_det = convert_det_to_list(det_selection)
        else:
            analysis_det = convert_det_to_list(base_dets)

        #print_nested_dict(base_params)
        if isinstance(analysis_det, list):
            selection_str = "-".join(analysis_det)
        elif analysis_det: # If it's a string like 'n6' or 'all'
            selection_str = str(analysis_det)

        analysis_input['analysis_det'] = analysis_det
        analysis_input['base_det'] = base_dets

        if base_params['pulse_shape'] in complex_pulse_list:
            final_summary_list = GBM_MVT_analysis_complex(input_info=analysis_input,
                    output_info = { 'file_path': output_analysis_path,
                                'file_info': param_dir.name,
                                'selection_str': selection_str})
        else:
            final_summary_list = GBM_MVT_analysis_det(input_info=analysis_input,
                    output_info = { 'file_path': output_analysis_path,
                                'file_info': param_dir.name,
                                'selection_str': selection_str})

    else:
        base_dets = analysis_det = sim_params['det']
        analysis_input['analysis_det'] = analysis_det
        analysis_input['base_det'] = base_dets
        if base_params['pulse_shape'] in complex_pulse_list:
            final_summary_list = Function_MVT_analysis_complex(input_info=analysis_input,
            output_info={ 'file_path': output_analysis_path,
                         'file_info': param_dir.name,
                         'selection_str': selection_str})
        else:
         # sim_type == 'function'
            final_summary_list = Function_MVT_analysis(input_info=analysis_input,
                output_info={ 'file_path': output_analysis_path,
                            'file_info': param_dir.name,
                            'selection_str': selection_str})


    return final_summary_list


# ========= MAIN EXECUTION BLOCK (REFACTORED FOR MEMORY STABILITY) =========

def main(config_filepath: str):
    now = datetime.now().strftime("%y_%m_%d-%H_%M")
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        
    data_path = Path(config['project_settings']['data_path'])
    results_path = Path(config['project_settings']['results_path'])

    analysis_bin = config['analysis_settings']['bin_widths_to_analyze_ms'][0]

    # Create a new, timestamped directory for this run's results
    run_results_path = results_path / f"run_{e_n(analysis_bin)}_{now}"
    run_results_path.mkdir(parents=True, exist_ok=True)
    
    log_file = run_results_path / f'analysis_{now}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    tasks = list(generate_analysis_tasks(config))
    if not tasks:
        logging.warning("No existing simulation directories found to analyze.")
        return
    
    ## ================== NEW: Print Task Summary ==================
    logging.info(f"Found {len(tasks)} analysis tasks to run.")
    logging.info("--- Task Summary ---")
    for i, task in enumerate(tasks):
        bp = task['base_params']
        pulse_shape = bp.get('pulse_shape', 'N/A')
        
        # Create a clean string of the key variable parameters
        param_parts = []
        key_params = ['peak_amplitude', 'position', 'sigma', 'width', 'rise_time', 'decay_time', 'overall_amplitude']
        for key in key_params:
            if key in bp:
                param_parts.append(f"{key[:3]}={bp[key]}")
        
        param_str = ", ".join(param_parts)
        
        # Get detector and bin width info
        det_str = f", dets={task.get('dets_to_analyze', 'default')}" if 'dets_to_analyze' in task else ""
        bin_width_str = f"bin={task['bin_width_to_process_ms']}ms"
        
        logging.info(f"  {i+1: >4}: {pulse_shape:<15} ({param_str}, {bin_width_str}{det_str})")
    logging.info("--------------------")
    ## =============================================================

    logging.info(f"Found {len(tasks)} parameter sets to analyze. Starting parallel processing.")

    
    # <<< NEW: Define a chunk size >>>
    # A good starting point is 2-4 times the number of workers.
    CHUNK_SIZE = MAX_WORKERS * 4 

    final_results_csv_path = run_results_path / RESULTS_FILE_NAME
    # <<< NEW: Loop through the tasks in chunks >>>
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[i:i + CHUNK_SIZE]
        logging.info(f"--- Processing chunk {i//CHUNK_SIZE + 1}/{-(-len(tasks)//CHUNK_SIZE)} (Tasks {i+1}-{i+len(chunk)}) ---")
        
        # <<< The ProcessPoolExecutor is now INSIDE the loop >>>
        # This creates a fresh set of workers for each chunk.
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(analyze_one_group, task, data_path, run_results_path) for task in chunk}
            logging.info(f"Processing {len(futures)} tasks...")
            for future in tqdm(as_completed(futures), total=len(chunk), desc="Analyzing Chunk"):
                try:
                    result_list = future.result()
                    if result_list:
                        # Create a DataFrame for just this one result
                        result_df = pd.DataFrame(result_list)

                        # Check if the CSV file already exists to decide whether to write the header
                        header = not final_results_csv_path.exists()
                    
                        # Append this one result to the CSV file
                        result_df.to_csv(final_results_csv_path, index=False, mode='a', header=header)
                except Exception as e:
                    logging.error(f"An analysis task failed in the pool: {e}", exc_info=True)
        logging.info(f"✅ Analysis complete! Summary saved to:\n{final_results_csv_path}")
    


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
        default=SIM_CONFIG_FILE,
        help="Path to the simulation YAML configuration file. "
             f"Defaults to '{SIM_CONFIG_FILE}' if not provided."
    )
    
    # Parse the arguments provided by the user from the command line
    args = parser.parse_args()
    
    # Call the main function, passing in the path to the config file
    main(args.config_file)

# ========= RECENT EDITS TO RELATED FILES =========