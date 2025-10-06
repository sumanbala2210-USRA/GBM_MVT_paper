"""
# generate_event.py

Suman Bala
16 Aug 2025: This script's sole purpose is to read a YAML configuration
            and generate a cache of raw Time-Tagged Event (TTE) files.
            All analysis is deferred to a separate script.
"""

# ========= Import necessary libraries =========
import matplotlib
matplotlib.use('Agg')
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
import platform
from email.message import EmailMessage
from SIM_lib import _parse_param, e_n, _create_param_directory_name, PULSE_MODEL_MAP, SIMULATION_REGISTRY, write_yaml, convert_det_to_list, send_email, complex_pulse_list

from TTE_SIM_v2 import generate_function_events, calculate_adaptive_simulation_params, create_final_plot, create_final_gbm_plot, generate_gbm_events_dets, sim_gbm_name_format, print_nested_dict#complex_pulse_example
# Assume your original simulation and helper functions are in a library
from sim_functions import constant


# ========= USER SETTINGS =========
MAX_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() - 2))
SIM_CONFIG_FILE = 'simulations_ALL.yaml' # Using the improved YAML
MAIL_FILE = 'config_mail.yaml'

# ========= GENERIC SIMULATION FRAMEWORK =========
# --- 1. The Simulation Registry (Unchanged) ---

class BaseSimulationTask:
    """A base class for simulation tasks."""
    def __init__(self, output_dir, variable_params, constant_params, analysis_settings):
        self.output_dir = output_dir
        self.params = {**variable_params, **constant_params}
        self.variable_params = variable_params
        self.analysis_settings = analysis_settings

        if 'trigger_set' in self.params:
            trigger_data = self.params.pop('trigger_set')
            if 'trigger' in trigger_data and 'trigger_number' not in trigger_data:
                trigger_data['trigger_number'] = trigger_data.pop('trigger')
            self.params.update(trigger_data)

    def run(self):
        raise NotImplementedError("The 'run' method must be implemented by a subclass.")

class AbstractPulseSimulationTask(BaseSimulationTask):
    def run(self):
        # --- 1. Common Setup for Both Simulation Types ---
        sim_type = 'gbm' if self.is_gbm else 'function'
        pulse_shape = self.params.get('pulse_shape')
        NN_target = self.params.get('total_sim', 1)
        original_seed = self.params.get('random_seed')

        param_dir_name = _create_param_directory_name(sim_type, pulse_shape, self.variable_params)
        #print(param_dir_name)
        #exit()

        #print_nested_dict(self.params)
        #print(type(self.params['pulse_shape']))
        #exit(1)

        dets = convert_det_to_list(self.params['det'])
        self.params['det'] = dets
        
        # Setup the pulse function and its parameters dynamically
        if pulse_shape not in PULSE_MODEL_MAP:
            logging.error(f"Pulse shape '{pulse_shape}' not defined in PULSE_MODEL_MAP. Skipping.")
            return

        func_to_use, required_params = PULSE_MODEL_MAP[pulse_shape]
        
        # Calculate adaptive simulation window and update params
        adaptive_params = calculate_adaptive_simulation_params(pulse_shape, self.params)
        self.params.update(adaptive_params)
        #print("Tstart:", self.params.get('t_start'), "Tstop:", self.params.get('t_stop'))

        # Special handling for triangular pulse start/stop times
        if pulse_shape == 'triangular':
            tpeak, width, peak_ratio = self.params['center_time'], self.params['width'], self.params['peak_time_ratio']
            self.params['t_start_tri'] = tpeak - (width * peak_ratio)
            self.params['t_stop_tri'] = self.params['t_start_tri'] + width

        try:
            func_par = tuple(self.params[key] for key in required_params)
        except KeyError as e:
            logging.error(f"Missing parameter {e} for pulse '{pulse_shape}' in {param_dir_name}. Skipping.")
            return

        # --- 2. Execute Logic Based on Simulation Type ---
        details_path = self.output_dir / sim_type / pulse_shape / param_dir_name
        # --- FUNCTION LOGIC: Generate/Append to a combined .npz file ---
        details_path.mkdir(parents=True, exist_ok=True)
        yaml_params = {**self.params, 'name_key': param_dir_name}
        write_yaml(yaml_params, details_path / f"{param_dir_name}.yaml")

        if self.is_gbm:
            for i in range(NN_target):
                iteration_seed = original_seed + i
                sim_params = {**self.params, 'random_seed': iteration_seed}

                # --- Corrected Caching Logic ---
                # First, check if ALL files for this seed already exist before doing anything.
                all_files_exist_for_this_seed = True
                for det in dets:
                    src_filename, bkgd_filename = sim_gbm_name_format(
                        trigger_number=self.params['trigger_number'],
                        det=det,
                        name_key=param_dir_name,
                        r_seed=iteration_seed
                    )
                    src_file_path = details_path / src_filename
                    bkgd_file_path = details_path / bkgd_filename
                    
                    if not (src_file_path.exists() and bkgd_file_path.exists()):
                        all_files_exist_for_this_seed = False
                        break  # Exit the check loop immediately

                # If all files were found, we can safely skip to the next seed.
                if all_files_exist_for_this_seed:
                    logging.info(f"All files for seed {iteration_seed} already exist. Skipping generation.")
                    continue
                # --- End of Corrected Logic ---

                try:
                    base_event_path = details_path / f"{param_dir_name}_r_seed_{iteration_seed}"
                    src_event_list, back_event_list = self.simulation_function(
                        event_file_path=base_event_path,
                        func=func_to_use, func_par=func_par,
                        back_func=constant, back_func_par=(self.params['background_level'],),
                        params=sim_params
                    )
                    if i == NN_target - 1:
                        create_final_gbm_plot(
                            src_event_list, back_event_list,
                            model_info={'func': func_to_use, 'func_par': func_par, 'base_params': self.params, 'snr_analysis': self.analysis_settings['snr_timescales']},
                            output_info={'file_path': details_path, 'file_info': param_dir_name}
                        )
                except Exception as e:
                    logging.warning(f"Failed to generate GBM files for seed {iteration_seed}. Error: {e}")

        else:
            
            #print(f"GEN_SCRIPT --- Creating directory: {details_path}") # <-- ADD THIS LINE
            combined_src_file_path = details_path / f"{param_dir_name}_src.npz"
            combined_bkgd_file_path = details_path / f"{param_dir_name}_bkgd.npz"

            existing_sources = []
            existing_backgrounds = []
            if combined_src_file_path.exists():
                try:
                    data_sources = np.load(combined_src_file_path, allow_pickle=True)
                    data_backgrounds = np.load(combined_bkgd_file_path, allow_pickle=True)
                    existing_sources = list(data_sources['realizations'])
                    existing_backgrounds = list(data_backgrounds['realizations'])
                except Exception as e:
                    logging.warning(f"Could not load {combined_src_file_path.name}, will overwrite. Error: {e}")

            num_existing = len(existing_sources)
            if num_existing >= NN_target:
                logging.info(f"Already have {num_existing}/{NN_target} realizations for {param_dir_name}. Skipping.")
                return

            start_index = num_existing
            new_sources = []
            new_backgrounds = []

            for i in range(start_index, NN_target):
                #print(f"Generating realization {i+1}/{NN_target} for {param_dir_name}...")
                iteration_seed = original_seed + i
                sim_params = {**self.params, 'random_seed': iteration_seed}
                
                try:
                    # This function returns the source events array in memory
                    source_events, back_events = self.simulation_function(
                        func=func_to_use, func_par=func_par,
                        back_func=constant, back_func_par=(self.params['background_level'],),
                        params=sim_params
                    )
                    new_sources.append(source_events)
                    new_backgrounds.append(back_events)
                    if i == start_index:
                        create_final_plot(
                            source_events=source_events,
                            background_events=back_events,
                            model_info={
                                'func': func_to_use,
                                'func_par': func_par,
                                'base_params': self.params,
                                'snr_analysis': self.analysis_settings['snr_timescales']
                            },
                            output_info={
                                'file_path': details_path,
                                'file_info': param_dir_name
                            }
                        )
                except Exception as e:
                    logging.warning(f"Function sim failed for seed {iteration_seed}. Error: {e}")

            all_sources = existing_sources + new_sources
            all_backgrounds = existing_backgrounds + new_backgrounds
            if all_sources:
                np.savez_compressed(
                    combined_src_file_path,
                    realizations=np.array(all_sources, dtype=object),
                    params=self.params
                )
            if all_backgrounds:
                np.savez_compressed(
                    combined_bkgd_file_path,
                    realizations=np.array(all_backgrounds, dtype=object),
                    params=self.params
                )

class GbmSimulationTask(AbstractPulseSimulationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_function = generate_gbm_events_dets
        self.is_gbm = True

class FunSimulationTask(AbstractPulseSimulationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_function = generate_function_events
        self.is_gbm = False


def generate_sim_tasks_from_config(config: Dict[str, Any], output_dir: 'Path') -> 'Generator':
    """
    Reads the new YAML config structure, merges parameters from all levels,
    and generates all parameter combinations for simulation tasks.
    """
    pulse_definitions = config.get('pulse_definitions', {})
    analysis_settings = config['analysis_settings']

    for campaign in config.get('simulation_campaigns', []):
        if not campaign.get('enabled', False):
            continue

        sim_type = campaign.get('type')
        TaskClass = globals().get(SIMULATION_REGISTRY.get(sim_type))
        if not TaskClass:
            logging.warning(f"Campaign '{campaign['name']}' has invalid type '{sim_type}'. Skipping.")
            continue

        logging.info(f"Generating tasks for campaign '{campaign['name']}'...")

        for pulse_config in campaign.get('pulses_to_run', []):
            pulse_shape = None
            pulse_override_constants = {}
            
            if isinstance(pulse_config, str):
                pulse_shape = pulse_config
            elif isinstance(pulse_config, dict):
                pulse_shape = list(pulse_config.keys())[0]
                if not pulse_config[pulse_config].get('enabled', True):
                    continue
                pulse_override_constants = pulse_config[pulse_shape].get('constants', {})

            if not pulse_shape:
                continue

            base_pulse_config = pulse_definitions.get(pulse_shape)
            if not base_pulse_config:
                logging.warning(f"Pulse shape '{pulse_shape}' not found in pulse_definitions. Skipping.")
                continue
            
            logging.info(f"  > Processing pulse: '{pulse_shape}'")

            # ▼▼▼ THE ONLY CHANGE IS HERE ▼▼▼
            variable_params_config = campaign.get('parameters', {}).copy()
            # ▲▲▲ THE ONLY CHANGE IS HERE ▲▲▲
            
            variable_params_config.update(base_pulse_config.get('parameters', {}))
            #print(variable_params_config)
    

            # =================== NEW CODE BLOCK STARTS HERE ===================
            # This forces the generation of a single, canonical template file.
            if pulse_shape in complex_pulse_list:
                if 'peak_amplitude' in variable_params_config:
                    variable_params_config['peak_amplitude'] = 1.0
                    logging.info(f"  > Overriding 'peak_amplitude' to 1.0 for '{pulse_shape}' generation.")
                
                if 'position' in variable_params_config:
                    variable_params_config['position'] = 0.0
                    logging.info(f"  > Overriding 'position' to 0.0 for '{pulse_shape}' generation.")
            # =================== NEW CODE BLOCK ENDS HERE ===================

            final_constants = {}
            final_constants.update(campaign.get('constants', {}))
            final_constants.update(base_pulse_config.get('constants', {}))
            final_constants.update(pulse_override_constants)
            final_constants['pulse_shape'] = pulse_shape

            param_names = list(variable_params_config.keys())
            param_values = [_parse_param(v) for v in variable_params_config.values()]
            #print(param_names)
            #print(param_values)
            
            if not param_values:
                param_values.append(())

            for combo in itertools.product(*param_values):
                current_variable_params = dict(zip(param_names, combo))
                #print(f"    - Params: {current_variable_params}")
                yield TaskClass(output_dir, current_variable_params, final_constants,  analysis_settings)

# ========= MAIN EXECUTION BLOCK =========
def main(config_filepath: str):
    # Setup directories and logging
    now = datetime.now().strftime("%y_%m_%d-%H_%M")
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
    data_path = Path(config['project_settings']['data_path'])

    hostname = platform.node()
    print("Hostname:", hostname)
    #exit(1)
    email_flag = True
    if hostname in ["sbalas-MBP.local", "sbalas-MBP.nsstc.nasa.gov"]:
        email_flag = False

    #shutil.rmtree(data_path, ignore_errors=True)
    data_path.mkdir(parents=True, exist_ok=True)
    log_file = data_path / f'gen_{now}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    # Generate tasks from the config file
    tasks = list(generate_sim_tasks_from_config(config, data_path))
    if not tasks:
        logging.warning("Task generation is not yet implemented for the new YAML format. Exiting.")
        return

    logging.info(f"Generated {len(tasks)} simulation tasks. Starting event file generation with {MAX_WORKERS} workers.")

    # Run all generation tasks in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # The worker is just task.run(), which now saves files as a side effect
        futures = {executor.submit(task.run) for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating Event Files"):
            try:
                future.result() # Check for errors
            except Exception as e:
                logging.error(f"A generation task failed: {e}", exc_info=True)

    logging.info("✅ All event files have been generated.")
    email_body = f"Fine generation is complete for {config_filepath}!"
    if email_flag:
        send_email(
            subject=f"{hostname}!!! All files have been generated successfully!!",
            body=email_body
        )

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
