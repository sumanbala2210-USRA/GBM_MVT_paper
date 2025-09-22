"""
Suman Bala
16 Aug 2025: This script's sole purpose is to read a YAML configuration
            and generate a cache of raw Time-Tagged Event (TTE) files.
            All analysis is deferred to a separate script.
"""

# ========= Import necessary libraries =========
import os

import numpy as np

import yaml
import logging
import smtplib
import mimetypes
from email.message import EmailMessage
import subprocess
import json
import tempfile
import os
from typing import Dict, Any, Tuple, Callable, List


from sim_functions import gaussian2, triangular, constant, norris, fred, lognormal, complex_pulse_wrapper
# ========= Import necessary libraries =========


GMAIL_FILE = 'config_mail.yaml'
#HAAR_ENV_PATH = "/Users/sbala/anaconda3/bin/python"
WRAPPER_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'run_haar_power_mod.py')




def run_mvt_in_subprocess(
    counts: np.ndarray,
    bin_width_s: float,
    haar_python_path: str,
    file_name: str = "test",
    doplot: int = 0,
    time_resolved: bool = False,
    window_size_s: float = 1.0,
    step_size_s: float = 1.0,
    tstart: float = 0.0
) -> List:
    """
    Runs the MVT analysis in a separate Python environment for either a standard
    or a time-resolved calculation.

    Args:
        counts (np.ndarray): The binned light curve data.
        bin_width_s (float): The bin width in seconds.
        haar_python_path (str): The full path to the Python executable.
        file_name (str): Base name for output files. Defaults to "test".
        doplot (int): Whether to generate plots. Defaults to 0.
        time_resolved (bool): If True, performs time-resolved analysis. Defaults to False.
        window_size_s (float): Window size in seconds for time-resolved mode.
        step_size_s (float): Step size in seconds for time-resolved mode.

    Returns:
        List: The results from the analysis.
    """
    with tempfile.NamedTemporaryFile(suffix='.npy') as tmp_input, \
         tempfile.NamedTemporaryFile(suffix='.json', mode='w+') as tmp_output:

        try:
            # 1. Save the input data
            np.save(tmp_input.name, counts)

            # 2. Construct the base command
            command = [
                haar_python_path,
                WRAPPER_SCRIPT_PATH,
                "--input", tmp_input.name,
                "--output", tmp_output.name,
                "--min_dt", str(bin_width_s),
                "--doplot", str(doplot),
                "--file", file_name
            ]

            # 3. Conditionally add arguments for time-resolved analysis
            if time_resolved:
                command.extend([
                    "--time-resolved",
                    "--window-size", str(window_size_s), # Must be a string for the command line
                    "--step-size", str(step_size_s),      # Must be a string for the command line
                    "--tstart", str(tstart)               # <<< NEW ARGUMENT
                ])

            # 4. Run the command
            subprocess.run(command, check=True, capture_output=True, text=True)

            # 5. Load and return the results
            tmp_output.seek(0)
            mvt_res = json.load(tmp_output)
            return mvt_res

        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess for MVT calculation failed. Stderr: {e.stderr}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred in run_mvt_in_subprocess: {e}")
            return []


def send_email(subject='Python Script Completed', body='!!', attachment_path=None):
    """
    Sends an email to a list of recipients with an optional attachment.

    Args:
        recipients (list): A list of email addresses to send the email to.
        subject (str): The subject line of the email.
        body (str): The plain text content of the email.
        attachment_path (str, optional): The file path of the file to attach. Defaults to None.
    """
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = '2210sumaanbala@gmail.com'
    # Join the list of recipients into a single comma-separated string
    recipients = ['sumanbala2210@gmail.com', 'pv0004@uah.edu']
    msg['To'] = ', '.join(recipients)
    msg.set_content(body)

    # --- Attach the file if a path is provided ---
    if attachment_path and os.path.exists(attachment_path):
        # Guess the MIME type of the file
        ctype, encoding = mimetypes.guess_type(attachment_path)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'  # Default for unknown file types
        
        maintype, subtype = ctype.split('/', 1)

        with open(attachment_path, 'rb') as fp:
            msg.add_attachment(fp.read(),
                               maintype=maintype,
                               subtype=subtype,
                               filename=os.path.basename(attachment_path))

    # --- Login and send the email ---
    with open(GMAIL_FILE, 'r') as f:
        config_mail = yaml.safe_load(f)

    gmail_user = config_mail['gmail_user']
    gmail_password = config_mail['gmail_password'] # Use your Gmail App Password

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(gmail_user, gmail_password)
            smtp.send_message(msg)
        print(f"Email successfully sent to: {', '.join(recipients)}")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")


def send_email_old(input='!!'):
    msg = EmailMessage()
    msg['Subject'] = 'Python Script Completed'
    msg['From'] = '2210sumaanbala@gmail.com'
    msg['To'] = 'sumanbala2210@gmail.com'
    msg.set_content(f'Hey, your script has finished running!\n{input}')

    with open(GMAIL_FILE, 'r') as f:
        config_mail = yaml.safe_load(f)

    # Use your Gmail App Password here
    gmail_user = config_mail['gmail_user']
    gmail_password = config_mail['gmail_password']

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(gmail_user, gmail_password)
        smtp.send_message(msg)


def convert_det_to_list(det_string):
    """
    Converts a comma-separated string to a list,
    prepending 'n' to each element.
    """
    det_val = 10
    all_dets = [f'n{i}' for i in range(det_val)] + ['na', 'nb']
    # 1. Split the input string into a list of smaller strings
    if isinstance(det_string, str) and det_string.lower() == 'all' :
        return all_dets
    if type(det_string) is list:
        if det_string == ['all']:
            return all_dets
        return det_string
    elif isinstance(det_string, str):
        elements = det_string.split(',')
    #    - f"n{...}" formats it into the desired 'nx' string.
        new_list = []
        for element in elements:
            element = element.strip()
            if element[0] == 'n':
                new_list.append(element)
            else:
                new_list.append(f"n{element}")

        return new_list

# ========= USER SETTINGS =========

def write_yaml(par_dictionary, yaml_file, comments=[]):
    """ 
    Write YAML file in a more compact format than yaml.safe_dump().

    Args:
        par_dictionary (dict):      Dictionary of values to write.
        yaml_file (str):            Path of the output file to write.
        comments (list):            Optional list of comments to include in the YAML file.
    """
    # Convert the dictionary to a YAML string
    yaml_content = format_par_as_yaml(par_dictionary, '')
    # Open the file for writing
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)



def format_par_as_yaml(dict_data, dict_name, count_dict=0, indent=0):
    """
    Formats a dictionary into a YAML-like string.
    """
    yaml_str = ''
    
    # Add the dictionary name with correct indentation
    if dict_name:
        yaml_str += ' ' * indent + dict_name + ':\n'
        indent += 4  # Increase indentation for child elements

    # Convert dictionary items to a list for iteration
    items = list(dict_data.items())
    count_dict += 1  # Increase depth level

    for i, (key, value) in enumerate(items):
        if isinstance(value, dict):
            # Recursive call, passing increased indent
            yaml_str += format_par_as_yaml(value, key, count_dict=count_dict, indent=indent)
        else:
            # Format the value based on its type
            if isinstance(value, str):
                value_str = f"'{value}'"
            elif isinstance(value, list):
                value_str = yaml.dump(value, default_flow_style=True).strip()
            else:
                value_str = str(value)
            
            yaml_str += f"{' ' * indent}{key}: {value_str}\n"

    return yaml_str


# ========= GENERIC SIMULATION FRAMEWORK =========
# --- 1. The Simulation Registry (Unchanged) ---
SIMULATION_REGISTRY = {
    'gbm': 'GbmSimulationTask',
    'function': 'FunSimulationTask',
}

# This map defines which keys from the `variable_params` dictionary are
# relevant for creating the directory name for each pulse shape.
# This map now has separate rules for 'gbm' and 'function' types.
RELEVANT_NAMING_KEYS = {
    'gbm': {
        # For GBM runs, 'angle' from the trigger set IS a meaningful parameter.
        'gaussian':   ['peak_amplitude', 'sigma', 'angle', 'background_level', 'det'],
        'triangular': ['peak_amplitude', 'width', 'peak_time_ratio', 'angle', 'background_level', 'det'],
        'norris':     ['peak_amplitude', 'rise_time', 'decay_time', 'angle', 'background_level', 'det'],
        'fred':       ['peak_amplitude', 'rise_time', 'decay_time', 'angle', 'background_level', 'det'],
        'lognormal':  ['peak_amplitude', 'sigma', 'center_time', 'angle', 'background_level', 'det'],
        'complex_pulse': ['peak_amplitude', 'position', 'angle', 'background_level', 'overall_amplitude', 'det'],
    },
    'function': {
        # For function runs, 'angle' etc. are dummy values and should be IGNORED.
        'gaussian':   ['peak_amplitude', 'sigma', 'background_level'],
        'triangular': ['peak_amplitude', 'width', 'peak_time_ratio', 'background_level'],
        'norris':     ['peak_amplitude', 'rise_time', 'decay_time', 'background_level'],
        'fred':       ['peak_amplitude', 'rise_time', 'decay_time', 'background_level'],
        'lognormal':  ['peak_amplitude', 'sigma', 'center_time', 'background_level'],
        'complex_pulse': ['peak_amplitude', 'position', 'background_level', 'overall_amplitude'],
    }
}

# names of the parameters it requires. This eliminates the big if/elif block.
PULSE_MODEL_MAP_old = {
    'gaussian':   (gaussian2, ['peak_amplitude', 'center_time', 'sigma']),
    'triangular': (triangular, ['peak_amplitude', 't_start_tri', 'center_time', 't_stop_tri']),
    'norris':     (norris, ['peak_amplitude', 'start_time', 'rise_time', 'decay_time']),
    'fred':       (fred, ['peak_amplitude', 'start_time', 'rise_time', 'decay_time']),
    'lognormal':  (lognormal, ['peak_amplitude', 'center_time', 'sigma']),
}



# In TTE_SIM_v2.py

PULSE_MODEL_MAP = {
    'gaussian':   (gaussian2, ['peak_amplitude', 'center_time', 'sigma']),
    'triangular': (triangular, ['peak_amplitude', 't_start_tri', 'center_time', 't_stop_tri']),
    'norris':     (norris, ['peak_amplitude', 'start_time', 'rise_time', 'decay_time']),
    'fred':       (fred, ['peak_amplitude', 'start_time', 'rise_time', 'decay_time']),
    'lognormal':  (lognormal, ['peak_amplitude', 'center_time', 'sigma']),
    #'complex_pulse': (complex_pulse_wrapper, ['peak_amplitude', 'position']),

    # === Add this new line for your complex pulse ===
    'complex_pulse': (complex_pulse_wrapper, ['peak_amplitude', 'position', 'overall_amplitude']),
}
# ========= HELPER FUNCTIONS =========
def e_n(number):
    if number == 0:
        return "0"
    abs_number = abs(number)
    # If number is "simple" (0.01 ≤ |number| ≤ 1000), use fixed-point
    if 1e-2 <= abs_number <= 1e3:
        if float(number).is_integer():
            return str(int(number))
        else:
            return str(number)
    # Otherwise, use scientific notation with em/e style
    scientific_notation = "{:.1e}".format(abs_number)
    base, exponent = scientific_notation.split('e')
    exponent = int(exponent)
    abs_exponent = abs(exponent)
    
    # Remove unnecessary '.0' if base is an integer
    if float(base).is_integer():
        base = str(int(float(base)))
    
    # Format with e/em style
    if exponent < 0:
        return f"{base}em{abs_exponent}"
    else:
        return f"{base}e{abs_exponent}"

# The refactored naming function
def _create_param_directory_name1(sim_type: str, pulse_shape: str, variable_params: dict, extra_pulse: bool = False) -> str:
    """
    Creates a descriptive directory name using only the parameters relevant
    to the given sim_type and pulse_shape.
    """
    key_abbreviations = {
        'peak_amplitude': 'amp', 'background_level': 'bkg', 'sigma': 'sig',
        'width': 'w', 'center_time': 't0', 'rise_time': 'tr', 'decay_time': 'td',
        'peak_time_ratio': 'pr', 'angle': 'ang', 'position': 'pos',
    }
    
    # Get the list of relevant keys for this specific sim_type and pulse_shape
    relevant_keys = RELEVANT_NAMING_KEYS.get(sim_type, {}).get(pulse_shape, [])
    
    name_parts = []
    
    # --- Special Handling for GBM Trigger Sets ---
    if sim_type == 'gbm' and 'trigger_set' in variable_params:
        trigger_info = variable_params['trigger_set']

                # Check if this is a single-GRB run with all detectors
        if trigger_info.get('det') == 'all':
            # Name the directory by the detector setting
            abbr = key_abbreviations.get('det', 'det')
            val_str = 'all'
            name_parts.append(f"{abbr}_{val_str}")
        else:
            # Otherwise, default to naming by the angle for your other studies
            if 'angle' in trigger_info:
                abbr = key_abbreviations.get('angle', 'ang')
                val_str = e_n(trigger_info['angle'])
                name_parts.append(f"{abbr}_{val_str}")

    # --- General Parameter Handling ---
    for key, value in sorted(variable_params.items()):
        # Skip parameters that aren't relevant for this sim type
        if key not in relevant_keys:
            continue
            
        # We already handled this dictionary above
        if key == 'trigger_set':
            continue
            
        abbr = key_abbreviations.get(key, key[:3])
        val_str = e_n(value)
        name_parts.append(f"{abbr}_{val_str}")
        #print(f"Key: {key}, Value: {value}, Abbr: {abbr}, Val_str: {val_str}")
        
    return "-".join(name_parts)


def _create_param_directory_name(sim_type: str, pulse_shape: str, variable_params: dict, extra_pulse: bool = False) -> str:
    """
    Creates a descriptive directory name using only the parameters relevant
    to the given sim_type and pulse_shape.
    """
    key_abbreviations = {
        'peak_amplitude': 'amp', 'background_level': 'bkg', 'sigma': 'sig',
        'width': 'w', 'center_time': 't0', 'rise_time': 'tr', 'decay_time': 'td',
        'peak_time_ratio': 'pr', 'angle': 'ang', 'position': 'pos', 'det': 'det', 
        'overall_amplitude': 'oamp'
    }
    
    relevant_keys = RELEVANT_NAMING_KEYS.get(sim_type, {}).get(pulse_shape, [])
    if extra_pulse:
        relevant_keys.append('position')
    name_parts = []
    
    # --- Special Handling for GBM Trigger Sets ---
    if sim_type == 'gbm' and 'trigger_set' in variable_params:
        trigger_info = variable_params['trigger_set']
        # For single GRB studies, prioritize 'det' if it is 'all'. Otherwise, use 'angle'.
        if trigger_info.get('det') == 'all':
            name_parts.append(f"det_all")
        elif 'angle' in trigger_info:
            name_parts.append(f"ang_{e_n(trigger_info['angle'])}")

    # --- General Parameter Handling ---
    for key, value in sorted(variable_params.items()):
        if key not in relevant_keys or key == 'trigger_set':
            continue
            
        abbr = key_abbreviations.get(key, key)
        val_str = e_n(value)
        name_parts.append(f"{abbr}_{val_str}")
        
    return "-".join(name_parts)



def _parse_param(param_config: Any) -> list:
    """
    Helper function to parse parameter definitions from the YAML config.
    (This function is the same as your original).
    """
    if isinstance(param_config, dict) and 'start' in param_config:
        # Handles np.arange style definitions: {start: 0, stop: 1, step: 0.1}
        return np.arange(**param_config)
    if isinstance(param_config, list):
        # Handles simple lists: [1, 2, 3]
        return param_config
    if isinstance(param_config, (int, float, str)):
        # Handles single values, wrapping them in a list for itertools
        return [param_config]
    # You might add handling for other types if needed
    raise TypeError(f"Unsupported parameter format in YAML: {param_config}")


