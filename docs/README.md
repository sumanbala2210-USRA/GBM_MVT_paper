# MVTfermiTools 🚀

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-orange.svg)

A Python package for Minimum Variability Timescale (MVT) analysis, tailored for Fermi GBM data and generalized light curves.

This toolkit provides a powerful, configuration-driven pipeline to perform high-throughput Monte Carlo simulations and detailed temporal analysis of astrophysical signals.

---
## Key Features

* **Flexible Simulations**: Generate Time-Tagged Event (TTE) data for both real Fermi GBM observations and generic, mathematically-defined pulse shapes (`gaussian`, `fred`, `norris`, etc.).
* **Configuration Driven**: A central `simulation_ALL.yaml` file controls the entire workflow, from defining pulse shapes to specifying complex analysis routines.
* **Advanced Analysis**: Perform MVT and multi-timescale Signal-to-Noise (SNR) analysis on simulated data.
* **Complex Pulse Assembly**: A powerful "assemble-in-analysis" workflow allows for the creation of complex signals by combining a pre-generated "template" pulse with a variable "feature" pulse on the fly.
* **Parallel Processing**: The analysis is performed in parallel using all available CPU cores, significantly speeding up large-scale studies.

---
## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sumanbala2210-USRA/mvt_fermi.git](https://github.com/sumanbala2210-USRA/mvt_fermi.git)
    cd mvt_fermi
    ```

2.  **Install the core package:**
    This will install the main library and all essential dependencies.
    ```bash
    pip install .
    ```

3.  **Install optional dependencies:**
    For visualization and interactive analysis, install the optional UI and development packages.
    ```bash
    pip install .[ui,dev]
    ```
    * `[ui]`: Includes `streamlit`, `seaborn`, and `plotly` for building graphical user interfaces and advanced plots.
    * `[dev]`: Includes `jupyter` and `notebook` for interactive development.

---
## Configuration (`simulation_ALL.yaml`)

The entire workflow is controlled by the `simulation_ALL.yaml` file. It is divided into four main sections:

### `project_settings`
This section defines global paths and settings.

```yaml
project_settings:
  data_path: '001_DATA'
  results_path: '01_ANALYSIS_RESULTS'
  haar_python_path: "/path/to/your/python" # Path to the Python env with haar_power installed
  extra_pulse: "path/to/your/feature_pulse.npz" # Used for complex analysis
```

### `pulse_definitions`
This is a library where you define all the pulse shapes you want to work with. Each pulse has `parameters` (which can be a list of values to iterate over) and `constants`.

```yaml
pulse_definitions:
  gaussian:
    parameters:
      sigma: [0.05, 0.1, 0.5]
    constants:
      center_time: 0.0
  
  complex_pulse:
    parameters:
      position: [0.0, 3.0, 6.0]
    constants: {}
```

### `simulation_campaigns`
This is where you define the actual simulation and analysis runs. You can have multiple campaigns, each enabled or disabled with the `enabled` flag.

```yaml
simulation_campaigns:
- name: GBM_Single_GRB_Analysis
  type: gbm
  enabled: true
  parameters:
    peak_amplitude: [10, 50, 100]
    trigger_set:
      - {trigger_number: 250709653, angle: 20, det: 'all'}
  pulses_to_run:
    - complex_pulse
```
* `type`: Can be `gbm` or `function`.
* `parameters`: A dictionary where each key has a list of values. The scripts will create a run for every possible combination of these parameters.
* `pulses_to_run`: A list of pulse shapes (from `pulse_definitions`) to use in this campaign.

### `analysis_settings`
This section controls the behavior of the `analyze_events.py` script.

```yaml
analysis_settings:
  bin_widths_to_analyze_ms: [0.1, 1.0, 10.0]
  snr_timescales: [0.016, 0.032, 0.064]
  
  # For flexible GBM analysis
  detector_selections: [ ['n6'], ['n7'], ['all'] ]
  
  # For complex pulse assembly
  extra_pulse:
    pulse_shape: 'gaussian'
    constants:
      sigma: 0.05
      center_time: 0.0
```
* `detector_selections`: Activates the flexible analysis mode for GBM, running a separate analysis for each entry in the list.
* `extra_pulse`: Activates the "assemble-in-analysis" mode for `complex_pulse` runs.

---
## 🔬 Basic Workflow

The process involves two main steps: generating the data and then analyzing it.

### Step 1: Generate Event Files
First, configure your `simulation_ALL.yaml` to define the data you want to create. For example, to generate the template and feature files for a complex analysis, you would set up two separate campaigns and enable them.

Then, run the generation script from your terminal:
```bash
python generate_event.py
```
This will read your `YAML` file and create all the necessary TTE event files in the `001_DATA` directory.

### Step 2: Run the Analysis
Next, modify your `simulation_ALL.yaml` to define the analysis you want to perform. For example, you would disable the generation campaigns and enable a campaign that uses `complex_pulse` and defines the `analysis_settings`.

Then, run the analysis script:
```bash
python analyze_events.py
```
This script will read the `YAML`, find the pre-generated data, perform all the MVT/SNR calculations in parallel, and save the results.

---
## 📊 Output Structure

All results are saved in a timestamped folder within `01_ANALYSIS_RESULTS` to prevent overwriting.

```
01_ANALYSIS_RESULTS/
└── run_0.1_25_09_02-11_46/      <-- New folder for each analysis run
    ├── gbm/
    │   └── complex_pulse/
    │       ├── amp_50-pos_3.0/         <-- Folder for one assembled pulse analysis
    │       │   ├── Detailed_...csv     (Per-realization results)
    │       │   ├── MVT_dis_...png      (MVT distribution plot)
    │       │   └── Params_...yaml      (Full parameters for this run)
    │       │
    │       └── amp_100-pos_3.0/
    │           └── ...
    │
    ├── gbm_complex_pulse_summary.csv   <-- Clean, symmetric summary for this group
    ├── gbm_gaussian_summary.csv        <-- Clean, symmetric summary for another group
    └── final_summary_results.csv       <-- Master summary with all results
```
* **Intermediate Files**: For each analysis run, the script saves a detailed CSV with per-realization results, a parameter file, and a plot of the MVT distribution.
* **Final Files**: The script produces a master CSV file with all results, as well as separate, clean summary CSVs for each `sim_type` and `pulse_shape`.

---
## License

This project is licensed under the Apache License 2.0.