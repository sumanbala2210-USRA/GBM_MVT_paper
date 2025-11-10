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
    git clone [https://github.com/sumanbala2210-USRA/GBM_MVT_paper.git](https://github.com/sumanbala2210-USRA/GBM_MVT_paper.git)
    cd GBM_MVT_paper
    ```
## Two Python Environments Required

**IMPORTANT — Please install `conda` or `miniconda` first.**  
The HAAR environment (ENV-B) requires Python 3.10.8, which is easiest to create with conda.  
(ENV-A may be created using either conda *or* venv.)

2. Install `CONDA` or `MINICONDA`

### ENV‑A (main analysis environment)

3.  **Install the core package:**
    This will install the main library and all essential dependencies.
    ```bash
    conda create -n mvt_fermi
    conda activate mvt_fermi
    pip install .
    ```

4.  **Install optional dependencies:**
    For visualization and interactive analysis, install the optional UI and development packages.
    ```bash
    pip install .[ui,dev]
    ```
    * `[ui]`: Includes `streamlit`, `seaborn`, and `plotly` for building graphical user interfaces and advanced plots.
    * `[dev]`: Includes `jupyter` and `notebook` for interactive development.

### ENV‑B (HAAR environment)
The HAAR MVT estimator must run under specific versions to reproduce the published values.

5. **Example HAAR environment creation:**

```bash
conda create -n haar_env python=3.10.8
conda activate haar_env
pip install -r requirements_haar.txt
```

6. **Find the python executable for ENV‑B:**

```bash
which python
```
7. **Then set that path inside `simulations_ALL.yaml`:**

```yaml
project_settings:

    haar_python_path: "/path/to/conda/envs/haar_env/bin/python"
```


## Verify MVT Results

Run:

```bash
conda activate mvt_fermi
python test_mvt.py
```

This script:
1) runs HAAR MVT inside the current Python environment and create `test_haar_mod.png`
2) runs HAAR MVT through ENV‑B via subprocess, create `Best_second_haar_mod.png`
3) prints both results for comparison and comape both the plots with `Best_MVT_ref.png`. 

Final summary CSV files are included so values can be verified directly.

---

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
- name: FUNCTION_Pulse_Simulation
  type: function
  enabled: true
  parameters:
    peak_amplitude: [10, 50, 100]

  constants:
    trigger_number: 99999999
    det: x
    angle: 0
    t_start: 4 
    t_stop: 12
    background_level: 1
    scale_factor: 1000 # Background Count/Sec
    random_seed: 37 # Starting random seed value
    grid_resolution: 0.00001
    total_sim: 30 # Total number of simulations

  pulses_to_run:
    # - gaussian
    # - triangular  # Example of disabling a pulse for this campaign
    - norris
    # - fred
    # - lognormal
    # - complex_pulse  # Example of enabling a pulse for this campaign
    # - complex_pulse_long  # Example of enabling a pulse for this campaign
    # - complex_pulse_short
    # - complex_pulse_short_2p10ms
    # - complex_pulse_short_2p3ms

    
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
## 🔬 Basic Workflow (Make sure your environment is `mvt_fermi`)

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
    ├── function/
    │   └── complex_pulse/
    │       ├── amp_50-pos_3.0/         <-- Folder for one assembled pulse analysis
    │       │   ├── Detailed_...csv     (Per-realization results)
    │       │   ├── MVT_dis_...png      (MVT distribution plot)
    │       │   └── Params_...yaml      (Full parameters for this run)
    │       │
    │       └── amp_100-pos_3.0/
    │           └── ...
    │
    ├── analysis_25_11_02-11_50.log     <-- log file
    └── final_summary_results.csv       <-- Master summary with all results
```
* **Intermediate Files**: For each analysis run, the script saves a detailed CSV with per-realization results, a parameter file, and a plot of the MVT distribution.
* **Final Files**: The script produces a master CSV file with all results, as well as separate, clean summary CSVs for each `sim_type` and `pulse_shape`.

---
## Classification of external MVT values

The helper script `classify_mvt_point.py` can be used to compare a new GRB MVT measurement against the validation boundary curve.

Example:

```bash
python classify_mvt_point.py --mvt_ms 5.0 --snr_mvt 120.0
python classify_mvt_point.py --mvt_ms 5.0 --snr_mvt 120.0 --mode plot
```

This returns which side of the validation boundary the point belongs to. 

---
## License

This project is licensed under the Apache License 2.0.