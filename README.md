# GBM MVT Simulation – Paper Release

This repository contains the exact simulation snapshot used to generate the published MVT–SNR_MVT results.

This is NOT the development repository — this snapshot is frozen only to reproduce the figures and values in the paper.

---

## Two Python Environments Required

**IMPORTANT — Please install `conda` or `miniconda` first.**  
The HAAR environment (ENV-B) requires Python 3.10.8, which is easiest to create with conda.  
(ENV-A may be created using either conda *or* venv.)



1. Install `CONDA` or `MINICONDA`

### ENV‑A (main analysis environment)

You may install this environment either **with conda** or **without conda**.

#### Option 1 — using conda

```bash
conda create -n mvt_fermi python=3.10
conda activate mvt_fermi
pip install .
```

Optional UI + development extras:

```bash
pip install .[ui,dev]
```

* `[ui]`: Includes `streamlit`, `seaborn`, and `plotly` for building GUIs and advanced plots.
* `[dev]`: Includes `jupyter` + `notebook` support for interactive development.

#### Option 2 — without conda (standard venv)

```bash
python3 -m venv mvt_fermi
source mvt_fermi/bin/activate      # (Linux/Mac)
mvt_fermi\Scripts\activate.bat    # (Windows)
pip install .
```

Optional extras:

```bash
pip install .[ui,dev]
```

---

### ENV‑B (HAAR environment)
The HAAR MVT estimator must run under specific versions to reproduce the published values.

Example HAAR environment creation:

```bash
conda create -n haar_env python=3.10.8
conda activate haar_env
pip install -r requirements_haar.txt
```

Find the python executable for ENV‑B:

```bash
which python
```

Then set that path inside `simulations_ALL.yaml`:

project_settings:
    haar_python_path: "/path/to/conda/envs/haar_env/bin/python"

---

## Verify Setup

Run:

```bash
python test_mvt.py
```

This script:
1) runs HAAR MVT inside the current Python environment
2) runs HAAR MVT through ENV‑B via subprocess
3) prints both results for comparison

Final summary CSV files are included so values can be verified directly.

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

## Full Usage Documentation

For complete detailed usage (generate_event, analyze_events, classify_mvt_point, etc.) see:

docs/README_full.md

