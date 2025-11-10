# GBM MVT Simulation – Paper Release

This repository contains the exact simulation snapshot used to generate the published MVT–SNR_MVT results.

This is NOT the development repository — this snapshot is frozen only to reproduce the figures and values in the paper.

---

## Two Python Environments Required

This work uses **two separate Python environments**.

### ENV‑A (main analysis environment)
Install using the existing `pyproject.toml` in this repository.

Example:
pip install .

(you may activate your own main venv/conda environment first)

### ENV‑B (HAAR environment)
The HAAR MVT estimator must run under specific versions to reproduce the published values.

Example HAAR environment creation:

conda create -n haar_env python=3.10.8
conda activate haar_env
pip install -r requirements_haar.txt

Find the python executable for ENV‑B:

which python

Then set that path inside `simulations_ALL.yaml`:

project_settings:
    haar_python_path: "/path/to/conda/envs/haar_env/bin/python"

---

## Verify Setup

Run:

python test_mvt.py

This script:
1) runs HAAR MVT inside the current Python environment
2) runs HAAR MVT through ENV‑B via subprocess
3) prints both results for comparison

Final summary CSV files are included so values can be verified directly.

---

## Full Usage Documentation

For complete detailed usage (generate_event, analyze_events, classify_mvt_point, etc.) see:

docs/README_full.md

