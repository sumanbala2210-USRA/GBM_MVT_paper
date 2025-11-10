GBM MVT Simulation – Paper Release

This repository contains the exact simulation snapshot used to generate the published MVT–SNR_MVT results.

This is NOT the development repository — this snapshot is frozen only to reproduce the figures and values in the paper.

Two environments are required:

ENV‑A (main) → install using pyproject.toml
ENV‑B (haar) → create using requirements_haar.txt

Example for HAAR environment:
conda create -n haar_env python=3.10.8
conda activate haar_env
pip install -r requirements_haar.txt

Then set haar_python_path inside simulations_ALL.yaml to point to the python in haar_env.

To verify:
python test_mvt.py

Final summary CSV files are included so values can be checked directly.

For full usage instructions (generate_event / analyze_events / classify_mvt_point etc.) see:
docs/README_full.md
