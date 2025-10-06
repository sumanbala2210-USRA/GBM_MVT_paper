from haar_power_mod import haar_power_mod
from numpy import loadtxt
import yaml

import numpy as np
import warnings
from SIM_lib import run_mvt_in_subprocess


SIM_CONFIG_FILE = 'simulations_ALL.yaml'




with open(SIM_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

haar_python_path = config['project_settings']['haar_python_path']

warnings.filterwarnings('ignore', r'divide by zero encountered')
warnings.filterwarnings('ignore', r'invalid value encountered')

min_dt = 1.0e-4

file='test_mvt_grb_lc.txt'
t,counts,drate = loadtxt(file,unpack=True,usecols=(0,2,3))
#print("Data loaded successfully.......")

print("Running variability analysis... using current PYTHON")
tsnr, tbeta, tmin, dtmin, slope, sigma_tsnr, sigma_tmin = haar_power_mod(
    counts, drate, min_dt=min_dt, max_dt=100., tau_bg_max=0.01, nrepl=2,
    doplot=True, bin_fac=4, zerocheck=False, afactor=-1., snr=3.,
    verbose=True, weight=True, file='test'
)
print("please check the PLOT 'test_haar_mod.png'.")

print(f"\n\nRunning MVT using....\nPATH = {haar_python_path}")
mvt_res = run_mvt_in_subprocess(
                            counts=counts,
                            bin_width_s=min_dt,
                            haar_python_path=haar_python_path,
                            doplot=1,
                            file_name='Best_second'
                        )
mvt_val = float(mvt_res[2]) #* 1000
mvt_err = float(mvt_res[3]) #* 1000
print(f"MVT value: {mvt_val} s,\nMVT error: {mvt_err} s")
print("please check the PLOT 'Best_second_haar_mod.png'.")

#print("\n--- Analysis Complete: Final Result ---")
