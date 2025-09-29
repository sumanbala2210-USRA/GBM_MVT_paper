import argparse
import os
from gdt.missions.fermi.gbm.finders import TriggerFinder
from pathlib import Path
from trigger_process import full_analysis_workflow

trigger_number = '211211549' # "250919020" # "191227723" # 
path = os.path.join(os.getcwd(), f"bn{trigger_number}")

trigger_ftp = TriggerFinder(trigger_number, protocol='AWS')
trigger_ftp.get_tte(download_dir=path)
trigger_ftp.get_rsp2(download_dir=path)
trigger_ftp.get_cat_files(download_dir=path)
trigger_ftp.get_trigdat(download_dir=path)

bw_snr = 0.010
