#!/bin/bash
#SBATCH --job-name=MVT_Analysis      # Job name
#SBATCH --nodes=1                    # Request a single node
#SBATCH --ntasks=1                   # Run a single instance of the script
#SBATCH --cpus-per-task=28           # Request 28 CPU cores for that single task
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH --qos=long_qos
#SBATCH -o MVT_%j.out                # Use job ID for unique log files
#SBATCH -e MVT_%j.err

# --- Script Commands Start Here ---

# Set the configuration file from the first command-line argument ($1).
# If no argument is given, it defaults to 'simulations_ALL.yaml'.
CONFIG_FILE=${1:-simulations_ALL.yaml}

echo "========================================================"
echo "Starting analysis job on node: $(hostname)"
echo "Using configuration file: ${CONFIG_FILE}"
echo "Job started on: $(date)"
echo "========================================================"

# Load the module environment
module load anaconda3

# Activate Conda environment
source /opt/apps/anaconda3/etc/profile.d/conda.sh
conda activate MVT

# Navigate to your project directory
# IMPORTANT: Update this path to your actual project directory
cd /home/rushikesh23/MVT_Project/GBM_MVT_simulation/MVTfermi

# --- CORRECTED LINE ---
# Run the python script, passing the config file directly as an argument
echo "Starting Python analysis script..."
python analyze_events.py "${CONFIG_FILE}"
echo "Script finished."

echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"