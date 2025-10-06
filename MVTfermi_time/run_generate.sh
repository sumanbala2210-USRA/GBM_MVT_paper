#!/bin/bash
#SBATCH --job-name=GenerateEvents   # Descriptive name for your job
#SBATCH --nodes=1                   # Request a single machine
#SBATCH --ntasks=1                  # Run a single instance of your script
#SBATCH --cpus-per-task=16          # Request 16 CPU cores for parallel generation
#SBATCH --mem=16G                   # Request 16 GB of memory
#SBATCH --time=02:00:00             # Set a time limit (e.g., 2 hours)
#SBATCH --partition=short           # Specify the partition/queue to run on
#SBATCH -o gen_events_%j.out        # File to capture standard output
#SBATCH -e gen_events_%j.err        # File to capture standard error

# --- Script Commands Start Here ---

# Set the configuration file from the first command-line argument.
# It defaults to 'simulations_ALL.yaml' if you don't provide one.
CONFIG_FILE=${1:-simulations_ALL.yaml}

echo "========================================================"
echo "Starting event generation job on node: $(hostname)"
echo "Using configuration file: ${CONFIG_FILE}"
echo "Job started on: $(date)"
echo "========================================================"

# Load the necessary software modules for your environment
module load anaconda3

# Activate your specific conda environment
source /opt/apps/anaconda3/etc/profile.d/conda.sh
conda activate MVT

# Navigate to the directory where your script is located
# IMPORTANT: Update this path to your actual project directory
cd /path/to/your/MVTfermi

# Execute your Python script, passing it the configuration file
python generate_event.py "${CONFIG_FILE}"

echo "========================================================"
echo "Job finished on: $(date)"
echo "========================================================"