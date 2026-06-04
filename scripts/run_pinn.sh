#!/bin/bash

# Usage: ./run_pinn.sh [PDE_NAME] [DOMAIN]
# Example: ./run_pinn.sh Black_Scholes phase

set -e

ORIGINAL_DIR=$(pwd)
# Function to restore directory on exit
restore_directory() {
    echo "Restoring to original directory: $ORIGINAL_DIR"
    cd "$ORIGINAL_DIR"
}
# Set up trap to restore directory on script exit
trap restore_directory EXIT

PDE=$1
DOMAIN=$2

if [[ -z "$PDE" || -z "$DOMAIN" ]]; then
  echo "Usage: $0 [PDE_NAME] [DOMAIN: weight|phase]"
  exit 1
fi

# Set base directory and config paths
BASE_DIR="./PINNs/$PDE"
PYTHON_CMD="python -u main.py -m both"

if [[ "$DOMAIN" == "weight" ]]; then
  CONFIG="-c configs_weight/Ours.ini"
elif [[ "$DOMAIN" == "phase" ]]; then
  CONFIG="-c configs_phase/Ours.ini -y configs_phase/tonn.yml"
else
  echo "Invalid domain: $DOMAIN. Use 'weight' or 'phase'."
  exit 1
fi

# Change directory and run command
cd "$BASE_DIR"
$PYTHON_CMD $CONFIG
