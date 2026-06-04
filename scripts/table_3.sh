#!/bin/bash

ORIGINAL_DIR=$(pwd)
# Function to restore directory on exit
restore_directory() {
    echo "Restoring to original directory: $ORIGINAL_DIR"
    cd "$ORIGINAL_DIR"
}
# Set up trap to restore directory on script exit
trap restore_directory EXIT

PDE=$1
cd "$BASE_DIR"
BASE_DIR="./PINNs/$PDE"

cd "$BASE_DIR"
echo "Ours"
python -u main.py -m both -c configs_phase/Ours.ini -y configs_phase/tonn.yml
echo "FLOPS"
python -u main.py -m both -c configs_phase/FLOPS.ini -y configs_phase/tonn.yml
echo "L2IGHT"
python -u main.py -m both -c configs_phase/L2IGHT.ini -y configs_phase/tonn.yml