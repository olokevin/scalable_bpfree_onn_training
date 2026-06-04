#!/bin/bash

PDE=$1

ORIGINAL_DIR=$(pwd)
# Function to restore directory on exit
restore_directory() {
    echo "Restoring to original directory: $ORIGINAL_DIR"
    cd "$ORIGINAL_DIR"
}
# Set up trap to restore directory on script exit
trap restore_directory EXIT

cd "$BASE_DIR"
BASE_DIR="./PINNs/$PDE"

cd "$BASE_DIR"
echo "AD_FO"
python -u main.py -m both -c configs_weight/AD_FO.ini
echo "SE_FO"
python -u main.py -m both -c configs_weight/SE_FO.ini
echo "SG_FO (ours)"
python -u main.py -m both -c configs_weight/SG_FO.ini