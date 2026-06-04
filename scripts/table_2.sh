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

echo "STD_FO"
python -u main.py -m both -c configs_weight/STD_FO.ini
echo "STD_ZO"
python -u main.py -m both -c configs_weight/STD_ZO.ini
echo "TT_FO"
python -u main.py -m both -c configs_weight/TT_FO.ini
echo "TT_ZO (Ours)"
python -u main.py -m both -c configs_weight/Ours.ini