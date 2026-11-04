#!/bin/bash

# This script runs a parameter sweep by modifying config.py using sed.

echo "Starting parameter sweep..."
echo "============================="

# Loop from 1 to 50
for FWHM in $(seq 105 5 120); do

    echo "Photon FWHM : ${FWHM} GHz"

    # --- Use sed to modify the config.py file in-place ---
    # -i means "edit in-place"
    # The pattern "s/.../.../g" means substitute:
    # 'INTERNODE_LENGTH = .*' matches the target line, whatever its current value.
    # 'INTERNODE_LENGTH = ${length} # km' is the replacement string.
    sed -i '' "s/^PHOTON_FWHM_GHZ = .*/PHOTON_FWHM_GHZ = ${FWHM} # 30.0/g" config.py
    sed -i '' "s/run_value = .*/run_value = ${FWHM}/g" qsi_source_fidelity.py
    # Check if sed command was successful
    if [ $? -ne 0 ]; then
        echo "Error: sed command failed. Aborting."
        exit 1
    fi
    # Define the output filename for this run
    #output_name="run_length_${length}km"

    # --- Run the Python simulation script ---
    # The script will now read the new value from config.py
    python qsi_source_fidelity.py

    echo "-----------------------------"
done

echo "Parameter sweep complete."
echo "============================="
