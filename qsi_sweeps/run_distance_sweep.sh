#!/bin/bash

# This script runs a parameter sweep by modifying config.py using sed.

echo "Starting parameter sweep..."
echo "============================="

# Loop from 1 to 50
for length in $(seq 0 50); do

    echo "Processing Length: ${length} km"

    # --- Use sed to modify the config.py file in-place ---
    # -i means "edit in-place"
    # The pattern "s/.../.../g" means substitute:
    # 'INTERNODE_LENGTH = .*' matches the target line, whatever its current value.
    # 'INTERNODE_LENGTH = ${length} # km' is the replacement string.
    sed -i '' "s/^INTERNODE_LENGTH = .*/INTERNODE_LENGTH = ${length} # km/g" config.py

    # Check if sed command was successful
    if [ $? -ne 0 ]; then
        echo "Error: sed command failed. Aborting."
        exit 1
    fi

    # Define the output filename for this run
    output_name="run_length_${length}km"

    # --- Run the Python simulation script ---
    # The script will now read the new value from config.py
    python qsi_source_fidelity_distance.py ${length}

    echo "-----------------------------"
done

echo "Parameter sweep complete."
echo "============================="
