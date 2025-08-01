#!/bin/bash

# Navigate to the base directory (change this if necessary)
BASE_DIR="/home/dsr/Documents/OMNI3D/omni3d_raw_data/"

# Iterate over each folder in the base directory
for folder in "$BASE_DIR"/*/Scan; do
    parent_name=$(basename "$(dirname "$folder")")
    
    # Rename files inside the Scan folder
    mv "$folder/Scan.obj" "$folder/${parent_name}.obj"
    #mv "$folder/Scan.mtl" "$folder/${parent_name}.mtl"
    #mv "$folder/Scan.jpg" "$folder/${parent_name}.jpg"

    # Rename the Scan folder itself
    mv "$folder" "$(dirname "$folder")/$parent_name"
done
