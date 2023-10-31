#!/bin/bash

# Define the input NetCDF file
input_nc="/home/mcarenso/code/tempest/cache/DYAMOND_9216x4608x74_7.5s_4km_4608_0000345600.LWDSA.2D.nc"

# Define the output NetCDF file
output_nc="/path/to/output/file.nc"

# Define the indices for the bounding box
lat_start=2304
lat_end=3840
lon_start=7936
lon_end=8704

# Use ncks to extract the desired region
ncks -d lat,$lat_start,$lat_end -d lon,$lon_start,$lon_end "$input_nc" temp_filtered.nc

# Use nccopy to copy the structure and variables to the output file
nccopy -k 4 -d 1 temp_filtered.nc "$output_nc"

# Clean up the temporary filtered file
rm temp_filtered.nc
