#!/bin/bash

# Define the options for the synth_mask parameter
options_synth_mask=("0.25" "0.50" "0.75")
options_encoding=("std")
options_dataset=("AustraliaTourism" "MetroTraffic" "BeijingAirQuality" "RossmanSales" "PanamaEnergy")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for synth_mask in "${options_synth_mask[@]}"
  do
#    python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride 1
    python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride 8
#    python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride 16
#    python3.12 synthesis_wavestitch_pipeline_strided_preconditioning.py -d $dataset -synth_mask $synth_mask -stride 32
  done
done