#!/bin/bash

# Define the options for the synth_mask parameter
options_synth_mask=("C" "M" "F")
options_encoding=("std")
options_dataset=("AustraliaTourism")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for synth_mask in "${options_synth_mask[@]}"
  do
    python3.12 synthesis_hyacinth_pipeline.py -d $dataset -synth_mask $synth_mask

    python3.12 synthesis_hyacinth_divide_and_conquer.py -d $dataset -synth_mask $synth_mask

    python3.12 synthesis_hyacinth_autoregressive.py -d $dataset -synth_mask $synth_mask -stride 16 -timesteps 200

    python3.12 synthesis_hyacinth_autoregressive.py -d $dataset -synth_mask $synth_mask -stride 32
  done
done