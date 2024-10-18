#!/bin/bash

# Define the options for the synth_mask parameter
options_synth_mask=("C" "M" "F")
#options_encoding=("std" "prop" "onehot" "ordinal")
options_encoding=("onehot")
options_dataset=("AustraliaTourism" "MetroTraffic")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for encoding in "${options_encoding[@]}"
  do
    for synth_mask in "${options_synth_mask[@]}"
    do
      if [[ "$encoding" == "std" ]]; then
        python3.12 synthesis_hyacinth_pipeline.py -d $dataset -synth_mask $synth_mask
      elif [[ "$encoding" == "prop" ]]; then
        python3.12 synthesis_hyacinth_pipeline.py -d $dataset -synth_mask $synth_mask -propCycEnc True
      elif [[ "$encoding" == "onehot" ]]; then
        python3.12 synthesis_hyacinth_onehot.py -d $dataset -synth_mask $synth_mask
      elif [[ "$encoding" == "ordinal" ]]; then
        python3.12 synthesis_hyacinth_ordinal.py -d $dataset -synth_mask $synth_mask
      fi
    done
  done  
done
