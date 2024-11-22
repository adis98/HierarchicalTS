#!/bin/bash

# Define the options for the synth_mask parameter
options_synth_mask=("C" "M" "F")
options_encoding=("onehot" "ordinal")
options_dataset=("AustraliaTourism" "MetroTraffic" "RossmanSales" "PanamaEnergy")
options_stride=(8 16 32)
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  for encoding in "${options_encoding[@]}"
  do
    for synth_mask in "${options_synth_mask[@]}"
    do
      for s in "${options_stride[@]}"
      do
        if [[ "$encoding" == "std" ]]; then
          python3.12 synthesis_hyacinth_pipeline.py -d $dataset -synth_mask $synth_mask -stride $s
        elif [[ "$encoding" == "prop" ]]; then
          python3.12 synthesis_hyacinth_pipeline.py -d $dataset -synth_mask $synth_mask -propCycEnc True -stride $s
        elif [[ "$encoding" == "onehot" ]]; then
          python3.12 synthesis_hyacinth_onehot.py -d $dataset -synth_mask $synth_mask -stride $s
        elif [[ "$encoding" == "ordinal" ]]; then
          python3.12 synthesis_hyacinth_ordinal.py -d $dataset -synth_mask $synth_mask -stride $s
        fi
      done
    done
  done  
done
