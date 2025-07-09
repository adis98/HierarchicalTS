#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("AustraliaTourism" "MetroTraffic" "BeijingAirQuality" "RossmanSales" "PanamaEnergy")
# Loop through all synth_mask options and run the Python script with each one
for dataset in "${options_dataset[@]}"
do
  #python3.12 training_hyacinth_ordinal.py -d $dataset -epochs 300
  #python3.12 training_hyacinth_onehot.py -d $dataset -epochs 300
  python3.12 training_SSSD.py -d $dataset -epochs 300
done