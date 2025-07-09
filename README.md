# WaveStitch: Flexible and Fast Conditional Time Series Generation With Diffusion Models

**WaveStitch** is a deep generative framework for **conditional time series synthesis**. It enables the generation of realistic time series data conditioned on auxiliary features (e.g., labels, metadata) and signal anchors (e.g., partial observations). This codebase provides tools for experimentation and ablation studies.

## Description
- **TSImputers/**  
  Shared neural network backbones used in diffusion and TimeGAN models (integrates components from SSSD and TimeGAN codebases)

- **experiments/**  
  Results for experiments featured in the paper (e.g., parallelism vs autoregressive, encoding ablations, SOTA comparisons)

- **utils/**  
  Helper functions for the diffusion model backbone

- **synthesis_*.py**  
  Scripts for generating synthetic time series data (includes preconditioning models, TimeGAN, TimeWeaver, TSDiff, WaveStitch with repaint-based conditioning)

- **training_*.py**  
  Scripts for training different model types (e.g., TimeGAN, TimeWeaver, WaveStitch, etc.)

- **ablation_*.py / ablation_*.sh**  
  Pairs of scripts for ablation studies. `.sh` files generate synthetic data, and corresponding `.py` files analyze results and store them in CSVs. (To avoid regenerating data, run analysis scripts first, then modify shell scripts to only generate missing data)

- **\*_analysis.py / \*_plotter.py**  
  Scripts for analyzing synthetic data (autocorrelation, cross-correlation, etc.) and visualizing experiment results (separate from ablation studies)

- **\*.sh**  
  Shell scripts for orchestrating runs, including training all backbones

- **WaveStitch_Appendix.pdf**  
  Contains algorithm details and illustrations for repaint-based WaveStitch implementation

- **generated/**  
  Directory for storing generated data during experiments  
  - **[dataset name]/**  
    - **c/** (Coarse-grained tasks - Root-level conditions)  
    - **m/** (Medium-grained tasks - Intermediate-level conditions)  
    - **f/** (Fine-grained tasks - Bottom-level conditions)

- **saved_models/**  
  Directory where trained models are saved after running training scripts

- **data_utils.py**  
  Preprocessing functions for each dataset

- **metasynth.py**  
  Generates tasks from datasets for different experiment configurations
