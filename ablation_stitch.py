import pandas as pd
from data_utils import Preprocessor
import numpy as np
import os

if __name__ == "__main__":
    ablation_data_parallelism = pd.DataFrame(
        columns=['Dataset', 'Parallelism', 'Level', 'Queries', 'Avg. MSE', 'Std. MSE', "Avg. Time", "Std. Time"])
    for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
        preprocessor = Preprocessor(dataset, False)
        for parallel in ["Pipe-1", "PipeNS-1", "Pipe-8", "PipeNS-8", "Pipe-16", "PipeNS-16", "Pipe-32", "PipeNS-32"]:
            for level in ["C", "M", "F"]:
                df_real = pd.read_csv(f"generated/{dataset}/{level}/real.csv").drop(columns=['Unnamed: 0'])
                df_real_cleaned = preprocessor.cleanDataset(dataset, df_real)
                non_hier_cols = [col for col in df_real_cleaned.columns if
                                 col not in preprocessor.hierarchical_features_cyclic]
                df_real_cleaned_selected = df_real_cleaned[non_hier_cols]
                mses = []
                # queries = None
                # time = None
                # std_time = None
                for trial in range(5):
                    if "-1" in parallel:
                        if "NS" in parallel:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_1_trial_{trial}_cycStd_grad_simplecoeff_nostitch.csv')
                        else:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_1_trial_{trial}_cycStd_grad_simplecoeff.csv')

                    elif "-8" in parallel:
                        if "NS" in parallel:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_simplecoeff_nostitch.csv')
                        else:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_simplecoeff.csv')

                    elif "-16" in parallel:
                        if "NS" in parallel:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_16_trial_{trial}_cycStd_grad_simplecoeff_nostitch.csv')
                        else:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_16_trial_{trial}_cycStd_grad_simplecoeff.csv')

                    if "-32" in parallel:
                        if "NS" in parallel:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_32_trial_{trial}_cycStd_grad_simplecoeff_nostitch.csv')
                        else:
                            df_synth = pd.read_csv(
                                f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_32_trial_{trial}_cycStd_grad_simplecoeff.csv')

                    df_synth = df_synth.drop(columns=['Unnamed: 0'])
                    df_synth_cleaned = preprocessor.cleanDataset(dataset, df_synth)
                    df_synth_cleaned_selected = df_synth_cleaned[non_hier_cols]
                    MSE = ((df_synth_cleaned_selected - df_real_cleaned_selected) ** 2).mean().mean()
                    mses.append(MSE)

                mses = np.array(mses)
                AVG_MSE = np.mean(mses)
                STD = np.std(mses)
                row = {'Dataset': dataset, 'Parallelism': parallel, 'Level': level, 'Avg. MSE': AVG_MSE,
                       'Std. MSE': STD}
                ablation_data_parallelism.loc[len(ablation_data_parallelism)] = row

    path = "experiments/ablations/stitching/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "ablation_stitch_wavestitch_grad_simplecoeff.csv")
    ablation_data_parallelism.to_csv(final_path)
