import pandas as pd
from data_utils import Preprocessor
import numpy as np
import os

if __name__ == "__main__":

    dataset = "AustraliaTourism"
    preprocessor = Preprocessor(dataset, False)
    ablation_data_parallelism = pd.DataFrame(
        columns=['Dataset', 'Parallelism', 'Level', 'Queries', 'Avg. MSE', 'Std. MSE'])
    for parallel in ["AR-8", "AR-16", "AR-32", "DNQ", "Pipeline"]:
        for level in ["C", "M", "F"]:
            df_real = pd.read_csv(f"generated/{dataset}/{level}/real.csv").drop(columns=['Unnamed: 0'])
            df_real_cleaned = preprocessor.cleanDataset(dataset, df_real)
            non_hier_cols = [col for col in df_real_cleaned.columns if
                             col not in preprocessor.hierarchical_features_cyclic]
            df_real_cleaned_selected = df_real_cleaned[non_hier_cols]
            mses = []
            queries = None
            for trial in range(5):
                if parallel == "AR-16":
                    df_synth = pd.read_csv(
                        f'generated/{dataset}/{level}/synth_hyacinth_autoregressive_stride_16_trial_{trial}_cycStd.csv')
                    if trial == 0:
                        with open(
                                f'generated/{dataset}/{level}/denoiser_calls_autoregressive_stride_16_cycStd.txt') as file:
                            queries = int(file.readline())
                elif parallel == "AR-8":
                    df_synth = pd.read_csv(
                        f'generated/{dataset}/{level}/synth_hyacinth_autoregressive_stride_8_trial_{trial}_cycStd.csv')
                    if trial == 0:
                        with open(
                                f'generated/{dataset}/{level}/denoiser_calls_autoregressive_stride_8_cycStd.txt') as file:
                            queries = int(file.readline())
                elif parallel == "AR-32":
                    df_synth = pd.read_csv(
                        f'generated/{dataset}/{level}/synth_hyacinth_autoregressive_stride_32_trial_{trial}_cycStd.csv')
                    if trial == 0:
                        with open(
                                f'generated/{dataset}/{level}/denoiser_calls_autoregressive_stride_32_cycStd.txt') as file:
                            queries = int(file.readline())
                elif parallel == "DNQ":
                    df_synth = pd.read_csv(
                        f'generated/{dataset}/{level}/synth_hyacinth_divide_and_conquer_trial_{trial}_cycStd.csv')
                    if trial == 0:
                        with open(f'generated/{dataset}/{level}/denoiser_calls_divide_and_conquer_cycStd.txt') as file:
                            queries = int(file.readline())
                else:
                    df_synth = pd.read_csv(
                        f'generated/{dataset}/{level}/synth_hyacinth_pipeline_trial_{trial}_cycStd.csv')
                    if trial == 0:
                        with open(f'generated/{dataset}/{level}/denoiser_calls_pipeline_cycStd.txt') as file:
                            queries = int(file.readline())
                df_synth = df_synth.drop(columns=['Unnamed: 0'])
                df_synth_cleaned = preprocessor.cleanDataset(dataset, df_synth)
                df_synth_cleaned_selected = df_synth_cleaned[non_hier_cols]
                MSE = ((df_synth_cleaned_selected - df_real_cleaned_selected) ** 2).mean().mean()
                mses.append(MSE)

            mses = np.array(mses)
            AVG_MSE = np.mean(mses)
            STD = np.std(mses)
            row = {'Dataset': dataset, 'Parallelism': parallel, 'Level': level, 'Queries': queries, 'Avg. MSE': AVG_MSE,
                   'Std. MSE': STD}
            ablation_data_parallelism.loc[len(ablation_data_parallelism)] = row

    path = "experiments/ablations/parallelism/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "ablation_parallelism.csv")
    ablation_data_parallelism.to_csv(final_path)