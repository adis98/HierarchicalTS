import pandas as pd
from data_utils import Preprocessor
import numpy as np
import os

if __name__ == "__main__":
    ablation_data_parallelism = pd.DataFrame(
        columns=['Dataset', 'Stride', 'Mask', 'Avg. MSE', 'Std. MSE'])
    for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
        preprocessor = Preprocessor(dataset, False)
        for parallel in ["Pipe-8"]:
            for level in ["0.25", "0.50", "0.75"]:
                df_real = pd.read_csv(f"generated/{dataset}/{level}/real.csv").drop(columns=['Unnamed: 0'])
                df_real_cleaned = preprocessor.cleanDataset(dataset, df_real)
                non_hier_cols = [col for col in df_real_cleaned.columns if
                                 col not in preprocessor.hierarchical_features_cyclic]
                df_real_cleaned_selected = df_real_cleaned[non_hier_cols]
                mses = []
                for trial in range(5):
                    df_synth = pd.read_csv(
                        f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_correction.csv')

                    df_synth = df_synth.drop(columns=['Unnamed: 0'])
                    df_synth_cleaned = preprocessor.cleanDataset(dataset, df_synth)
                    df_synth_cleaned_selected = df_synth_cleaned[non_hier_cols]
                    MSE = ((df_synth_cleaned_selected - df_real_cleaned_selected) ** 2).mean().mean()
                    mses.append(MSE)

                mses = np.array(mses)
                AVG_MSE = np.mean(mses)
                STD = np.std(mses)
                row = {'Dataset': dataset, 'Stride': parallel, 'Mask': level, 'Avg. MSE': AVG_MSE,
                       'Std. MSE': STD}
                ablation_data_parallelism.loc[len(ablation_data_parallelism)] = row

    path = "experiments/ablations/randommissing/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "ablation_randommissing_wavestitch_grad_correction.csv")
    ablation_data_parallelism.to_csv(final_path)
