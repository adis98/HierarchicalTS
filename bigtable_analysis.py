from data_utils import Preprocessor
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    datasets = ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]
    levels = ['C', 'M', 'F']
    bigtable = pd.DataFrame(
        columns=['Dataset', 'Method', 'Level', 'Avg. MSE', 'Std. MSE'])
    for dataset in datasets:
        preprocessor = Preprocessor(dataset, False)
        for level in levels:
            for method in ["TimeGAN", "TimeWeaver", "TSDiff-0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "Pipe-1", "Pipe-8", "Pipe-16", "Pipe-32"]:
                df_real = pd.read_csv(f"generated/{dataset}/{level}/real.csv").drop(columns=['Unnamed: 0'])
                df_real_cleaned = preprocessor.cleanDataset(dataset, df_real)
                non_hier_cols = [col for col in df_real_cleaned.columns if
                                 col not in preprocessor.hierarchical_features_cyclic]
                df_real_cleaned_selected = df_real_cleaned[non_hier_cols]
                mses = []
                for trial in range(5):
                    df_synth = None
                    if "TSDiff" in method:
                        strength = float(method.split('-')[1])
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_tsdiff_strength_{strength}_trial_{trial}.csv')

                    elif "Pipe" in method:
                        stride = int(method.split('-')[1])
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_hyacinth_pipeline_stride_{stride}_trial_{trial}_cycStd.csv')
                    elif method == "TimeWeaver":
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_timeweaver_trial_{trial}_cycStd.csv')
                    elif method == "TimeGAN":
                        df_synth = pd.read_csv(f'generated/{dataset}/{level}/synth_timegan_trial_{trial}_cycStd.csv')

                    df_synth = df_synth.drop(columns=['Unnamed: 0'])

                    df_synth_cleaned = preprocessor.cleanDataset(dataset, df_synth)
                    df_synth_cleaned_selected = df_synth_cleaned[non_hier_cols]
                    MSE = ((df_synth_cleaned_selected - df_real_cleaned_selected) ** 2).mean().mean()
                    mses.append(MSE)

                mses = np.array(mses)
                AVG_MSE = np.mean(mses)
                STD = np.std(mses)
                row = {'Dataset': dataset, 'Method': method, 'Level': level, 'Avg. MSE': AVG_MSE,
                       'Std. MSE': STD}

                bigtable.loc[len(bigtable)] = row

            path = "experiments/bigtable/"
            if not os.path.exists(path):
                os.makedirs(path)
            final_path = os.path.join(path, "bigtable.csv")
            bigtable.to_csv(final_path)


