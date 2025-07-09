import pandas as pd
from data_utils import Preprocessor
import numpy as np
import os

if __name__ == "__main__":
    ablation_data_stitchfn = pd.DataFrame(
        columns=['Dataset', 'Loss', 'Level', 'Avg. MSE', 'Std. MSE'])
    for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
        preprocessor = Preprocessor(dataset, False)
        for method in ["mae", 'cosine', 'tcorr', 'mse']:
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
                    if "mae" in method:
                        df_synth = pd.read_csv(f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_correction_mae.csv')
                    elif "cosine" in method:
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_correction_cosine.csv')
                    elif "tcorr" in method:
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_correction_tcorr.csv')
                    else:
                        df_synth = pd.read_csv(
                            f'generated/{dataset}/{level}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_simplecoeff.csv')

                    df_synth = df_synth.drop(columns=['Unnamed: 0'])
                    df_synth_cleaned = preprocessor.cleanDataset(dataset, df_synth)
                    df_synth_cleaned_selected = df_synth_cleaned[non_hier_cols]
                    MSE = ((df_synth_cleaned_selected - df_real_cleaned_selected) ** 2).mean().mean()
                    mses.append(MSE)

                mses = np.array(mses)
                AVG_MSE = np.mean(mses)
                STD = np.std(mses)
                row = {'Dataset': dataset, 'Loss': method, 'Level': level, 'Avg. MSE': AVG_MSE,
                       'Std. MSE': STD}
                ablation_data_stitchfn.loc[len(ablation_data_stitchfn)] = row

    path = "experiments/ablations/stitchloss/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "ablation_stitchfn_wavestitch_grad_revision.csv")
    ablation_data_stitchfn.to_csv(final_path)
