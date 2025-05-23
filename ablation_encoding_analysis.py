import os

import numpy as np
import pandas as pd
from data_utils import Preprocessor, PreprocessorOrdinal, PreprocessorOneHot

if __name__ == "__main__":
    datasets = ["MetroTraffic", "AustraliaTourism", "PanamaEnergy", "RossmanSales", "BeijingAirQuality"]
    encodings = ["STD", "ORD", "OHE"]
    ablation_encoding_csv = pd.DataFrame(columns=["Dataset", "Encoding", "Mask", "Avg. MSE", "Std. MSE", "stride", "indim"])
    mask_levels = ["C", "M", "F"]
    num_trials = 5
    for dataset in datasets:
        preprocessor_cycstd = Preprocessor(dataset, False)
        preprocessor_ord = PreprocessorOrdinal(dataset)
        preprocessor_ohe = PreprocessorOneHot(dataset)
        preprocessors = {"STD": preprocessor_cycstd, "ORD": preprocessor_ord, "OHE": preprocessor_ohe}
        non_hierarchical_features = [col for col in preprocessor_ord.df_orig.columns if col not in preprocessor_ord.hierarchical_features]
        for mask in mask_levels:
            dir_path = f"generated/{dataset}/{mask}/"
            path_real = os.path.join(dir_path, "real.csv")
            df_real = pd.read_csv(path_real).drop(columns=['Unnamed: 0'])
            df_real_scaled = preprocessors["ORD"].cleanDataset(dataset, df_real)
            non_hierarchical_features = [col for col in df_real.columns if col not in preprocessor_ord.hierarchical_features]
            for stride in [8]:
                for encoding in encodings:
                    hit_accuracies = []
                    mses = []
                    indim = preprocessors[encoding].df_cleaned.values.shape[1]
                    for trial in range(num_trials):
                        if encoding == "OHE":
                            path_synth = os.path.join(dir_path, f"synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_onehot_grad_simplecoeff.csv")
                        elif encoding == "ORD":
                            path_synth = os.path.join(dir_path, f'synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_ordinal_grad_simplecoeff.csv')
                        elif encoding == "STD":
                            path_synth = os.path.join(dir_path, f'synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_cycStd_grad_simplecoeff.csv')

                        df_synth = pd.read_csv(path_synth).drop(columns=['Unnamed: 0'])
                        cat_non_hier_columns = [col for col in non_hierarchical_features if col in preprocessor_ohe.onehot_encoded_columns]
                        real_non_hier_columns = [col for col in non_hierarchical_features if col not in preprocessor_ohe.onehot_encoded_columns]
                        if len(cat_non_hier_columns) > 0:
                            df_real_cat_filtered = df_real[cat_non_hier_columns]
                            df_synth_cat_filtered = df_synth[cat_non_hier_columns]
                            matched = (df_real_cat_filtered == df_synth_cat_filtered) | (pd.isna(df_real_cat_filtered) & pd.isna(df_synth_cat_filtered))
                            hit_accuracy = matched.sum().sum()/matched.size
                            hit_accuracies.append(hit_accuracy)

                        df_synth_scaled_rnh = preprocessors["ORD"].cleanDataset(dataset, df_synth)[real_non_hier_columns]
                        df_real_scaled_rnh = df_real_scaled[real_non_hier_columns]
                        mse = ((df_synth_scaled_rnh - df_real_scaled_rnh) ** 2).mean().mean()
                        mses.append(mse)

                    hit_accuracies = np.array(hit_accuracies)
                    mses = np.array(mses)
                    if len(hit_accuracies) > 0:
                        AVG_ACC, ACC_STDDEV = np.mean(hit_accuracies), np.std(hit_accuracies)
                    else:
                        AVG_ACC, ACC_STDDEV = np.nan, np.nan
                    AVG_MEAN, MEAN_STDDEV = np.mean(mses), np.std(mses)

                    row = {"Dataset": dataset, "Encoding": encoding, "Mask": mask,
                           "Avg. MSE": AVG_MEAN, "Std. MSE": MEAN_STDDEV, "stride": stride, 'indim': indim}

                    ablation_encoding_csv.loc[len(ablation_encoding_csv)] = row

    path = "experiments/ablations/encoding/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "ablation_encoding_wavestitch_grad_simplecoeff.csv")
    ablation_encoding_csv.to_csv(final_path)
