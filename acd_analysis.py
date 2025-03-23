import pandas as pd
import numpy as np
from data_utils import Preprocessor
import os
import warnings
lags = 100
if __name__ == "__main__":
    acdtable = pd.DataFrame(
        columns=['Dataset', 'Method', 'Level', 'Avg. ACD', 'Std. ACD'])
    for dataset in ["MetroTraffic", "RossmanSales", "BeijingAirQuality", "AustraliaTourism", "PanamaEnergy"]:
        preprocessor = Preprocessor(dataset, False)
        for method in ['algo-8', 'algo-16', 'algo-32', 'algo-1', "timegan", "timeweaver", "tsdiff-0.5"]:
            for mask in ['C', 'M', 'F']:
                real = pd.read_csv(f'generated/{dataset}/{mask}/real.csv')
                non_hier_cols = [col for col in real.columns if
                                 col not in preprocessor.hierarchical_features_uncyclic and col != 'Unnamed: 0']

                filt = real[non_hier_cols].values
                stds = np.std(filt, axis=0)
                boolmask = stds == 0
                stds[boolmask] = 1.0
                filt_centered = (filt - np.mean(filt, axis=0))/stds
                autocorr_real = np.ones((len(non_hier_cols), lags))
                for lag in range(1, lags):
                    acf = np.mean(filt_centered[lag:, :] * filt_centered[:-lag, :], axis=0)
                    autocorr_real[:, lag] = acf

                MAES = []
                for trial in range(5):
                    if method == "timegan":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timegan_trial_{trial}_cycStd.csv')
                    elif method == "timeweaver":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timeweaver_trial_{trial}_cycStd.csv')
                    elif "tsdiff" in method:
                        strength = method.split('-')[1]
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_tsdiff_strength_{strength}_trial_{trial}.csv')
                    else:
                        stride = method.split('-')[1]
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_wavestitch_pipeline_stride_{stride}_trial_{trial}_cycStd_grad_simplecoeff.csv')
                    data = data[non_hier_cols].values
                    stds_meth = np.std(data, axis=0)
                    boolmask_meth = stds_meth == 0
                    stds_meth[boolmask_meth] = 1.0
                    data_centered = (data - np.mean(data, axis=0))/stds_meth
                    autocorr = np.ones((len(non_hier_cols), lags))
                    for lag in range(1, lags):
                        acf = np.mean(data_centered[lag:, :] * data_centered[:-lag, :], axis=0)
                        autocorr[:, lag] = acf

                    undefmask = boolmask_meth | boolmask
                    diff = np.abs(autocorr_real - autocorr)
                    complement = undefmask==False
                    diff_filtered = diff[complement, :]
                    MAE = np.mean(diff_filtered)
                    MAES.append(MAE)
                arr = np.array(MAES)
                avg = np.mean(arr)
                std = np.std(arr)
                if "algo" in method:
                    tech = 'wavestitch-'+method.split('-')[1]
                else:
                    tech = method
                row = {"Dataset": dataset, "Method": tech, "Level": mask, 'Avg. ACD': avg, 'Std. ACD': std}
                acdtable.loc[len(acdtable)] = row
    path = "experiments/acdtable/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "acdtable_wavestitch_grad_simplecoeff.csv")
    acdtable.to_csv(final_path)



