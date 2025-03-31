import argparse
import numpy as np
import pandas as pd
from data_utils import Preprocessor
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
    dataset = 'BeijingAirQuality'
    preprocessor = Preprocessor(dataset, False)
    row = 0
    diffs = np.zeros((3, 4))
    stds = np.zeros((3, 4))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for task in ['C', 'M', 'F']:
            real = pd.read_csv(f'generated/{dataset}/{task}/real.csv')
            non_hier_cols = [col for col in real.columns if
                             col not in preprocessor.hierarchical_features_uncyclic and col != 'Unnamed: 0']
            filt = real[non_hier_cols]
            acs = {channel: np.zeros((1, 100)) for channel in filt.columns}
            acs_hyacinth = {channel: np.zeros((5, 100)) for channel in filt.columns}
            acs_tsdiff = {channel: np.zeros((5, 100)) for channel in filt.columns}
            acs_timeweaver = {channel: np.zeros((5, 100)) for channel in filt.columns}
            for lags in range(100):
                for channel in filt.columns:
                    acs[channel][0, lags] = (pd.Series(filt[channel]).autocorr(lag=lags))

            for trial in range(5):
                hyacinth = pd.read_csv(
                    f'generated/{dataset}/{task}/synth_wavestitch_pipeline_stride_8_trial_{trial}_cycStd_grad_simplecoeff.csv')
                tsdiff = pd.read_csv(f'generated/{dataset}/{task}/synth_tsdiff_strength_0.5_trial_{trial}.csv')
                timeweaver = pd.read_csv(f'generated/{dataset}/{task}/synth_timeweaver_trial_{trial}_cycStd.csv')
                filt_hyacinth = hyacinth[non_hier_cols]
                filt_tsdiff = tsdiff[non_hier_cols]
                filt_timeweaver = timeweaver[non_hier_cols]

                for lags in range(100):
                    for channel in filt.columns:
                        acs_hyacinth[channel][trial, lags] = (pd.Series(filt_hyacinth[channel]).autocorr(lag=lags))
                        acs_tsdiff[channel][trial, lags] = (pd.Series(filt_tsdiff[channel]).autocorr(lag=lags))
                        acs_timeweaver[channel][trial, lags] = (pd.Series(filt_timeweaver[channel]).autocorr(lag=lags))

            # for key in acs.keys():
            #     diffs[row, 0] = 0.0
            #     diffs[row, 1] = np.mean(abs(acs[key][0] - np.mean(acs_hyacinth[key], axis=0)))
            #     diffs[row, 2] = np.mean(abs(acs[key][0] - np.mean(acs_tsdiff[key], axis=0)))
            #     diffs[row, 3] = np.mean(abs(acs[key][0] - np.mean(acs_timeweaver[key], axis=0)))

            for key in acs.keys():
                axs[row, 0].plot(acs[key][0], label=key)
            #     axs[row, 0].set_title(f'{diffs[row, 0]: .2f}')
                axs[row, 1].plot(np.mean(acs_hyacinth[key], axis=0), label=key)
            #     axs[row, 1].set_title(f'{diffs[row, 1]: .2f}')
                axs[row, 2].plot(np.mean(acs_tsdiff[key], axis=0), label=key)
            #     axs[row, 2].set_title(f'{diffs[row, 2]: .2f}')
                axs[row, 3].plot(np.mean(acs_timeweaver[key], axis=0), label=key)
            #     axs[row, 3].set_title(f'{diffs[row, 3]: .2f}')
            row += 1

    axs[2, 0].set_xlabel('Real', fontweight="bold")
    axs[2, 1].set_xlabel('WaveStitch', fontweight="bold")
    axs[2, 2].set_xlabel('TSDiff', fontweight="bold")
    axs[2, 3].set_xlabel('TimeWeaver', fontweight="bold")
    axs[0, 0].set_ylabel('R', rotation=0, fontweight="bold")
    axs[1, 0].set_ylabel('I', rotation=0, fontweight="bold")
    axs[2, 0].set_ylabel('B', rotation=0, fontweight="bold")
    plt.savefig(f'acfplot{dataset}_wavestitch_grad_simplecoeff.pdf', bbox_inches='tight')
