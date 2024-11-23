import pandas as pd
from matplotlib import pyplot as plt
from data_utils import Preprocessor
import seaborn as sns

if __name__ == "__main__":
    dataset = "PanamaEnergy"
    preprocessor = Preprocessor(dataset, False)
    fig, axs = plt.subplots(3, 4)
    non_hier_cols = [col for col in preprocessor.df_orig.columns if col not in preprocessor.hierarchical_features_uncyclic and col != 'Unnamed: 0']

    row = 0
    for task in ["C", "M", "F"]:
        real = pd.read_csv(f"generated/{dataset}/{task}/real.csv")[non_hier_cols]
        real_corr = real.corr()
        hyacinth_corr = real_corr.copy()
        tsdiff_corr = real_corr.copy()
        timeweaver_corr = real_corr.copy()
        sns.heatmap(real_corr, cmap='coolwarm', square=True, cbar=False, ax=axs[row, 0])
        for trial in range(5):
            hyacinth = pd.read_csv(f'generated/{dataset}/{task}/synth_hyacinth_pipeline_stride_8_trial_{trial}_cycStd.csv')[non_hier_cols]
            hyacinth_corr.iloc[:, :] = (hyacinth_corr * trial + hyacinth.corr())/(trial + 1)

            tsdiff = pd.read_csv(f'generated/{dataset}/{task}/synth_tsdiff_strength_0.5_trial_{trial}.csv')[non_hier_cols]
            tsdiff_corr.iloc[:, :] = (tsdiff_corr * trial + tsdiff.corr()) / (trial + 1)

            timeweaver = pd.read_csv(f'generated/{dataset}/{task}/synth_timeweaver_trial_{trial}_cycStd.csv')[non_hier_cols]
            timeweaver_corr.iloc[:, :] = (timeweaver_corr * trial + timeweaver.corr()) / (trial + 1)

        # axs[row, 0].set_title(f'0.0')
        # axs[row, 1].set_title(f'{(real_corr - hyacinth_corr).abs().mean().mean(): .2f}')
        # axs[row, 2].set_title(f'{(real_corr - tsdiff_corr).abs().mean().mean(): .2f}')
        # axs[row, 3].set_title(f'{(real_corr - timeweaver_corr).abs().mean().mean(): .2f}')
        axs[0, 0].set_ylabel('R', rotation=0, fontweight="bold")
        axs[1, 0].set_ylabel('I', rotation=0, fontweight="bold")
        axs[2, 0].set_ylabel('B', rotation=0, fontweight="bold")

        axs[2, 0].set_xlabel('Real', fontweight="bold")

        sns.heatmap(hyacinth_corr, cmap='coolwarm', square=True, cbar=False, ax=axs[row, 1])
        sns.heatmap(tsdiff_corr, cmap='coolwarm', square=True, cbar=False, ax=axs[row, 2])
        sns.heatmap(timeweaver_corr, cmap='coolwarm', square=True, cbar=False, ax=axs[row, 3])
        row += 1

    for ax in axs.flat:
        ax.set_xticks([])  # Disable x-ticks
        ax.set_yticks([])  # Disable y-ticks
    axs[2, 1].set_xlabel('WaveStitch', fontweight="bold")
    axs[2, 2].set_xlabel('TSDiff', fontweight="bold")
    axs[2, 3].set_xlabel('TimeWeaver', fontweight="bold")

    plt.savefig(f'crosscorrplot{dataset}.pdf', bbox_inches='tight')
