import argparse
import numpy as np
import pandas as pd
from data_utils import Preprocessor
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-stride', type=int, default=1)
    parser.add_argument('-propCycEnc', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    constraints = {'month': 11, 'day':12, 'hour': 2}
    path = f'generated/{args.dataset}/{str(constraints)}/'

    if args.propCycEnc:
        synthetic_df = pd.read_csv(f'{path}synth_dnq_stride_{args.stride}_prop.csv').drop(columns=['Unnamed: 0'])
    else:
        synthetic_df = pd.read_csv(f'{path}synth_dnq_stride_{args.stride}.csv').drop(columns=['Unnamed: 0'])
    real_df = pd.read_csv(f'{path}real.csv').drop(columns=['Unnamed: 0'])
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    real_df = real_df[preprocessor.df_orig.columns]
    real_df_cyclic_normalized = preprocessor.scale(preprocessor.cyclicEncode(real_df))
    synthetic_df_cyclic_normalized = preprocessor.scale(preprocessor.cyclicEncode(synthetic_df))
    hierarchical_column_indices = synthetic_df_cyclic_normalized.columns.get_indexer(
        preprocessor.hierarchical_features_cyclic)
    all_indices = np.arange(len(synthetic_df_cyclic_normalized.columns))
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    non_hier_cols = np.array(remaining_indices)

    corr_real = real_df_cyclic_normalized.iloc[:, non_hier_cols].corr()
    corr_synth = synthetic_df_cyclic_normalized.iloc[:, non_hier_cols].corr()
    corr_real.fillna(0, inplace=True)
    corr_synth.fillna(0, inplace=True)
    for column in corr_real.columns:
        corr_real.loc[column, column] = 1
        corr_synth.loc[column, column] = 1
    corr_diff = corr_real - corr_synth

    # Step 4: Plot the three matrices side-by-side in a single figure

    plt.figure(figsize=(21, 6))  # Adjust figure size to accommodate 3 plots

    # Plot for real data
    plt.subplot(1, 3, 1)
    sns.heatmap(corr_real, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Real Data Correlation Matrix')

    # Plot for synthetic data
    plt.subplot(1, 3, 2)
    sns.heatmap(corr_synth, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Synthetic Data Correlation Matrix')

    # Plot for difference
    plt.subplot(1, 3, 3)
    sns.heatmap(corr_diff, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Difference Between Correlation Matrices')

    plt.tight_layout()  # Ensures the plots don’t overlap
    plt.show()
