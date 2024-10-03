import argparse
import pandas as pd
from data_utils import Preprocessor
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-synth_mask', type=str, help="(C)oarse (M)edium or (F)ine", required=True)
    parser.add_argument('-n_trials', type=int, help="number of trials", default=5)
    args = parser.parse_args()
    dataset = args.dataset
    preprocessor_prop = Preprocessor(dataset, True)
    preprocessor_std = Preprocessor(dataset, False)
    mses = {'prop': [], 'std': []}
    for trial in range(args.n_trials):
        df_real = pd.read_csv(f'generated/MetroTraffic/{args.synth_mask}/real.csv').drop(columns=['Unnamed: 0'])
        df_synth_prop = pd.read_csv(f'generated/MetroTraffic/M/synth_hyacinth_1_trial_{trial}_prop.csv').drop(columns=['Unnamed: 0'])
        df_synth_std = pd.read_csv(f'generated/MetroTraffic/M/synth_hyacinth_1_trial_{trial}.csv').drop(columns=['Unnamed: 0'])

        df_real_cyc = preprocessor_std.cleanDataset(args.dataset, df_real)
        df_synth_prop_cyc = preprocessor_std.cleanDataset(args.dataset, df_synth_prop)
        df_synth_std_cyc = preprocessor_std.cleanDataset(args.dataset, df_synth_std)
        df_synth_prop_cyc.columns = df_real_cyc.columns
        df_synth_std_cyc.columns = df_real_cyc.columns
        hierarchical_column_indices = np.array([df_real_cyc.columns.get_loc(col) for col in preprocessor_std.hierarchical_features_cyclic])
        remaining_indices = np.setdiff1d(np.arange(len(df_real_cyc.columns)), hierarchical_column_indices)
        non_hier_cols = np.array(remaining_indices)
        mse = np.mean((df_real_cyc.iloc[:, non_hier_cols] - df_synth_prop_cyc.iloc[:, non_hier_cols])**2)
        mse2 = np.mean((df_real_cyc.iloc[:, non_hier_cols] - df_synth_std_cyc.iloc[:, non_hier_cols])**2)
        mses['prop'].append(mse)
        mses['std'].append(mse2)

    print(f'Avg. prop MSE: {np.mean(np.array(mses['prop']))} +/- {np.std(np.array(mses['prop']))}')
    print(f'Avg. std MSE: {np.mean(np.array(mses['std']))} +/- {np.std(np.array(mses['std']))}')
    # plt.plot(df_real['temp'], c='black')
    # plt.plot(df_synth_prop['temp'], c='green')
    # plt.plot(df_synth_std['temp'], c='orange')
    # plt.show()