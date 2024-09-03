import pandas as pd
import argparse
from main import Preprocessor, CyclicEncoder, datasets
import itertools


def metaSynth(hierarchical_feats, df):
    df_meta = df[hierarchical_feats]
    unique_values = {col: sorted(df_meta[col].unique()) for col in df_meta.columns}
    combinations = list(itertools.product(*unique_values.values()))
    hierarchical_df = pd.DataFrame(combinations, columns=unique_values.keys())
    merged_df = hierarchical_df.merge(df, how='outer', on=hierarchical_feats, indicator=True)
    return merged_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)

    args = parser.parse_args()
    dataset = args.dataset
    preprocessor = Preprocessor(dataset)
    df = preprocessor.df_orig
    hierarchy = ['year', 'month', 'day', 'hour']
    df_meta = df[hierarchy]
    unique_values = {col: sorted(df_meta[col].unique()) for col in df_meta.columns}
    combinations = list(itertools.product(*unique_values.values()))
    hierarchical_df = pd.DataFrame(combinations, columns=unique_values.keys())
    hierarchical_df.to_csv("synth_meta_bruteforce.csv")
