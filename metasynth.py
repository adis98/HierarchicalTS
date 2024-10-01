import numpy as np
import pandas as pd
import argparse
from data_utils import Preprocessor, CyclicEncoder, datasets
import itertools


def metaSynthHyacinth(hierarchical_feats, df):
    df_meta = df[hierarchical_feats]
    unique_values = {col: sorted(df_meta[col].unique()) for col in df_meta.columns}
    combinations = list(itertools.product(*unique_values.values()))
    hierarchical_df = pd.DataFrame(combinations, columns=unique_values.keys())
    merged_df = hierarchical_df.merge(df, how='outer', on=hierarchical_feats, indicator=True)
    return merged_df


def metaSynthTimeWeaver(constraints, hierarchical_feats, df):
    df_meta = df[hierarchical_feats]
    unique_values = {col: sorted(df_meta[col].unique()) for col in df_meta.columns if col not in constraints}
    for key in constraints.keys():
        unique_values[key] = [constraints[key]]
    combinations = list(itertools.product(*unique_values.values()))
    hierarchical_df = pd.DataFrame(combinations, columns=unique_values.keys())
    for column in df.columns:
        if column not in hierarchical_feats:
            hierarchical_df[column] = np.NAN
    return hierarchical_df


def metadataMask(metadata, synthmask, dataset):
    if dataset == "MetroTraffic":
        if synthmask == "C":
            return metadata['year'] == 2018
        elif synthmask == "M":
            return (metadata['year'] == 2018) & (metadata['day'] == 15)
        elif synthmask == "F":
            return (metadata['year'] == 2018) & (metadata['hour'] == 6)
