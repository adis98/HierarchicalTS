from main import datasets, CyclicEncoder, Preprocessor
from copy import deepcopy
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str, help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='SSSDS4, AutoDiffuse', default='AutoDiffuse')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    args = parser.parse_args()
    dataset = args.dataset
    preprocessor = Preprocessor(dataset)
    cols = preprocessor.df_cleaned.columns
    hierarchical_cols = ["year", "day", "hour", "month"]
    temp = [x+'_sine' for x in hierarchical_cols]
    temp2 = [x+'_cos' for x in hierarchical_cols]
    temp.extend(temp2)
    metadata = preprocessor.df_cleaned[temp]
    real_metadata = deepcopy(metadata).iloc[np.random.permutation(len(metadata))]
    # model = Model(diffusion_params, backbone, model_params)
    print()
