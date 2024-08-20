import argparse

import torch

from main import Preprocessor
from training import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np

from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='S4')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset)
    df = preprocessor.df_cleaned
    hierarchical_column_indices = df.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    training_samples = []
    for i in range(0, len(df) - args.window_size + 1, args.stride):
        window = df.iloc[i:i + args.window_size].values
        training_samples.append(window)

    in_dim = len(df.columns)
    out_dim = len(df.columns) - len(hierarchical_column_indices)
    training_dataset = MyDataset(from_numpy(np.array(training_samples)).float())
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size)
    all_indices = np.arange(len(df.columns))

    # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)

    # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    with torch.no_grad():
        for step in range(diffusion_config['T'] - 1, -1, -1):
            print(f"backward step: {step}")
            times = torch.full(size=(training_data.shape[0], 1), fill_value=step)
            epsilon_pred = model(data, times_normalized)
            difference_coeff = diffusion_config['betas'][step] / torch.sqrt(1 - diffusion_config['alpha_bars'][step])
            denom = diffusion_config['alphas'][step]
            sigma = diffusion_config['betas'][step] * (1 - diffusion_config['alpha_bars'][step - 1]) / (
                    1 - diffusion_config['alpha_bars'][step])
            for batch in dataloader:
                batch = batch.to(device)
                timesteps = randint(diffusion_config['T'], size=(batch.shape[0],)).to(device)
                sigmas = normal(0, 1, size=batch.shape).to(device)
                """Forward noising"""
                alpha_bars = diffusion_config['alpha_bars'].to(device)
                coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
                coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
                conditional_mask = np.ones(batch.shape)
                conditional_mask[:, :, non_hier_cols] = 0
                conditional_mask = from_numpy(conditional_mask).float().to(device)
                batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
                batch_noised = batch_noised.to(device)
                timesteps = timesteps.reshape((-1, 1))
                # timesteps = timesteps.to(device)
                sigmas_predicted = model(batch_noised, timesteps)