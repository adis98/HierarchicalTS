import argparse

import torch

from Utility_models.Transformer import Transformer
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from torch import device, cuda, from_numpy, optim, nn, zeros_like
from main import Preprocessor
from training import MyDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-univariate', type=bool, help='univariate or multivariate task', default=False)
    parser.add_argument('-batch_size', type=int, help='batch size', default=16)
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')

    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if cuda.is_available() else 'cpu')
    constraints = {'day': 13}
    path = f'generated/{args.dataset}/{str(constraints)}/'

    synthetic_df = pd.read_csv(f'{path}synth_autoregressive_stride_{args.stride}.csv').drop(columns=['Unnamed: 0'])
    real_df = pd.read_csv(f'{path}real.csv').drop(columns=['Unnamed: 0'])

    preprocessor = Preprocessor(dataset)
    real_df_cyclic_normalized = preprocessor.scale(preprocessor.cyclicEncode(real_df))
    synthetic_df_cyclic_normalized = preprocessor.scale(preprocessor.cyclicEncode(synthetic_df))
    horizon_forecast_split = 0.8  # 80% of a window's data is used for the horizon and 20% is used as the forecast per window
    test_samples = [real_df_cyclic_normalized.iloc[i: i + args.window_size, :] for i in range(0, len(real_df_cyclic_normalized)-args.window_size + 1, 1)]
    train_samples = [synthetic_df_cyclic_normalized.iloc[i: i + args.window_size, :] for i in range(0, len(synthetic_df_cyclic_normalized)-args.window_size + 1, 1)]
    # if args.window_size > len(synthetic_df_cyclic_normalized) or args.window_size > len(real_df_cyclic_normalized):
    #     train_samples.append(synthetic_df_cyclic_normalized.iloc[:, :])
    #     test_samples.append(synthetic_df_cyclic_normalized.iloc[:, :])
    train_dataset = MyDataset(from_numpy(np.array(train_samples)).float())
    test_dataset = MyDataset(from_numpy(np.array(test_samples)).float())
    hierarchical_column_indices = synthetic_df_cyclic_normalized.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    all_indices = np.arange(len(synthetic_df_cyclic_normalized.columns))
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    non_hier_cols = np.array(remaining_indices)
    forecaster = Transformer(feature_size=train_dataset.inputs.shape[2], out_size=len(non_hier_cols))

    """TRAINING"""
    optimizer = optim.Adam(forecaster.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size)
    horizon_length = int(0.8 * train_dataset.inputs.shape[1])
    forecast_length = train_dataset.inputs.shape[1] - horizon_length
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader_train:
            loss_mask = zeros_like(batch)
            loss_mask[:, horizon_length:, non_hier_cols] = 1.0
            optimizer.zero_grad()
            prediction = forecaster(batch, device=device)
            boolean_loss_mask = loss_mask.bool()
            actual = batch[:, horizon_length:, non_hier_cols]
            pred = prediction[:, horizon_length:, :]
            loss = criterion(pred, actual)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f'EPOCH: {epoch}, LOSS: {total_loss}')

    """TESTING"""
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size)
    total_test_loss = 0.0
    forecaster.eval()
    with torch.no_grad():
        for batch in dataloader_test:
            prediction = forecaster(batch, device)
            loss_mask = zeros_like(batch)
            loss_mask[:, horizon_length:, non_hier_cols] = 1.0
            boolean_loss_mask = loss_mask.bool()
            actual = batch[:, horizon_length:, non_hier_cols]
            pred = prediction[:, horizon_length:, :]
            loss = criterion(pred, actual)
            total_test_loss += loss

        print(f"TEST LOSS: {total_test_loss}")

        """UNIVARIATE PLOT TEST"""
        sample = dataloader_test.dataset.inputs[0]
        predicted = forecaster(sample, device)
        actual = sample[:, non_hier_cols]
        predicted[:horizon_length, :] = actual[:horizon_length, :]
        plt.plot(predicted[:, 0], c='orange', label='forecast')
        plt.plot(actual[:, 0], c='green', label='real')
        plt.axvline(x=horizon_length-1, color='black', linestyle='--', label='Horizon/Forecast Boundary')
        plt.title(str(constraints))
        plt.legend()
        plt.show()


