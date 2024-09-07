import argparse

import torch

from Utility_models.Transformer import Transformer
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from torch import device, cuda, from_numpy, optim, nn, zeros_like
from data_utils import Preprocessor
from training_utils import MyDataset
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

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
    parser.add_argument('-propCycEnc', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if cuda.is_available() else 'cpu')
    constraints = {'year': 2013}
    path = f'generated/{args.dataset}/{str(constraints)}/'

    if args.propCycEnc:
        synthetic_df = pd.read_csv(f'{path}synth_dnq_stride_{args.stride}_prop.csv').drop(columns=['Unnamed: 0'])
    else:
        synthetic_df = pd.read_csv(f'{path}synth_dnq_stride_{args.stride}.csv').drop(columns=['Unnamed: 0'])
    real_df = pd.read_csv(f'{path}real.csv').drop(columns=['Unnamed: 0'])

    # filtered_real = real_df[real_df['day'] == 7].reset_index()
    # filtered_synth = synthetic_df[synthetic_df['day'] == 7].reset_index()
    # plt.plot(filtered_real['traffic_volume'], c='red')
    # plt.plot(filtered_synth['traffic_volume'], c='green')
    # plt.show()
    # exit()
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    real_df = real_df[preprocessor.df_orig.columns]
    real_df_cyclic_normalized = preprocessor.scale(preprocessor.cyclicEncode(real_df))
    synthetic_df_cyclic_normalized = preprocessor.scale(preprocessor.cyclicEncode(synthetic_df))
    hierarchical_column_indices = synthetic_df_cyclic_normalized.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    all_indices = np.arange(len(synthetic_df_cyclic_normalized.columns))
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    non_hier_cols = np.array(remaining_indices)

    # corr_real = real_df_cyclic_normalized.iloc[:, non_hier_cols].corr()
    # corr_synth = synthetic_df_cyclic_normalized.iloc[:, non_hier_cols].corr()
    #
    # corr_diff = corr_real - corr_synth
    #
    # # Step 4: Plot the three matrices side-by-side in a single figure
    #
    # plt.figure(figsize=(21, 6))  # Adjust figure size to accommodate 3 plots
    #
    # # Plot for real data
    # plt.subplot(1, 3, 1)
    # sns.heatmap(corr_real, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title('Real Data Correlation Matrix')
    #
    # # Plot for synthetic data
    # plt.subplot(1, 3, 2)
    # sns.heatmap(corr_synth, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title('Synthetic Data Correlation Matrix')
    #
    # # Plot for difference
    # plt.subplot(1, 3, 3)
    # sns.heatmap(corr_diff, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title('Difference Between Correlation Matrices')
    #
    # plt.tight_layout()  # Ensures the plots don’t overlap
    # plt.show()
    # exit()

    horizon_forecast_split = 0.8  # 80% of a window's data is used for the horizon and 20% is used as the forecast per window
    test_samples = [real_df_cyclic_normalized.iloc[i: i + args.window_size, :] for i in range(0, len(real_df_cyclic_normalized)-args.window_size + 1, 1)]
    train_samples = [synthetic_df_cyclic_normalized.iloc[i: i + args.window_size, :] for i in range(0, len(synthetic_df_cyclic_normalized)-args.window_size + 1, 1)]
    # if args.window_size > len(synthetic_df_cyclic_normalized) or args.window_size > len(real_df_cyclic_normalized):
    #     train_samples.append(synthetic_df_cyclic_normalized.iloc[:, :])
    #     test_samples.append(synthetic_df_cyclic_normalized.iloc[:, :])
    train_dataset = MyDataset(from_numpy(np.array(train_samples)).float())
    test_dataset = MyDataset(from_numpy(np.array(test_samples)).float())
    forecaster = Transformer(feature_size=train_dataset.inputs.shape[2], out_size=train_dataset.inputs.shape[2])

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
            loss_mask[:, horizon_length:, 0] = 1.0
            optimizer.zero_grad()
            prediction = forecaster(batch, device=device)
            boolean_loss_mask = loss_mask.bool()
            # actual = batch[:, horizon_length:, non_hier_cols]
            actual = batch[boolean_loss_mask]
            pred = prediction[boolean_loss_mask]
            # pred = prediction[:, horizon_length:, :]
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
            loss_mask[:, horizon_length:, 0] = 1.0
            boolean_loss_mask = loss_mask.bool()
            actual = batch[boolean_loss_mask]
            pred = prediction[boolean_loss_mask]
            loss = criterion(pred, actual)
            total_test_loss += loss

        print(f"TEST LOSS: {total_test_loss}")

        """UNIVARIATE PLOT TEST"""
        for batch in dataloader_test:
            sample = batch[0]
            predicted = forecaster(sample, device)
            actual = sample[:, :]
            # predicted[:horizon_length, :] = actual[:horizon_length, :]
            plt.plot(predicted[:, 0], c='orange', label='forecast')
            plt.plot(actual[:, 0], c='green', label='real')
            plt.axvline(x=horizon_length-1, color='black', linestyle='--', label='Horizon/Forecast Boundary')
            plt.title(str(constraints))
            plt.legend()
            plt.show()
            break


