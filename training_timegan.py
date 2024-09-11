import argparse

import torch

from data_utils import Preprocessor
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np

from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4, TimeGAN', default='TimeGAN')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('-num_layer', type=int, default=3, help='number of layers')
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    parser.add_argument('-propCycEnc', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned
    hierarchical_column_indices = df.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    training_samples = []
    for i in range(0, len(df) - args.window_size + 1, args.stride):
        window = df.iloc[i:i + args.window_size].values
        training_samples.append(window)

    in_dim = len(df.columns)
    out_dim = len(df.columns) - len(hierarchical_column_indices)
    args.in_dim = in_dim
    training_dataset = MyDataset(from_numpy(np.array(training_samples)).float())
    model = fetchModel(in_dim, out_dim, args).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size)
    all_indices = np.arange(len(df.columns))

    # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)

    # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    """TRAINING"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)