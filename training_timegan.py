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
    all_indices = np.arange(len(df.columns))

    # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
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
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    """TRAINING AUTOENCODER"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, non_hier_cols] = 0
            conditional_mask = from_numpy(conditional_mask).float().to(device)
            bool_mask = conditional_mask.bool()
            batch = batch.to(device)
            embed = model.nete(batch)
            recovered = model.netr(embed)
            model.optimizer_e.zero_grad()
            model.optimizer_r.zero_grad()
            loss = model.l_mse(recovered[~bool_mask], batch[~bool_mask])
            loss.backward(retain_graph=True)
            model.optimizer_e.step()
            model.optimizer_r.step()
            total_loss += loss.detach().numpy()
        print(f'AUTOENCODER EPOCH {epoch}, LOSS: {total_loss}')

    """TRAINING SUPERVISOR"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, non_hier_cols] = 0
            conditional_mask = from_numpy(conditional_mask).float().to(device)
            bool_mask = conditional_mask.bool()
            batch = batch.to(device)
            embed = model.nete(batch)
            out = model.nets(embed)
            model.optimizer_s.zero_grad()
            loss = model.l_mse(out[:, :-1, non_hier_cols], batch[:, 1:, non_hier_cols])
            loss.backward(retain_graph=True)
            model.optimizer_s.step()
            total_loss += loss.detach().numpy()
        print(f'RECOVERY EPOCH {epoch}, LOSS: {total_loss}')

    """TRAINING GENERATOR"""
    for epoch in range(args.epochs):
        total_loss_g, total_loss_er, total_loss_d = 0.0, 0.0, 0.0
        for batch in dataloader:
            """generator part"""
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, non_hier_cols] = 0
            conditional_mask = from_numpy(conditional_mask).float().to(device)
            bool_mask = conditional_mask.bool()
            batch = batch.to(device)
            z = batch.clone.detach()
            z[:, :, non_hier_cols] = torch.normal(0, 1, batch[:, :, non_hier_cols].shape).to(device)
            out_e = model.nete(batch)
            out_se = model.nets(out_e)
            out_g = model.netg(z)
            out_sg = model.nets(out_g)
            out_rsg = model.netr(out_sg)
            y_fake = model.netd(out_sg)
            y_fake_e = model.netd(out_g)
            model.optimizer_g.zero_grad()
            model.optimizer_s.zero_grad()
            err_g_U = model.l_bce(y_fake, torch.ones_like(y_fake))
            err_g_U_e = model.l_bce(y_fake_e, torch.ones_like(y_fake_e))
            err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(out_rsg[:, :, non_hier_cols],[0])[1] + 1e-6) - torch.sqrt(torch.std(batch[:, :, non_hier_cols],[0])[1] + 1e-6)))   # |a^2 - b^2|
            err_g_V2 = torch.mean(torch.abs((torch.mean(out_rsg[:, :, non_hier_cols],[0])[0]) - (torch.mean(batch[:, :, non_hier_cols],[0])[0])))  # |a - b|
            err_s = model.l_mse(out_se[:, :-1, non_hier_cols], out_e[:, 1:, non_hier_cols])
            loss_g = err_g_U + err_g_U_e * 1 + err_g_V1 * 100 + err_g_V2 * 100 + torch.sqrt(err_s)
            loss_g.backward(retain_graph=True)
            total_loss_g += loss_g.detach().numpy()
            model.optimizer_s.step()
            model.optimizer_g.step()

            """er part"""
            out_er = model.netr(out_e)
            out_se = model.nets(out_e)
            model.optimizer_e.zero_grad()
            model.optimizer_r.zero_grad()
            err_er_ = model.l_mse(out_er[:, :, non_hier_cols], batch[:, :, non_hier_cols])
            err_s = model.l_mse(out_se[:, :-1, non_hier_cols], out_e[:, 1:, non_hier_cols])
            loss_er = 10 * torch.sqrt(err_er_) + 0.1 * err_s
            loss_er.backward(retain_graph=True)
            model.optimizer_e.step()
            model.optimizer_r.step()
            total_loss_er += loss_er.detach().numpy()

            """discriminator part"""
            out_e = model.nete(batch)
            z = batch.clone.detach()
            z[:, :, non_hier_cols] = torch.normal(0, 1, batch[:, :, non_hier_cols].shape).to(device)
            out_g = model.netg(z)
            out_sg = model.nets(out_g)
            y_real = model.netd(out_e)
            y_fake = model.netd(out_sg)
            y_fake_e = model.netd(out_g)
            err_d_real = model.l_bce(y_real, torch.ones_like(y_real))
            err_d_fake = model.l_bce(y_fake, torch.zeros_like(y_fake))
            err_d_fake_e = model.l_bce(y_fake_e, torch.zeros_like(y_fake_e))
            loss_d = err_d_real + err_d_fake + err_d_fake_e * 1
            model.optimizer_d.zero_grad()
            if loss_d > 0.15:
                loss_d.backward(retain_graph=True)
            model.optimizer_d.step()
            total_loss_d = loss_d.detach.numpy()

        print(f'GENERATOR EPOCH {epoch}, LOSS_G: {total_loss_g}, LOSS_ER: {total_loss_er}, LOSS_D: {total_loss_d}')


