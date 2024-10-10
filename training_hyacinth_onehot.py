import argparse

import torch
from autoencoders import TransformerAutoEncoderOneHot
from data_utils import PreprocessorOneHot
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np

from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='S4')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=64, help='hidden embedding dimension')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    parser.add_argument('-num_res_layers', type=int, default=4, help='the number of residual layers')
    parser.add_argument('-res_channels', type=int, default=64, help='the number of res channels')
    parser.add_argument('-skip_channels', type=int, default=64, help='the number of skip channels')
    parser.add_argument('-diff_step_embed_in', type=int, default=32, help='input embedding size diffusion')
    parser.add_argument('-diff_step_embed_mid', type=int, default=64, help='middle embedding size diffusion')
    parser.add_argument('-diff_step_embed_out', type=int, default=64, help='output embedding size diffusion')
    parser.add_argument('-s4_lmax', type=int, default=100)
    parser.add_argument('-s4_dstate', type=int, default=64)
    parser.add_argument('-s4_dropout', type=float, default=0.0)
    parser.add_argument('-s4_bidirectional', type=bool, default=True)
    parser.add_argument('-s4_layernorm', type=bool, default=True)

    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = PreprocessorOneHot(dataset)
    df = preprocessor.df_cleaned
    training_df = df.loc[preprocessor.train_indices]
    test_df = df.loc[preprocessor.test_indices]
    hierarchical_column_indices = training_df.columns.get_indexer(preprocessor.hierarchical_features_onehot)
    training_samples = []
    for i in range(0, len(training_df) - args.window_size + 1, args.stride):
        window = training_df.iloc[i:i + args.window_size].values.astype(np.float32)
        training_samples.append(window)

    training_dataset = MyDataset(from_numpy(np.array(training_samples)).float())
    diffusion_config = fetchDiffusionConfig(args)
    criterion = nn.MSELoss()
    criterion_cep = nn.CrossEntropyLoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    all_indices = np.arange(len(training_df.columns))

    # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    d_model = 2 * (len(preprocessor.df_orig.columns) - len(preprocessor.hierarchical_features))
    model_autoenc = TransformerAutoEncoderOneHot(input_dim=len(remaining_indices), d_model=d_model).to(device)
    # Convert to an ndarray
    optimizer_autoenc = optim.Adam(model_autoenc.parameters(), lr=args.lr)
    mapper = {}
    mseindices = np.arange(len(training_df.columns))
    for col, names in preprocessor.one_hot_mapper.items():
        mapper[col] = [training_df.columns.get_loc(name) for name in names]
        mseindices = np.setdiff1d(mseindices, mapper[col])
    mapper['msecols'] = mseindices
    """TRAINING AUTOENCODER"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            outs = model_autoenc(batch[:, :, remaining_indices])
            loss = 0.0
            start_index = 0
            optimizer_autoenc.zero_grad()
            for col, indices in mapper.items():
                if col not in preprocessor.hierarchical_features:
                    last_index = start_index + len(indices)
                    outs_sel = outs[:, :, start_index:last_index]
                    if col != 'msecols':
                        targets = torch.argmax(batch[:, :, indices], dim=-1)
                        outs_sel = outs_sel.view(-1, outs_sel.size(-1))
                        targets = targets.view(-1, )
                        loss += criterion_cep(outs_sel, targets)
                    else:
                        loss += len(indices) * criterion(outs_sel, batch[:, :, indices])
                    start_index = last_index

            loss.backward()
            total_loss += loss
            optimizer_autoenc.step()
        print(f'EPOCH {epoch}, LOSS: {total_loss}')

    latents = None
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            latent = model_autoenc.encode(batch[:, :, remaining_indices])
            latent = torch.cat((batch[:, :, hierarchical_column_indices], latent), 2)
            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), 0)

    latent_hierarchical_indices = np.arange(0, len(hierarchical_column_indices))
    latent_non_hierarchical_indices = np.arange(len(hierarchical_column_indices), latents.shape[2])
    latent_dataset = MyDataset(latents)
    latent_dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)

    in_dim = len(latent_hierarchical_indices) + len(latent_non_hierarchical_indices)
    out_dim = len(latent_non_hierarchical_indices)
    model = fetchModel(in_dim, out_dim, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    """TRAINING"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in latent_dataloader:
            batch = batch.to(device)
            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],)).to(device)
            sigmas = normal(0, 1, size=batch.shape).to(device)
            """Forward noising"""
            alpha_bars = diffusion_config['alpha_bars'].to(device)
            coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, latent_non_hierarchical_indices] = 0
            conditional_mask = from_numpy(conditional_mask).float().to(device)
            batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
            batch_noised = batch_noised.to(device)
            timesteps = timesteps.reshape((-1, 1))
            # timesteps = timesteps.to(device)
            sigmas_predicted = model(batch_noised, timesteps)
            optimizer.zero_grad()
            sigmas_permuted = sigmas[:, :, latent_non_hierarchical_indices].permute((0, 2, 1))
            sigmas_permuted = sigmas_permuted.to(device)
            loss = criterion(sigmas_predicted, sigmas_permuted)
            loss.backward()
            total_loss += loss
            optimizer.step()
        print(f'epoch: {epoch}, loss: {total_loss}')
    path = f'saved_models/{args.dataset}/'
    filename = "model_onehot.pth"
    filepath = os.path.join(path, filename)

    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model_autoenc.state_dict(), os.path.join(path, "model_autoenc_onehot.pth"))
    torch.save(model.state_dict(), filepath)
