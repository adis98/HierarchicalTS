import pandas as pd
import torch

from data_utils import datasets, CyclicEncoder, Preprocessor
from copy import deepcopy
import argparse
import numpy as np
from diffusion_backbones import BiLinearDiffusionBackbone, TransformerDiffusionBackbone
import torch.optim as optim
from torch import nn, from_numpy
from torch.utils.data import Dataset, DataLoader
from TSImputers.SSSDS4Imputer import SSSDS4Imputer, SSSDS4Weaver
from TSImputers.TimeGAN import TimeGAN


class MyDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def fetchModel(in_features, out_features, args):
    model = None
    if args.backbone.lower() == 'bilinear':
        model = BiLinearDiffusionBackbone(in_dim=in_features, t_dim=1, h_dim=args.hdim, n_layers=args.layers)
    elif args.backbone.lower() == 'transformer':
        model = TransformerDiffusionBackbone(in_dim=in_features, t_dim=1, h_dim=args.hdim, n_layers=args.layers)
    elif args.backbone.lower() == 's4':
        model = SSSDS4Imputer(in_features, args.res_channels, args.skip_channels,
                              out_features, args.num_res_layers, args.diff_step_embed_in,
                              args.diff_step_embed_mid, args.diff_step_embed_out,
                              args.s4_lmax, args.s4_dstate, args.s4_dropout,
                              args.s4_bidirectional, args.s4_layernorm)
    elif args.backbone.lower() == 's4weaver':
        model = SSSDS4Weaver(in_features, args.res_channels, args.skip_channels, out_features,
                             args.num_res_layers, args.diff_step_embed_in,
                             args.diff_step_embed_mid, args.diff_step_embed_out,
                             args.s4_lmax, args.s4_dstate, args.s4_dropout,
                             args.s4_bidirectional, args.s4_layernorm)

    elif args.backbone.lower() == 'timegan':
        model = TimeGAN(args, None)
    return model


def fetchDiffusionConfig(args):
    betas = np.linspace(args.beta_0, args.beta_T, args.timesteps).reshape((-1, 1))
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas).reshape((-1, 1))
    diffusion_config = {'betas': from_numpy(betas).float(), 'alpha_bars': from_numpy(alpha_bars).float(),
                        'alphas': from_numpy(alphas).float(), 'T': args.timesteps}
    return diffusion_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='Transformer')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=128, help='hidden embedding dimension')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    args = parser.parse_args()
    dataset = args.dataset
    preprocessor = Preprocessor(dataset)
    cols = preprocessor.df_cleaned.columns
    hierarchical_cols = ["year", "month", "day"]
    temp = [x + '_sine' for x in hierarchical_cols]
    temp2 = [x + '_cos' for x in hierarchical_cols]
    temp.extend(temp2)
    metadata = preprocessor.df_cleaned[temp]
    # metadata.to_csv('orig_meta_metrotraffic.csv')
    # exit()
    real_metadata = deepcopy(metadata).iloc[np.random.permutation(len(metadata))]
    # real_metadata = deepcopy(metadata)  # no shuffling of samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_data = real_metadata.values
    in_features = training_data.shape[1]
    diffusion_config = fetchDiffusionConfig(args)
    model = fetchModel(in_features, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    # model = Model(diffusion_params, backbone, model_params)
    dataset = MyDataset(from_numpy(training_data).float())
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            timesteps = torch.randint(diffusion_config['T'], size=(batch.shape[0],))
            sigmas = torch.normal(0, 1, size=batch.shape)
            """Forward noising"""
            alpha_bars = diffusion_config['alpha_bars']
            batch_noised = torch.sqrt(alpha_bars[timesteps]) * batch + torch.sqrt(1 - alpha_bars[timesteps]) * sigmas
            batch_noised = batch_noised.to(device)
            timesteps_normalized = (timesteps / diffusion_config['T']).reshape((-1, 1))
            timesteps_normalized = timesteps_normalized.to(device)
            sigmas_predicted = model(batch_noised, timesteps_normalized)
            optimizer.zero_grad()
            sigmas = sigmas.to(device)
            loss = criterion(sigmas_predicted, sigmas)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print(f'epoch: {epoch}, loss: {total_loss}')

    """Synthesis"""
    data = torch.normal(0, 1, size=training_data.shape).to(device)
    with torch.no_grad():
        for step in range(diffusion_config['T'] - 1, -1, -1):
            print(f"backward step: {step}")
            times = torch.full(size=(training_data.shape[0], 1), fill_value=step)
            times_normalized = (times / (diffusion_config['T'])).to(device)
            epsilon_pred = model(data, times_normalized)
            difference_coeff = diffusion_config['betas'][step] / torch.sqrt(1 - diffusion_config['alpha_bars'][step])
            denom = diffusion_config['alphas'][step]
            sigma = diffusion_config['betas'][step] * (1 - diffusion_config['alpha_bars'][step - 1]) / (
                    1 - diffusion_config['alpha_bars'][step])
            sigma = torch.sqrt(sigma) * torch.normal(0, 1, training_data.shape)
            sigma = sigma.to(device)
            difference_coeff = difference_coeff.to(device)
            denom = denom.to(device)
            data = (data - difference_coeff * epsilon_pred) / denom + sigma
    synth = data.cpu().numpy()
    synth_meta_dataframe = pd.DataFrame(synth, columns=metadata.columns)
    synth_meta_decoded = preprocessor.cyclicDecode(synth_meta_dataframe)
    synth_meta_decoded = synth_meta_decoded.sort_values(by=hierarchical_cols).reset_index(drop=True)
    # synth_meta_decoded.drop(columns=['Unnamed:0'], inplace=True)
    synth_meta_decoded.to_csv('synths/synth_meta_metrotraffic.csv')
    real = preprocessor.df_orig
    real = real[hierarchical_cols]
    real = real.sort_values(by=hierarchical_cols).reset_index(drop=True)
    # real.drop(columns=['Unnamed:0'], inplace=True)
    real.to_csv('synths/real_meta_metrotraffic.csv')
    print('finished')
