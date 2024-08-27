import argparse
import torch
from main import Preprocessor
from training import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


if __name__ == "__main__":
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
    preprocessor = Preprocessor(dataset)
    df = preprocessor.df_cleaned

    train_df_with_hierarchy = preprocessor.cyclicDecode(df)
    test_df_with_hierarchy = train_df_with_hierarchy.copy()
    hierarchical_column_indices = df.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    constraints = {'year': 2012, 'month': 10}  # determines which rows need synthetic data
    rows_to_synth = pd.Series([True] * len(test_df_with_hierarchy))
    # Iterate over the dictionary to create masks for each column
    for col, value in constraints.items():
        column_mask = test_df_with_hierarchy[col] == value
        rows_to_synth &= column_mask

    real_df = df.loc[rows_to_synth]
    """Approach 1: Divide and conquer"""
    test_samples = []
    mask_samples = []
    for i in range(0, len(df) - args.window_size + 1, args.window_size):
        window = df.iloc[i:i + args.window_size].values
        mask_window = rows_to_synth.iloc[i:i + args.window_size].values
        if any(mask_window):
            test_samples.append(window)
            mask_samples.append(mask_window)
    #
    in_dim = len(df.columns)
    out_dim = len(df.columns) - len(hierarchical_column_indices)
    test_dataset = MyDataset(from_numpy(np.array(test_samples)).float())
    mask_dataset = MyDataset(from_numpy(np.array(mask_samples)))
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch_size)
    all_indices = np.arange(len(df.columns))
    #
    # # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    #
    # # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    saved_params = torch.load(f'saved_models/{args.dataset}/model.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = False
    model.eval()

    with torch.no_grad():
        for (test_batch, mask_batch) in zip(test_dataloader, mask_dataloader):
            x = torch.normal(0, 1, test_batch.shape).to(device)
            for step in range(diffusion_config['T'] - 1, -1, -1):
                print(f"backward step: {step}")
                times = torch.full(size=(test_dataset.inputs.shape[0], 1), fill_value=step).to(device)
                alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                alpha_t = diffusion_config['alphas'][step].to(device)
                beta_t = diffusion_config['betas'][step].to(device)
                sampled_noise = torch.normal(0, 1, test_batch.shape).to(device)
                cached_denoising = torch.sqrt(alpha_bar_t) * test_batch + torch.sqrt(1 - alpha_bar_t) * sampled_noise
                mask_expanded = np.zeros_like(test_batch, dtype=bool)
                for channel in non_hier_cols:
                    mask_expanded[:, :, channel] = mask_batch
                x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
                x[~mask_expanded] = cached_denoising[~mask_expanded]
                epsilon_pred = model(x, times)
                epsilon_pred = epsilon_pred.permute((0, 2, 1))
                if step > 0:
                    vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
                                                                                             size=epsilon_pred.shape)
                else:
                    vari = 0.0

                normal_denoising = torch.normal(0, 1, test_batch.shape).to(device)
                normal_denoising[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - (
                        (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
                normal_denoising[:, :, non_hier_cols] += vari
                masked_binary = mask_batch.int()
                # x[mask_batch][:, non_hier_cols] = normal_denoising[mask_batch]
                x[mask_expanded] = normal_denoising[mask_expanded]
                x[~mask_expanded] = test_batch[~mask_expanded]
                # if step == 0:
                #     x[~mask_batch][:, non_hier_cols] = test_batch[~mask_batch][:, non_hier_cols]
                #     k = x[~mask_batch][:, non_hier_cols]
                #     print()

    synth = x.view(-1, test_samples[0].shape[1])
    df_synth_alt = pd.DataFrame(np.array(test_samples).reshape((-1, synth.shape[1])), columns=df.columns)
    df_synth = pd.DataFrame(synth.cpu().numpy(), columns=df.columns)
    real_df_reconverted = preprocessor.decode(real_df, rescale=True).reset_index(drop=True)
    decimal_accuracy = real_df_reconverted.apply(decimal_places).to_dict()
    synth_df_reconverted = preprocessor.decode(df_synth, rescale=True)

    rows_to_select_synth = pd.Series([True] * len(synth_df_reconverted))
    for col, value in constraints.items():
        column_mask = synth_df_reconverted[col] == value
        rows_to_select_synth &= column_mask
    synth_df_reconverted_selected = synth_df_reconverted.loc[rows_to_select_synth]
    synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
    synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
    path = f'generated/{args.dataset}/{str(constraints)}/'
    if not os.path.exists(path):
        os.makedirs(path)
    real_df_reconverted.to_csv(path + 'real.csv')
    synth_df_reconverted_selected.to_csv(path + 'synth_divide.csv')
