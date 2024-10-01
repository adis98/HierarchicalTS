import argparse
import torch
from data_utils import Preprocessor
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os
from metasynth import metadataMask


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


def create_pipelined_noise(test_batch, args):
    sampled = torch.normal(0, 1, (test_batch.shape[0] + test_batch.shape[1] - 1, test_batch.shape[2]))
    sampled_noise = sampled.unfold(0, args.window_size, 1).transpose(1, 2)
    return sampled_noise


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
    parser.add_argument('-propCycEnc', type=bool, default=False)
    parser.add_argument('-synth_mask', type=str, required=True, help="the hierarchy masking type, coarse (C), fine (F), mid (M)")
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned
    training_df = df.loc[preprocessor.train_indices]
    test_df = df.loc[preprocessor.test_indices]
    test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
    decimal_accuracy_orig = preprocessor.df_orig.apply(decimal_places).to_dict()
    decimal_accuracy_processed = test_df_with_hierarchy.apply(decimal_places).to_dict()
    decimal_accuracy = {}
    for key in decimal_accuracy_processed.keys():
        decimal_accuracy[key] = decimal_accuracy_orig[key]

    metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
    rows_to_synth = metadataMask(metadata, args.synth_mask, args.dataset)
    # constraints = {'year': 2013}  # determines which rows need synthetic data
    # metadata = metaSynthHyacinth(preprocessor.hierarchical_features_uncyclic, train_df_with_hierarchy)

    # Iterate over the dictionary to create masks for each column
    # for col, value in constraints.items():
    #     column_mask = metadata[col] == value
    #     rows_to_synth &= column_mask

    # rows_to_synth_orig = rows_to_synth
    # real_exclusive = rows_to_synth & (metadata['_merge'] == 'both')  # rows in the real data that need re-synthesis
    # rows_to_synth |= metadata['_merge'] == 'left_only'
    # real_df = metadata.loc[real_exclusive]
    # real_df = real_df.drop(columns=['_merge'])
    real_df = test_df_with_hierarchy[rows_to_synth]
    # df_synth = metadata.copy()
    # df_synth = preprocessor.cyclicEncode(df_synth)
    # df_synth = df_synth.drop(columns=['_merge'])
    df_synth = test_df.copy()
    """Approach 1: Divide and conquer"""
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = rows_to_synth.values

    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, 1).transpose(1, 2)
    masks = m_vals_tensor.unfold(0, args.window_size, 1)
    condition = torch.any(masks, dim=1)
    windows = windows[condition]
    masks = masks[condition]
    hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    in_dim = len(df_synth.columns)
    out_dim = len(df_synth.columns) - len(hierarchical_column_indices)
    test_dataset = MyDataset(windows.float())
    mask_dataset = MyDataset(masks)
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch_size)
    all_indices = np.arange(len(df_synth.columns))
    #
    # # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    #
    # # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    if args.propCycEnc:
        saved_params = torch.load(f'saved_models/{args.dataset}/model_prop.pth', map_location=device)
    else:
        saved_params = torch.load(f'saved_models/{args.dataset}/model.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = False
    model.eval()

    with torch.no_grad():
        synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
        for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
            x = create_pipelined_noise(test_batch, args).to(device)
            print(f'batch: {idx} of {len(test_dataloader)}')
            for step in range(diffusion_config['T'] - 1, -1, -1):
                test_batch = test_batch.to(device)
                mask_batch = mask_batch.to(device)
                print(f"backward step: {step}")
                times = torch.full(size=(test_batch.shape[0], 1), fill_value=step).to(device)
                alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                alpha_t = diffusion_config['alphas'][step].to(device)
                beta_t = diffusion_config['betas'][step].to(device)

                sampled_noise = create_pipelined_noise(test_batch, args).to(device)
                cached_denoising = torch.sqrt(alpha_bar_t) * test_batch + torch.sqrt(1 - alpha_bar_t) * sampled_noise
                mask_expanded = torch.zeros_like(test_batch, dtype=bool)
                for channel in non_hier_cols:
                    mask_expanded[:, :, channel] = mask_batch
                x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
                x[~mask_expanded] = cached_denoising[~mask_expanded]
                epsilon_pred = model(x, times)
                epsilon_pred = epsilon_pred.permute((0, 2, 1))
                if step > 0:
                    vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
                                                                                             size=epsilon_pred.shape).to(
                        device)
                else:
                    vari = 0.0

                normal_denoising = create_pipelined_noise(test_batch, args).to(device)
                normal_denoising[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - (
                        (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
                normal_denoising[:, :, non_hier_cols] += vari
                masked_binary = mask_batch.int()
                # x[mask_batch][:, non_hier_cols] = normal_denoising[mask_batch]
                x[mask_expanded] = normal_denoising[mask_expanded]
                x[~mask_expanded] = test_batch[~mask_expanded]
                rolled_x = x.roll(1, 0)
                x[1:, : args.window_size - 1, :] = rolled_x[1:, 1: args.window_size, :]
                # if step == 0:
                #     x[~mask_batch][:, non_hier_cols] = test_batch[~mask_batch][:, non_hier_cols]
                #     k = x[~mask_batch][:, non_hier_cols]

            first_sample = x[0]
            last_timesteps = x[1:, -1, :]
            if idx == 0:
                generated = torch.cat((first_sample, last_timesteps), dim=0)
            else:
                generated = x[:, -1, :]
            synth_tensor = torch.cat((synth_tensor, generated), dim=0)

    df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
    real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
    real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
    # decimal_accuracy = real_df_reconverted.apply(decimal_places).to_dict()
    synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)

    # rows_to_select_synth = pd.Series([True] * len(synth_df_reconverted))
    # for col, value in constraints.items():
    #     column_mask = synth_df_reconverted[col] == value
    #     rows_to_select_synth &= column_mask
    synth_df_reconverted_selected = synth_df_reconverted.loc[rows_to_synth]
    synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
    synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
    path = f'generated/{args.dataset}/{args.synth_mask}/'
    if not os.path.exists(path):
        os.makedirs(path)
    real_df_reconverted.to_csv(f'{path}real.csv')
    synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
    if args.propCycEnc:
        synth_df_reconverted_selected.to_csv(f'{path}synth_dnq_stride_{args.stride}_prop.csv')
    else:
        synth_df_reconverted_selected.to_csv(f'{path}synth_dnq_stride_{args.stride}.csv')
