import argparse
import torch
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os
from metasynth import metaSynthTimeWeaver, metadataMask
from data_utils import Preprocessor
import torch.nn.functional as F


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


def score_func(y, t, observation, observation_mask, features, fast_noise_estimate, diffusion_config):
    with torch.enable_grad():
        y.requires_grad_(True)
        Ey = energy_func(
            y, t, observation, observation_mask, features, fast_noise_estimate, diffusion_config
        )
        return -torch.autograd.grad(Ey, y)[0]


def energy_func(y, t, observation, observation_mask, features, fast_noise_estimate, diffusion_config):
    alpha_bar_t = diffusion_config['alpha_bars'][t].to(device)
    initial_guess = (y[:, :, features] - sqrt(1 - alpha_bar_t) * fast_noise_estimate) / sqrt(alpha_bar_t)
    """mean-squared self-guidance"""
    return F.mse_loss(
        initial_guess,
        observation[:, :, features],
        reduction="none",
    )[observation_mask == 1].sum()


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
    # parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
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
    parser.add_argument('-synth_mask', type=str, required=True,
                        help="the hierarchy masking type, coarse (C), fine (F), mid (M)")
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned
    extra_indices = args.window_size - len(preprocessor.test_indices) % args.window_size
    test_df = df.loc[preprocessor.train_indices[-extra_indices:] + preprocessor.test_indices]
    test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
    decimal_accuracy_orig = preprocessor.df_orig.apply(decimal_places).to_dict()
    decimal_accuracy_processed = test_df_with_hierarchy.apply(decimal_places).to_dict()
    decimal_accuracy = {}
    for key in decimal_accuracy_processed.keys():
        decimal_accuracy[key] = decimal_accuracy_orig[key]

    metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
    rows_to_synth = metadataMask(metadata, args.synth_mask, args.dataset)
    real_df = test_df_with_hierarchy[rows_to_synth]
    df_synth = test_df.copy()
    """Approach 1: Divide and conquer"""
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = rows_to_synth.values
    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, args.window_size).transpose(1, 2)
    # last_index_start = len(d_vals) - len(d_vals) % args.window_size
    # window_final = d_vals_tensor[last_index_start:].unsqueeze(0)
    masks = m_vals_tensor.unfold(0, args.window_size, args.window_size)
    # masks_final = m_vals_tensor[last_index_start:]
    condition = torch.any(masks, dim=1)
    windows = windows[condition]
    masks = masks[condition]
    hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    in_dim = len(df_synth.columns)
    out_dim = len(df_synth.columns) - len(hierarchical_column_indices)
    test_dataset = MyDataset(windows.float())
    mask_dataset = MyDataset(masks)
    # test_dataset_final = MyDataset(window_final.float())
    # mask_dataset_final = MyDataset(masks_final)
    # test_final_dataloader = DataLoader(test_dataset_final, batch_size=args.batch_size)
    # mask_final_dataloader = DataLoader(mask_dataset_final, batch_size=args.batch_size)
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
            x = torch.normal(0, 1, test_batch.shape).to(device)
            x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
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

                mask_expanded = torch.zeros_like(test_batch, dtype=bool)
                for channel in non_hier_cols:
                    mask_expanded[:, :, channel] = mask_batch

                epsilon_pred = model(x, times)
                epsilon_pred = epsilon_pred.permute((0, 2, 1))
                if step > 0:
                    vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
                                                                                             size=epsilon_pred.shape).to(
                        device)
                else:
                    vari = 0.0

                normal_denoising = torch.normal(0, 1, test_batch.shape).to(device)
                normal_denoising[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - (
                        (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
                normal_denoising[:, :, non_hier_cols] += vari
                masked_binary = mask_batch.int()
                # x[mask_batch][:, non_hier_cols] = normal_denoising[mask_batch]
                x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]

                fast_noise_estimate = model(x, times - 1).permute((0, 2, 1))
                add_term = score_func(x, step - 1, test_batch, mask_expanded[:, :, non_hier_cols], non_hier_cols, fast_noise_estimate,
                                      diffusion_config)  # already denoised by one step

                x[mask_expanded] = normal_denoising[mask_expanded] + add_term[mask_expanded]

                x[~mask_expanded] = test_batch[~mask_expanded]

            generated = x.view(-1, x.shape[2])
            synth_tensor = torch.cat((synth_tensor, generated), dim=0)

        # for idx, (test_batch, mask_batch) in enumerate(zip(test_final_dataloader, mask_final_dataloader)):
        #     x = torch.normal(0, 1, test_batch.shape).to(device)
        #     print(f'batch: {idx} of {len(test_dataloader)}')
        #     for step in range(diffusion_config['T'] - 1, -1, -1):
        #         test_batch = test_batch.to(device)
        #         mask_batch = mask_batch.to(device)
        #         print(f"backward step: {step}")
        #         times = torch.full(size=(test_batch.shape[0], 1), fill_value=step).to(device)
        #         alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
        #         alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
        #         alpha_t = diffusion_config['alphas'][step].to(device)
        #         beta_t = diffusion_config['betas'][step].to(device)
        #
        #         # sampled_noise = torch.normal(0, 1, test_batch.shape).to(device)
        #         # cached_denoising = torch.sqrt(alpha_bar_t) * test_batch + torch.sqrt(1 - alpha_bar_t) * sampled_noise
        #         mask_expanded = torch.zeros_like(test_batch, dtype=bool)
        #         for channel in non_hier_cols:
        #             mask_expanded[:, :, channel] = mask_batch
        #
        #         epsilon_pred = model(x, times)
        #         epsilon_pred = epsilon_pred.permute((0, 2, 1))
        #         if step > 0:
        #             vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
        #                                                                                      size=epsilon_pred.shape).to(
        #                 device)
        #         else:
        #             vari = 0.0
        #
        #         normal_denoising = torch.normal(0, 1, test_batch.shape).to(device)
        #         normal_denoising[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - (
        #                 (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
        #         normal_denoising[:, :, non_hier_cols] += vari
        #         masked_binary = mask_batch.int()
        #         # x[mask_batch][:, non_hier_cols] = normal_denoising[mask_batch]
        #         x[mask_expanded] = normal_denoising[mask_expanded]
        #         x[~mask_expanded] = test_batch[~mask_expanded]
        #
        #     generated = x.view(-1, x.shape[2])
        #     synth_tensor = torch.cat((synth_tensor, generated), dim=0)

    df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
    real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
    real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
    # decimal_accuracy = real_df_reconverted.apply(decimal_places).to_dict()
    synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)

    rows_to_select_synth = rows_to_synth.reset_index(drop=True)
    # for col, value in constraints.items():
    #     column_mask = synth_df_reconverted[col] == value
    #     rows_to_select_synth &= column_mask
    synth_df_reconverted_selected = synth_df_reconverted.loc[rows_to_select_synth]
    synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
    synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
    path = f'generated/{args.dataset}/{args.synth_mask}/'
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(f'{path}real.csv'):
        real_df_reconverted.to_csv(f'{path}real.csv')
    synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
    if args.propCycEnc:
        synth_df_reconverted_selected.to_csv(f'{path}synth_tsdiff_prop.csv')
    else:
        synth_df_reconverted_selected.to_csv(f'{path}synth_tsdiff.csv')