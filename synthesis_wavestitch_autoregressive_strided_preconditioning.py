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
from timeit import default_timer as timer


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


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
    parser.add_argument('-synth_mask', type=str, required=True,
                        help="the hierarchy masking type, coarse (C), fine (F), mid (M)")
    parser.add_argument('-n_trials', type=int, default=5)
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned

    #  Add some more samples form the training set as additional context for synthesis
    test_df = df.loc[preprocessor.train_indices[-args.window_size:] + preprocessor.test_indices]
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
    """Approach 2: Autoregressive"""
    hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    in_dim = len(df_synth.columns)
    out_dim = len(df_synth.columns) - len(hierarchical_column_indices)
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)

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

    total_to_synth = rows_to_synth.sum()
    num_ops = 0  # start measuring the number of compute steps for the whole generation time
    exec_times = []
    for trial in range(args.n_trials):
        start = timer()
        synthetic_df = test_df.copy()
        synthetic_mask = rows_to_synth.copy()
        test_mask = rows_to_synth.copy()
        with torch.no_grad():
            total_generated = 0
            for i in range(0, len(test_df) - args.window_size + 1, args.stride):
                window = synthetic_df.iloc[i:i + args.window_size].values
                mask_window = synthetic_mask.iloc[i:i + args.window_size].values
                test_mask_window = test_mask.iloc[:i:i + args.window_size].values
                if any(mask_window):
                    test_batch = from_numpy(np.reshape(window, (1, window.shape[0], window.shape[1]))).float().to(
                        device)
                    mask_batch = from_numpy(mask_window).to(device)
                    x = torch.normal(0, 1, test_batch.shape).to(device)
                    x.requires_grad_()
                    x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
                    for step in range(diffusion_config['T'] - 1, -1, -1):
                        times = torch.full(size=(test_batch.shape[0], 1), fill_value=step).to(device)
                        alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                        alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                        alpha_t = diffusion_config['alphas'][step].to(device)
                        beta_t = diffusion_config['betas'][step].to(device)
                        sampled_noise = torch.normal(0, 1, test_batch.shape).to(device)
                        cached_denoising = torch.sqrt(alpha_bar_t) * test_batch + torch.sqrt(
                            1 - alpha_bar_t) * sampled_noise
                        mask_expanded = torch.zeros_like(test_batch, dtype=torch.bool, device=device)
                        for channel in non_hier_cols:
                            mask_expanded[:, :, channel] = mask_batch
                        if step == diffusion_config['T']-1:
                            x[~mask_expanded] = cached_denoising[~mask_expanded]
                        x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]

                        with torch.enable_grad():
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

                            loss = torch.sum(~mask_expanded[:, :, non_hier_cols] * (
                                        x[:, :, non_hier_cols] - test_batch[:, :, non_hier_cols]) ** 2, dim=(1, 2))

                            grad = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]
                        x[mask_expanded] = normal_denoising[mask_expanded]
                        eps = -0.1 * grad[:, :, non_hier_cols]
                        x[:, :, non_hier_cols] = x[:, :, non_hier_cols] + eps

                        if trial == 0:
                            num_ops += 1

                    x[~mask_expanded] = test_batch[~mask_expanded]
                    total_generated += np.sum(mask_window)
                    print(f'{total_generated} synthesized out of {total_to_synth}')
                    synthetic_df.iloc[i:i + args.window_size] = x.cpu().numpy()
                    synthetic_mask.iloc[i:i + args.window_size] = np.zeros_like(mask_window, dtype=bool)

        end = timer()
        exec_times.append(end - start)
        df_synthesized = synthetic_df[df.columns].reset_index(drop=True)
        real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
        real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
        synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)
        rows_to_synth_reset = rows_to_synth.reset_index(drop=True)
        synth_df_reconverted_selected = synth_df_reconverted[rows_to_synth_reset]
        synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
        synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
        path = f'generated/{args.dataset}/{args.synth_mask}/'
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(f'{path}real.csv'):
            real_df_reconverted.to_csv(f'{path}real.csv')
        synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
        if args.propCycEnc:
            synth_df_reconverted_selected.to_csv(
                f'{path}synth_wavestitch_autoregressive_stride_{args.stride}_trial_{trial}_cycProp_grad_simplecoeff.csv')
            if trial == 0:
                with open(f'{path}denoiser_calls_autoregressive_stride_{args.stride}_cycProp_grad_simplecoeff.txt', 'w') as file:
                    file.write(str(num_ops))
        else:
            synth_df_reconverted_selected.to_csv(
                f'{path}synth_wavestitch_autoregressive_stride_{args.stride}_trial_{trial}_cycStd_grad_simplecoeff.csv')
            if trial == 0:
                with open(f'{path}denoiser_calls_autoregressive_stride_{args.stride}_cycStd_grad_simplecoeff.txt', 'w') as file:
                    file.write(str(num_ops))

    with open(
            f'generated/{args.dataset}/{args.synth_mask}/denoiser_calls_autoregressive_stride_{args.stride}_cycStd_grad_simplecoeff.txt',
            'a') as file:
        arr_time = np.array(exec_times)
        file.write('\n' + str(np.mean(arr_time)) + '\n')
        file.write(str(np.std(arr_time)))
