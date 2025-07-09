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
from matplotlib import pyplot as plt
from copy import deepcopy
import torch.nn.functional as F


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


def create_pipelined_noise(test_batch, args):
    sampled = torch.normal(0, 1, (args.stride * (test_batch.shape[0] - 1) + args.window_size, test_batch.shape[2]))
    sampled_noise = sampled.unfold(0, args.window_size, args.stride).transpose(1, 2)
    return sampled_noise


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
    end = preprocessor.test_indices[-1]
    start = preprocessor.test_indices[0]
    window_cnt = ((end + 1 - args.window_size - start) // args.stride) + 1
    tilde_start = end + 1 - args.window_size - (window_cnt * args.stride)
    additional_indices = start - tilde_start
    test_df = df.loc[preprocessor.train_indices[-additional_indices:] + preprocessor.test_indices]
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
    """Approach 1: Pipeline"""
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = rows_to_synth.values

    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, args.stride).transpose(1, 2)
    masks = m_vals_tensor.unfold(0, args.window_size, args.stride)
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
            param.requires_grad = True
    model.eval()
    num_ops = 0  # start measuring the number of compute steps for the whole generation time
    exec_times = []
    # mses = []
    for trial in range(args.n_trials):
        start = timer()
        with torch.no_grad():
            synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
            for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
                test_batch = test_batch.to(device)
                mask_batch = mask_batch.to(device)
                x = create_pipelined_noise(test_batch, args).to(device)
                x.requires_grad_()
                x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
                print(f'batch: {idx} of {len(test_dataloader)}')
                mask_expanded = torch.zeros_like(test_batch, dtype=bool)
                for channel in non_hier_cols:
                    mask_expanded[:, :, channel] = mask_batch

                for step in range(diffusion_config['T'] - 1, -1, -1):
                    print(f"backward step: {step}")
                    times = torch.full(size=(test_batch.shape[0], 1), fill_value=step).to(device)
                    alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                    alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                    alpha_t = diffusion_config['alphas'][step].to(device)
                    beta_t = diffusion_config['betas'][step].to(device)
                    sampled_noise = create_pipelined_noise(test_batch, args).to(device)
                    conditional_fwd = sqrt(alpha_bar_t) * test_batch + sqrt(1 - alpha_bar_t) * sampled_noise
                    if step == diffusion_config['T'] - 1:
                        x[~mask_expanded] = conditional_fwd[~mask_expanded]
                    x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]

                    with torch.enable_grad():
                        epsilon_pred = model(x, times).permute((0, 2, 1))
                        # grad_xt = torch.autograd.grad(epsilon_pred, x, grad_outputs=torch.ones_like(epsilon_pred), retain_graph=True)[0]
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

                        # x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                        # x[~mask_expanded] = test_batch[~mask_expanded]
                        rolled_x = normal_denoising.roll(1, 0)
                        rolled_x[0, args.stride:, :] = normal_denoising[0, :(args.window_size - args.stride), :]
                        # loss2 = 0.0
                        """MSE LOSS"""
                        loss1 = torch.sum((normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] - rolled_x[:, args.stride:args.window_size, non_hier_cols])**2, dim=(1, 2))
                        """MAE LOSS"""
                        # loss1 = torch.sum(torch.abs(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] - rolled_x[:, args.stride:args.window_size, non_hier_cols]), dim=(1, 2))
                        """COSINE SIMILARITY"""
                        # dot = torch.sum(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] * rolled_x[:, args.stride:args.window_size, non_hier_cols], dim=1)
                        # unorm = torch.norm(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols], p=2, dim=1)
                        # vnorm = torch.norm(rolled_x[:, args.stride:args.window_size, non_hier_cols], p=2, dim=1)
                        # cosinesim = dot/(unorm*vnorm + 1e-8)
                        # loss1 = 1 - cosinesim.mean(dim=1)

                        """TEMPORAL CORRELATION"""
                        # umean = torch.mean(normal_denoising[:, :(args.window_size - args.stride), non_hier_cols], dim=1, keepdim=True)
                        # vmean = torch.mean(rolled_x[:, args.stride:args.window_size, non_hier_cols], dim=1, keepdim=True)
                        # ucentred = normal_denoising[:, :(args.window_size - args.stride), non_hier_cols] - umean
                        # vcentred = rolled_x[:, args.stride:args.window_size, non_hier_cols] - vmean
                        # num = torch.sum(ucentred*vcentred, dim=1)
                        # den = torch.sqrt(torch.sum(ucentred**2, dim=1) * torch.sum(vcentred**2, dim=1) + 1e-8)
                        # score = num/den
                        # loss1 = 1 - score.mean(dim=1)

                        loss2 = torch.sum(~mask_expanded[:, :, non_hier_cols] * ((x[:, :, non_hier_cols]-torch.sqrt(1-alpha_bar_t)*epsilon_pred)/(torch.sqrt(alpha_bar_t)) - test_batch[:, :, non_hier_cols])**2, dim=(1, 2))
                        # loss1 = 0.0
                        loss = loss1 + loss2
                        # print(torch.sum(loss.cpu()))
                        grad = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

                    x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                    eps = -0.1 * grad[:, :, non_hier_cols]
                    x[:, :, non_hier_cols] = x[:, :, non_hier_cols] + eps
                    # x[:, :, non_hier_cols] += eps/sqrt(alpha_t) - (eps*beta_t/(sqrt(alpha_t)*sqrt(1-alpha_bar_t))) * grad_xt[:, :, non_hier_cols]
                    # x[1:, : (args.window_size - args.stride), :] = rolled_x[1:, args.stride: args.window_size, :]
                    # x[~mask_expanded] = test_batch[~mask_expanded]
                    # if step == 0:
                    #     pass
                        # x[~mask_expanded] = test_batch[~mask_expanded]
                        # x[1:, : (args.window_size - args.stride), :] = rolled_x[1:, args.stride: args.window_size, :]
                    if trial == 0:
                        num_ops += 1

                x[~mask_expanded] = test_batch[~mask_expanded]
                first_sample = x[0]
                # mse = torch.mean((x[mask_expanded] - test_batch[mask_expanded])**2).cpu()
                # print(f'MSE: {mse}')
                # plt.plot(x[0, :, non_hier_cols].cpu())
                # plt.plot(test_batch[0, :, non_hier_cols].cpu())
                # plt.title(f'{mse}')
                # plt.show()
                # exit()
                last_timesteps = x[1:, (args.window_size - args.stride):, :]
                if idx == 0:
                    last_timesteps = last_timesteps.reshape(-1, last_timesteps.shape[2])
                    generated = torch.cat((first_sample, last_timesteps), dim=0)
                else:
                    generated = x[:, (args.window_size - args.stride):, :]
                    generated = generated.reshape(-1, generated.shape[2])
                synth_tensor = torch.cat((synth_tensor, generated), dim=0)

        end = timer()
        diff = end - start
        exec_times.append(diff)
        df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
        real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
        real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
        synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)
        rows_to_synth_reset = rows_to_synth.reset_index(drop=True)
        synth_df_reconverted_selected = synth_df_reconverted[rows_to_synth_reset]
        synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
        synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
    #     real_df_cleaned = preprocessor.cleanDataset(args.dataset, real_df_reconverted)
    #     synth_df_cleaned = preprocessor.cleanDataset(args.dataset, synth_df_reconverted_selected)
    #     nhc = [c for c in real_df_cleaned.columns if
    #                      c not in preprocessor.hierarchical_features_cyclic]
    #     MSE = ((synth_df_cleaned[nhc] - real_df_cleaned[nhc]) ** 2).mean().mean()
    #     mses.append(MSE)
    #     print(trial)
    # print(np.mean(np.array(mses)))
        path = f'generated/{args.dataset}/{args.synth_mask}/'
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(f'{path}real.csv'):
            real_df_reconverted.to_csv(f'{path}real.csv')
        synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
        if args.propCycEnc:
            synth_df_reconverted_selected.to_csv(
                f'{path}synth_wavestitch_pipeline_stride_{args.stride}_trial_{trial}_cycProp_grad_correction.csv')
            if trial == 0:
                with open(f'{path}denoiser_calls_pipeline_stride_{args.stride}_cycProp_grad_correction.txt', 'w') as file:
                    file.write(str(num_ops))
        else:
            synth_df_reconverted_selected.to_csv(
                f'{path}synth_wavestitch_pipeline_stride_{args.stride}_trial_{trial}_cycStd_grad_correction.csv')
            if trial == 0:
                with open(f'{path}denoiser_calls_pipeline_stride_{args.stride}_cycStd_grad_correction.txt', 'w') as file:
                    file.write(str(num_ops))

    with open(f'generated/{args.dataset}/{args.synth_mask}/denoiser_calls_pipeline_stride_{args.stride}_cycStd_grad_correction.txt',
              'a') as file:
        arr_time = np.array(exec_times)
        file.write('\n' + str(np.mean(arr_time)) + '\n')
        file.write(str(np.std(arr_time)))
