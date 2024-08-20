# import os
# import argparse
# import json
# import numpy as np
# import torch
# import torch.nn as nn
#
# from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
# from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
#
# from imputers.DiffWaveImputer import DiffWaveImputer
# from imputers.SSSDSAImputer import SSSDSAImputer
# from imputers.SSSDS4Imputer import SSSDS4Imputer
#
#
# def train(output_directory,
#           ckpt_iter,
#           n_iters,
#           iters_per_ckpt,
#           iters_per_logging,
#           learning_rate,
#           use_model,
#           only_generate_missing,
#           masking,
#           missing_k):
#     """
#     Train Diffusion Models
#
#     Parameters:
#     output_directory (str):         save model checkpoints to this path
#     ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
#                                     automatically selects the maximum iteration if 'max' is selected
#     data_path (str):                path to dataset, numpy array.
#     n_iters (int):                  number of iterations to train
#     iters_per_ckpt (int):           number of iterations to save checkpoint,
#                                     default is 10k, for models with residual_channel=64 this number can be larger
#     iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
#     learning_rate (float):          learning rate
#
#     use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
#     only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
#     masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
#     missing_k (int):                k missing time steps for each feature across the sample length.
#     """
#
#     # generate experiment (local) path
#     local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
#                                               diffusion_config["beta_0"],
#                                               diffusion_config["beta_T"])
#
#     # Get shared output_directory ready
#     output_directory = os.path.join(output_directory, local_path)
#     if not os.path.isdir(output_directory):
#         os.makedirs(output_directory)
#         os.chmod(output_directory, 0o775)
#     print("output directory", output_directory, flush=True)
#
#     # map diffusion hyperparameters to gpu
#     for key in diffusion_hyperparams:
#         if key != "T":
#             diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
#
#     # predefine model
#     if use_model == 0:
#         net = DiffWaveImputer(**model_config).cuda()
#     elif use_model == 1:
#         net = SSSDSAImputer(**model_config).cuda()
#     elif use_model == 2:
#         net = SSSDS4Imputer(**model_config).cuda()
#     else:
#         print('Model chosen not available.')
#     print_size(net)
#
#     # define optimizer
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#
#     # load checkpoint
#     if ckpt_iter == 'max':
#         ckpt_iter = find_max_epoch(output_directory)
#     if ckpt_iter >= 0:
#         try:
#             # load checkpoint file
#             model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
#             checkpoint = torch.load(model_path, map_location='cpu')
#
#             # feed model dict and optimizer state
#             net.load_state_dict(checkpoint['model_state_dict'])
#             if 'optimizer_state_dict' in checkpoint:
#                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#             print('Successfully loaded model at iteration {}'.format(ckpt_iter))
#         except:
#             ckpt_iter = -1
#             print('No valid checkpoint model found, start training from initialization try.')
#     else:
#         ckpt_iter = -1
#         print('No valid checkpoint model found, start training from initialization.')
#
#     ### Custom data loading and reshaping ###
#
#     training_data = np.load(trainset_config['train_data_path'])
#     training_data = np.split(training_data, 160, 0)
#     training_data = np.array(training_data)
#     training_data = torch.from_numpy(training_data).float().cuda()
#     print('Data loaded')
#
#     # training
#     n_iter = ckpt_iter + 1
#     while n_iter < n_iters + 1:
#         for batch in training_data:
#
#             if masking == 'rm':
#                 transposed_mask = get_mask_rm(batch[0], missing_k)
#             elif masking == 'mnr':
#                 transposed_mask = get_mask_mnr(batch[0], missing_k)
#             elif masking == 'bm':
#                 transposed_mask = get_mask_bm(batch[0], missing_k)
#
#             mask = transposed_mask.permute(1, 0)
#             mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
#             loss_mask = ~mask.bool()
#             batch = batch.permute(0, 2, 1)
#
#             assert batch.size() == mask.size() == loss_mask.size()
#
#             # back-propagation
#             optimizer.zero_grad()
#             X = batch, batch, mask, loss_mask
#             loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
#                                  only_generate_missing=only_generate_missing)
#
#             loss.backward()
#             optimizer.step()
#
#             if n_iter % iters_per_logging == 0:
#                 print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
#
#             # save checkpoint
#             if n_iter > 0 and n_iter % iters_per_ckpt == 0:
#                 checkpoint_name = '{}.pkl'.format(n_iter)
#                 torch.save({'model_state_dict': net.state_dict(),
#                             'optimizer_state_dict': optimizer.state_dict()},
#                            os.path.join(output_directory, checkpoint_name))
#                 print('model at iteration %s is saved' % n_iter)
#
#             n_iter += 1
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
#                         help='JSON file for configuration')
#
#     args = parser.parse_args()
#
#     with open(args.config) as f:
#         data = f.read()
#
#     config = json.loads(data)
#     print(config)
#
#     train_config = config["train_config"]  # training parameters
#
#     global trainset_config
#     trainset_config = config["trainset_config"]  # to load trainset
#
#     global diffusion_config
#     diffusion_config = config["diffusion_config"]  # basic hyperparameters
#
#     global diffusion_hyperparams
#     diffusion_hyperparams = calc_diffusion_hyperparams(
#         **diffusion_config)  # dictionary of all diffusion hyperparameters
#
#     global model_config
#     if train_config['use_model'] == 0:
#         model_config = config['wavenet_config']
#     elif train_config['use_model'] == 1:
#         model_config = config['sashimi_config']
#     elif train_config['use_model'] == 2:
#         model_config = config['wavenet_config']
#
#     train(**train_config)
import argparse

import torch

from main import Preprocessor
from training import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np

from torch import from_numpy, optim, nn, randint, normal, sqrt, device
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='S4')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=128, help='hidden embedding dimension')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    parser.add_argument('-num_res_layers', type=int, default=4, help='the number of residual layers')
    parser.add_argument('-res_channels', type=int, default=128, help='the number of res channels')
    parser.add_argument('-skip_channels', type=int, default=128, help='the number of skip channels')
    parser.add_argument('-diff_step_embed_in', type=int, default=64, help='input embedding size diffusion')
    parser.add_argument('-diff_step_embed_mid', type=int, default=128, help='middle embedding size diffusion')
    parser.add_argument('-diff_step_embed_out', type=int, default=128, help='output embedding size diffusion')
    parser.add_argument('-s4_lmax', type=int, default=100)
    parser.add_argument('-s4_dstate', type=int, default=64)
    parser.add_argument('-s4_dropout', type=float, default=0.0)
    parser.add_argument('-s4_bidirectional', type=bool, default=True)
    parser.add_argument('-s4_layernorm', type=bool, default=True)
    args = parser.parse_args()
    dataset = args.dataset
    preprocessor = Preprocessor(dataset)
    df = preprocessor.df_cleaned
    hierarchical_column_indices = df.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    training_samples = []
    for i in range(0, len(df) - args.window_size + 1, args.stride):
        window = df.iloc[i:i + args.window_size].values
        training_samples.append(window)

    in_dim = len(df.columns)
    out_dim = len(df.columns) - len(hierarchical_column_indices)
    training_dataset = MyDataset(from_numpy(np.array(training_samples)).float())
    model = fetchModel(in_dim, out_dim, args)
    diffusion_config = fetchDiffusionConfig(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size)
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    all_indices = np.arange(len(df.columns))

    # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)

    # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    """TRAINING"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],))
            sigmas = normal(0, 1, size=batch.shape)
            """Forward noising"""
            alpha_bars = diffusion_config['alpha_bars']
            coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, non_hier_cols] = 0
            conditional_mask = from_numpy(conditional_mask).float()
            batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
            batch_noised = batch_noised.to(device)
            timesteps = timesteps.reshape((-1, 1))
            timesteps = timesteps.to(device)
            sigmas_predicted = model(batch_noised, timesteps)
            optimizer.zero_grad()
            sigmas_permuted = sigmas[:, :, non_hier_cols].permute((0, 2, 1))
            sigmas_permuted = sigmas_permuted.to(device)
            loss = criterion(sigmas_predicted, sigmas_permuted)
            loss.backward()
            total_loss += loss
            optimizer.step()
        print(f'epoch: {epoch}, loss: {total_loss}')