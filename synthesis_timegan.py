import argparse
import torch
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os
from metasynth import metadataMask
from data_utils import Preprocessor
from timeit import default_timer as timer


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4, Timegan', default='Timegan')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-embed_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('-num_layer', type=int, default=3, help='number of layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
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
    test_df = df.loc[preprocessor.test_indices]
    test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
    decimal_accuracy_orig = preprocessor.df_orig.apply(decimal_places).to_dict()
    decimal_accuracy_processed = test_df_with_hierarchy.apply(decimal_places).to_dict()
    decimal_accuracy = {}
    for key in decimal_accuracy_processed.keys():
        decimal_accuracy[key] = decimal_accuracy_orig[key]

    metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
    rows_to_synth = metadataMask(metadata, args.synth_mask, args.dataset)
    real_df = test_df_with_hierarchy[rows_to_synth]
    df_synth = test_df[rows_to_synth]
    """Approach 3: Divide and conquer"""
    extra_samples = None
    if (len(df_synth) % args.window_size) > 0:
        extra_samples = args.window_size - (len(df_synth) % args.window_size) if (len(df_synth) % args.window_size) > 0 else 0
        extra_rows = df_synth.iloc[-extra_samples:, :].values
        extra_rows = np.zeros_like(extra_rows)
        new_rows = pd.DataFrame(extra_rows, columns=df_synth.columns)
        # Concatenate the original DataFrame with the new rows
        df_synth = pd.concat([df_synth, new_rows], ignore_index=True)
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = np.array([True] * len(df_synth))
    if extra_samples is not None:
        m_vals[-extra_samples:] = False
    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, args.window_size).transpose(1, 2)
    masks = m_vals_tensor.unfold(0, args.window_size, args.window_size)
    hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    in_dim = len(df.columns) - len(hierarchical_column_indices)
    out_dim = len(df.columns) - len(hierarchical_column_indices)
    total_dim = len(df.columns)
    args.in_dim = in_dim
    args.embed_dim = in_dim - 1 if in_dim > 1 else in_dim
    args.cond_dim = len(hierarchical_column_indices)
    test_dataset = MyDataset(windows.float())
    mask_dataset = MyDataset(masks)
    model = fetchModel(in_dim, out_dim, args).to(device)
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
        saved_params = torch.load(f'saved_models/{args.dataset}/model_timegan_prop.pth', map_location=device)
    else:
        saved_params = torch.load(f'saved_models/{args.dataset}/model_timegan.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = False
    model.eval()
    exec_times = []
    for trial in range(args.n_trials):
        start = timer()
        with torch.no_grad():
            synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
            for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
                test_batch = test_batch.to(device)
                x = torch.normal(0, 1, (test_batch.shape[0], test_batch.shape[1], args.embed_dim)).to(device)
                print(f'batch: {idx} of {len(test_dataloader)}')
                out_g = model.netg(x, test_batch[:, :, hierarchical_column_indices])
                out_r = model.netr(out_g)
                generated = torch.zeros_like(test_batch).to(device)
                generated[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
                generated[:, :, non_hier_cols] = out_r
                synth_tensor = torch.cat((synth_tensor, generated.view(-1, generated.shape[2])), dim=0)

        end = timer()
        diff = end - start
        exec_times.append(diff)
        df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
        real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
        real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
        synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)
        # rows_to_synth_reset = rows_to_synth.reset_index(drop=True)
        synth_df_reconverted_selected = synth_df_reconverted.iloc[:-extra_samples,
                                        :] if extra_samples is not None else synth_df_reconverted.iloc[:, :]
        synth_df_reconverted_selected = synth_df_reconverted_selected.round(decimal_accuracy)
        synth_df_reconverted_selected = synth_df_reconverted_selected.reset_index(drop=True)
        path = f'generated/{args.dataset}/{args.synth_mask}/'
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(f'{path}real.csv'):
            real_df_reconverted.to_csv(f'{path}real.csv')
        synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
        """remember to uncomment the following lines in the committed version"""
        if args.propCycEnc:
            synth_df_reconverted_selected.to_csv(f'{path}synth_timegan_trial_{trial}_cycProp.csv')
        else:
            synth_df_reconverted_selected.to_csv(f'{path}synth_timegan_trial_{trial}_cycStd.csv')

    with open(
            f'generated/{args.dataset}/{args.synth_mask}/denoiser_calls_timegan_cycStd.txt',
            'a') as file:
        arr_time = np.array(exec_times)
        file.write(str(1) + '\n')
        file.write(str(np.mean(arr_time)) + '\n')
        file.write(str(np.std(arr_time)))
