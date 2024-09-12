import argparse
import torch
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os
from metasynth import metaSynthTimeWeaver
from data_utils import Preprocessor


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
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    parser.add_argument('-propCycEnc', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned

    train_df_with_hierarchy = preprocessor.df_orig.copy()
    decimal_accuracy_orig = preprocessor.df_orig.apply(decimal_places).to_dict()
    decimal_accuracy_processed = train_df_with_hierarchy.apply(decimal_places).to_dict()
    decimal_accuracy = {}
    for key in decimal_accuracy_processed.keys():
        decimal_accuracy[key] = decimal_accuracy_orig[key]
    test_df_with_hierarchy = train_df_with_hierarchy.copy()
    constraints = {'year': 2013}  # determines which rows need synthetic data
    metadata = metaSynthTimeWeaver(constraints, preprocessor.hierarchical_features_uncyclic, train_df_with_hierarchy)
    rows_in_real = pd.Series([True] * len(train_df_with_hierarchy))
    for key in constraints.keys():
        rows_in_real &= train_df_with_hierarchy[key] == constraints[key]

    real_df = train_df_with_hierarchy.loc[rows_in_real]
    df_synth = metadata.copy()
    df_synth = preprocessor.cyclicEncode(df_synth)
    rows_to_synth = pd.Series([True] * len(metadata))
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = rows_to_synth.values
    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, args.window_size).transpose(1, 2)
    last_index_start = len(d_vals) - len(d_vals) % args.window_size
    window_final = d_vals_tensor[last_index_start:].unsqueeze(0)
    masks = m_vals_tensor.unfold(0, args.window_size, args.window_size)
    masks_final = m_vals_tensor[last_index_start:]
    condition = torch.any(masks, dim=1)
    windows = windows[condition]
    masks = masks[condition]
    hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    all_indices = np.arange(len(df_synth.columns))
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    non_hier_cols = np.array(remaining_indices)
    in_dim = len(df_synth.columns) - len(hierarchical_column_indices)
    args.in_dim = in_dim
    args.embed_dim = in_dim-1
    args.cond_dim = len(hierarchical_column_indices)
    out_dim = len(df_synth.columns) - len(hierarchical_column_indices)
    test_dataset = MyDataset(windows.float())
    mask_dataset = MyDataset(masks)
    test_dataset_final = MyDataset(window_final.float())
    mask_dataset_final = MyDataset(masks_final)
    test_final_dataloader = DataLoader(test_dataset_final, batch_size=args.batch_size)
    mask_final_dataloader = DataLoader(mask_dataset_final, batch_size=args.batch_size)
    model = fetchModel(in_dim, out_dim, args).to(device)
    if args.propCycEnc:
        saved_params = torch.load(f'saved_models/{args.dataset}/model_timegan_prop.pth', map_location=device)
    else:
        saved_params = torch.load(f'saved_models/{args.dataset}/model_timegan.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = False
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch_size)

    with torch.no_grad():
        synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
        for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
            x = torch.normal(0, 1, (test_batch.shape[0], test_batch.shape[1], args.embed_dim)).to(device)
            print(f'batch: {idx} of {len(test_dataloader)}')
            out_g = model.netg(x, test_batch[:, :, hierarchical_column_indices])
            out_r = model.netr(out_g)
            generated = torch.zeros_like(test_batch).to(device)
            generated[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
            generated[:, :, non_hier_cols] = out_r
            synth_tensor = torch.cat((synth_tensor, generated.view(-1, generated.shape[2])), dim=0)

    df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
    real_df_reconverted = preprocessor.rescale(real_df).reset_index(drop=True)
    real_df_reconverted = real_df_reconverted.round(decimal_accuracy)
    # decimal_accuracy = real_df_reconverted.apply(decimal_places).to_dict()
    synth_df_reconverted = preprocessor.decode(df_synthesized, rescale=True)

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
    real_df_reconverted.to_csv(f'{path}real.csv')
    synth_df_reconverted_selected = synth_df_reconverted_selected[real_df_reconverted.columns]
    if args.propCycEnc:
        synth_df_reconverted_selected.to_csv(f'{path}synth_timegan_stride_{args.stride}_prop.csv')
    else:
        synth_df_reconverted_selected.to_csv(f'{path}synth_timegan_stride_{args.stride}.csv')
