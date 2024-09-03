import torch.nn as nn
from torch import ones, triu


def _generate_square_subsequent_mask(sz):
    mask = (triu(ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size=7, out_size=1, num_layers=3, dropout=0):
        super(Transformer, self).__init__()
        self.init_embedding = nn.Linear(in_features=feature_size, out_features=2 * feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2 * feature_size, nhead=2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(2 * feature_size, out_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, device):
        src_embed = self.init_embedding(src)
        mask = _generate_square_subsequent_mask(src_embed.shape[1]).to(device)
        output = self.transformer_encoder(src_embed)
        output = self.decoder(output)
        return output
