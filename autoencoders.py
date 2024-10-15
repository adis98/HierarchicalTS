import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TransformerAutoEncoderOneHot(nn.Module):
    def __init__(self, input_dim, d_model, num_encoder_layers=2, num_decoder_layers=2, n_heads=2):
        super(TransformerAutoEncoderOneHot, self).__init__()

        # Positional encoding for sequence data
        # Transformer encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Input and output projections
        self.input_projection = nn.Linear(input_dim, d_model)  # For numerical features
        self.output_projection = nn.Linear(d_model, input_dim)  # Output projection for numerical features
        # self.output_projection_categorical = nn.Linear(d_model,
        #                                                num_categories)  # Output projection for categorical features

    def forward(self, x):
        # x_numerical: (batch_size, seq_length, input_dim)
        # x_one_hot: (batch_size, seq_length, num_categories)

        # Concatenate numerical and one-hot encoded categorical features

        # Project combined input to model dimension
        x = self.input_projection(x)

        # Add positional encoding

        # Encoder step
        memory = self.encoder(x)

        # Decoder step (using encoder output as memory)
        output = self.decoder(memory, memory)

        # Project back to input space for numerical features
        output = self.output_projection(output)

        # Project back to categorical space (one-hot encoding)
        # categorical_output = self.output_projection_categorical(output)

        return output

    def encode(self, x):
        with torch.no_grad():
            x = self.input_projection(x)
            return self.encoder(x)

    def decode(self, x):
        with torch.no_grad():
            x = self.decoder(x, x)
            return self.output_projection(x)
