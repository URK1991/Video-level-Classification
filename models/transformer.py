import torch
import torch.nn as nn
from .mlp import MLP
from positional_encodings.torch_encodings import Summer, PositionalEncodingPermute1D

class Transformer(nn.Module):
    def __init__(self, num_features=1, max_sequence_len=188, num_layers=2, num_heads=8, device='cuda'):
        super(Transformer, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_heads, batch_first=True).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers).to(device)
        self.mlp = MLP(max_sequence_len * num_features, 4)

    def forward(self, x):
        pos_enc = Summer(PositionalEncodingPermute1D(x.shape[1])).to(self.device)
        inp = pos_enc(x)
        output = self.transformer_encoder(inp)
        return self.mlp(output)
