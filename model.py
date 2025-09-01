import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x = x + self.pe[:, :T, :D]
        return x

class VQVAE(nn.Module):
    def __init__(self, 
                 data_dim: int,
                 latent_dim: int, 
                 codebook_size: int, 
                 nhead: int = 8, 
                 num_layers: int = 4):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(data_dim, latent_dim)
        self.pos_embed = PositionalEmbedding(latent_dim, max_len=2048)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.vq = ResidualVQ(
            dim = latent_dim,
            num_quantizers = 4,
            codebook_size = codebook_size,
            commitment_weight = 0.25,
            rotation_trick = True
        )
        
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, data_dim),
        )

    def forward(self, x: torch.Tensor):

        x_proj = self.input_proj(x)
        x_proj = self.pos_embed(x_proj)

        seq_len = x_proj.shape[1]
        device = x_proj.device

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        x_proj = self.transformer_encoder(x_proj, mask=causal_mask)

        quantized, indices, commit_loss = self.vq(x_proj)

        decoded_sequence = self.decoder(quantized)

        return decoded_sequence, commit_loss.mean(), indices

if __name__ == "__main__":
    model = VQVAE(data_dim=150, latent_dim=256, codebook_size=256)
    x = torch.randn(1, 4, 150)
    decoded_sequence, commit_loss, indices = model(x)
    print(decoded_sequence.shape)
    print(commit_loss)
    print(indices.shape)