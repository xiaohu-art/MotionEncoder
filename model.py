import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

SMPL_J24_NUM_NODES = 24
SMPL_J24_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),    # left leg
    (0, 2), (2, 5), (5, 8), (8, 11),    # right leg
    (0, 3), (3, 6), (6, 9),             # spine
    (9, 12), (12, 15),                  # head
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), # left arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), # right arm
]

class SpatialGNN(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 64):
        super().__init__()
        self.num_nodes = SMPL_J24_NUM_NODES
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.register_buffer(
            'edges', 
            torch.tensor(SMPL_J24_EDGES, dtype=torch.long).t().contiguous()
        )
        
        self.gnn1 = GCNConv(in_dim, out_dim * 2)
        self.gnn2 = GCNConv(out_dim * 2, out_dim)
        
        self.cache = {}

    def _batch_edges(self, G: int, N: int):
        if (G, N) in self.cache:
            return self.cache[(G, N)]

        _, E = self.edges.shape
        edges_g = self.edges.unsqueeze(0).expand(G, -1, -1).clone()
        offsets = torch.arange(G, device=self.edges.device).view(G, 1, 1) * N
        edges_g += offsets                      # [G, 2, E]
        edges = edges_g.permute(1, 0, 2).reshape(2, G * E) 
        self.cache[(G, N)] = edges
        return edges

    def forward(self, joints_xyz: torch.Tensor):
        '''
        joints_xyz: [B, T, N, 3]
        return:
            node_features: [B, T, N, out_dim]
        '''
        B, T, N, _ = joints_xyz.shape
        x = joints_xyz.reshape(B * T * N, -1)    # [B*T*N, 3]
        edges = self._batch_edges(G=B * T, N=N)

        x = self.gnn1(x, edges)
        x = F.gelu(x)
        x = self.gnn2(x, edges)
        x = x.reshape(B, T, N, -1) # [B, T, N, out_dim]
        return x

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

class QuatUnitNorm(nn.Module):
    def __init__(self, rot_start=3, rot_dim=4):
        super().__init__()
        self.rot_start = rot_start
        self.rot_dim = rot_dim

    def forward(self, y: torch.Tensor):
        # y: [B, T, D]
        s, e = self.rot_start, self.rot_start + self.rot_dim
        rt  = y[..., :s]
        rr  = y[..., s:e]
        kps = y[..., e:]
        rr  = F.normalize(rr, p=2, dim=-1, eps=1e-12)  # unit quaternion
        return torch.cat([rt, rr, kps], dim=-1)


class AutoEncoder(nn.Module):
    def __init__(self, 
                 data_dim: int,
                 latent_dim: int, 
                 nhead: int = 8, 
                 num_layers: int = 4):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self.spatial_gnn = SpatialGNN()

        per_frame_dim = 7 + self.spatial_gnn.num_nodes * self.spatial_gnn.out_dim
        self.input_proj = nn.Linear(per_frame_dim, latent_dim)
        self.pos_embed = PositionalEmbedding(latent_dim, max_len=2048)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, data_dim),
        )

        self.quat_unit_norm = QuatUnitNorm(rot_start=3, rot_dim=4)

    def forward(self, x: torch.Tensor):
        '''
        x: [B, T, D]
        return:
            decoded_sequence: [B, T, D]
        '''
        B, T, D = x.shape

        root = x[..., :7]
        kps = x[..., 7:].reshape(B, T, self.spatial_gnn.num_nodes, -1)
        kps_feat = self.spatial_gnn(kps)        # [B, T, N, latent_dim]

        x = torch.cat([root, kps_feat.reshape(B, T, -1)], dim=-1)   # [B, T, 7 + N * latent_dim]
        x_proj = self.input_proj(x)
        x_proj = self.pos_embed(x_proj)

        seq_len = x_proj.shape[1]
        device = x_proj.device

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        x_proj = self.transformer_encoder(x_proj, mask=causal_mask)

        decoded_sequence = self.decoder(x_proj)
        decoded_sequence = self.quat_unit_norm(decoded_sequence)

        return decoded_sequence