import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import MotionDataset
from torch.utils.data import DataLoader

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
    def __init__(
        self,
        latent_dim: int,
        orient_dim: int = 4,
        num_joints: int = 24,
        joint_dim: int = 3,
        beta_dim: int = 10,
        nhead: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.orient_dim = orient_dim
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.beta_dim = beta_dim
        self.motion_dim = self.orient_dim + self.num_joints * self.joint_dim
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(self.motion_dim, latent_dim)
        self.pos_embed = PositionalEmbedding(latent_dim, max_len=2048)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_input_dim = latent_dim + self.beta_dim
        self.decoder = nn.Sequential(
            nn.LayerNorm(decoder_input_dim),
            nn.Linear(decoder_input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.motion_dim),
        )

        self.quat_unit_norm = QuatUnitNorm(rot_start=0, rot_dim=self.orient_dim)

    def forward(
        self,
        global_orient: torch.Tensor,
        joints: torch.Tensor,
        betas: torch.Tensor,
    ):
        """
        Args:
            global_orient: [B, T, orient_dim] quaternion rotations.
            joints: [B, T, num_joints, joint_dim] joint coordinates.
            betas: [B, beta_dim] or [B, 1, beta_dim] shape embeddings.
        """
        B, T, _, _ = joints.shape
        motion = torch.cat(
            [global_orient, joints.reshape(B, T, self.num_joints * self.joint_dim)],
            dim=-1,
        )

        x_proj = self.input_proj(motion)
        x_proj = self.pos_embed(x_proj)

        seq_len = x_proj.shape[1]
        device = x_proj.device

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        encoded = self.transformer_encoder(x_proj, mask=causal_mask)

        if betas.dim() == 2:
            betas = betas.unsqueeze(1)
        if betas.shape[1] != seq_len:
            betas = betas.expand(-1, seq_len, -1)

        decoder_input = torch.cat([encoded, betas], dim=-1)
        decoded_sequence = self.decoder(decoder_input)
        decoded_sequence = self.quat_unit_norm(decoded_sequence)

        return decoded_sequence

if __name__ == "__main__":
    torch.manual_seed(0)
    dataset = MotionDataset(
        data_path="data/train",
        seq_len=150,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = AutoEncoder(
        latent_dim=256,
        orient_dim=4,
        num_joints=24,
        joint_dim=3,
        beta_dim=10,
        nhead=4,
        num_layers=2,
    )

    batch = next(iter(dataloader))
    betas = batch["betas"].squeeze(1)  # [B, beta_dim]
    global_orient = batch["global_orient"]  # [B, T, 4]
    joints = batch["joints"]  # [B, T, 24, 3]

    with torch.no_grad():
        recon = model(global_orient, joints, betas)

    print(f"Input global_orient shape: {global_orient.shape}")
    print(f"Input joints shape: {joints.shape}")
    print(f"Input betas shape: {betas.shape}")
    print(f"Reconstructed motion shape: {recon.shape}")
