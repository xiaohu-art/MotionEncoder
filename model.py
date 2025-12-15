import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import MotionDataset
from torch.utils.data import DataLoader
from smplx import SMPL

from utils import prepare_motion_batch

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
        
        decoder_input_dim = latent_dim + self.beta_dim + 1  # +1 for scale
        self.decoder_backbone = nn.Sequential(
            nn.LayerNorm(decoder_input_dim),
            nn.Linear(decoder_input_dim, latent_dim),
            nn.GELU()
        )
        self.head_orient = nn.Linear(latent_dim, self.orient_dim)
        self.head_joints = nn.Linear(latent_dim, self.num_joints * self.joint_dim)

    def forward(
        self,
        global_orient: torch.Tensor,
        joints: torch.Tensor,
        betas: torch.Tensor,
        scale: torch.Tensor = None,
    ):
        """
        Args:
            global_orient: [B, T, orient_dim] quaternion rotations.
            joints: [B, T, num_joints, joint_dim] joint coordinates.
            betas: [B, beta_dim] or [B, 1, beta_dim] shape embeddings.
            scale: [B, 1] or [B, 1, 1] scale values. If None, defaults to 1.0.
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

        # Handle scale: if None, use 1.0; ensure shape is [B, 1, 1] then expand to [B, T, 1]
        if scale is None:
            scale = torch.ones(B, 1, device=device)
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)  # [B, 1, 1]
        if scale.shape[1] != seq_len:
            scale = scale.expand(-1, seq_len, -1)  # [B, T, 1]

        decoder_input = torch.cat([encoded, betas, scale], dim=-1)
        hidden_state = self.decoder_backbone(decoder_input) # [B, T, latent_dim]

        decoded_global_orient = self.head_orient(hidden_state)               # [B, T, orient_dim]
        decoded_global_orient = F.normalize(
            decoded_global_orient,
            p=2,
            dim=-1,
            eps=1e-12
        )

        decoded_joints_flat = self.head_joints(hidden_state)                 # [B, T, num_joints*3]
        decoded_joints = decoded_joints_flat.view(
            B, T, self.num_joints, self.joint_dim
        )  # [B, T, num_joints, 3]

        return decoded_global_orient, decoded_joints

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MotionDataset(
        data_path="data",
        seq_len=150,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    body_model = SMPL(model_path="smpl", gender="neutral").to(device)

    model = AutoEncoder(
        latent_dim=256,
        orient_dim=4,
        num_joints=24,
        joint_dim=3,
        beta_dim=10,
        nhead=4,
        num_layers=2,
    ).to(device)
    
    batch = next(iter(dataloader))
    motion = prepare_motion_batch(batch, body_model, device, beta_augment_std=0.0, scale_augment_std=0.0)
    
    # For testing, use original betas and scale (no augmentation)
    betas = motion["betas"]
    scale = motion["scale"]  # Original scale (1.0)
    global_orient = motion["global_orient"]
    joints = motion["joints"]

    with torch.no_grad():
        recon_global_orient, recon_joints = model(global_orient, joints, betas, scale)

    print(f"Input global_orient shape: {global_orient.shape}")
    print(f"Input joints shape: {joints.shape}")
    print(f"Input betas shape: {betas.shape}")
    print(f"Reconstructed global_orient shape: {recon_global_orient.shape}")
    print(f"Reconstructed joints shape: {recon_joints.shape}")
