import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from smplx import SMPL

from dataset import MotionDataset
from model import AutoEncoder

from utils import quat_error_magnitude, prepare_motion_batch

def parse_args():
    p = argparse.ArgumentParser(description="Train AutoEncoder with configurable args.")
    # data and path
    p.add_argument("--data-path", type=str, default="data",
                   help="absolute path to directory containing *.npz")
    p.add_argument("--body-model-path", type=str, default="smpl",
                   help="directory containing SMPL body model files")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="checkpoint dir")
    p.add_argument("--best-model-name", type=str, default="ae_best_model.pth",
                   help="best model name")
    p.add_argument("--log-dir", type=str, default="runs/ae_experiment_split",
                   help="log dir")
    # training hyperparams
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=150)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    # scheduler
    p.add_argument("--t-max", type=int, default=20, help="CosineAnnealingLR T_max, same as epochs")
    p.add_argument("--eta-min", type=float, default=1e-6, help="CosineAnnealingLR eta_min")
    # device
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)

scaler = torch.amp.GradScaler('cuda')

def train_one_epoch(model, dataloader, body_model, device):
    model.train()
    total_train_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        motion = prepare_motion_batch(batch, body_model, device)
        # Encoder input: original beta's global_orient and joints (scale=1)
        global_orient = motion["global_orient"]
        joints = motion["joints"]
        # Decoder modulation: augmented beta + augmented scale
        betas_augmented = motion["betas_augmented"]
        scale_augmented = motion["scale_augmented"]
        # Loss target: augmented beta+scale's global_orient and joints
        global_orient_target = motion["global_orient_target"]
        joints_target = motion["joints_target"]
        
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            # Encoder uses original beta's motion (scale=1), decoder uses augmented beta + scale
            recon_global_orient, recon_joints = model(global_orient, joints, betas_augmented, scale_augmented)
            # Loss computed against augmented beta+scale's targets
            orientation_loss = quat_error_magnitude(recon_global_orient, global_orient_target).mean()
            joint_loss = F.mse_loss(recon_joints, joints_target)
            loss = orientation_loss + joint_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        progress_bar.set_postfix(
            loss=loss.item(),
            orientation_loss=orientation_loss.item(),
            joint_loss=joint_loss.item()
        )
    return total_train_loss / len(dataloader)

def validate(model, dataloader, body_model, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            motion = prepare_motion_batch(batch, body_model, device)
            # Encoder input: original beta's global_orient and joints (scale=1)
            global_orient = motion["global_orient"]
            joints = motion["joints"]
            # Decoder modulation: augmented beta + augmented scale
            betas_augmented = motion["betas_augmented"]
            scale_augmented = motion["scale_augmented"]
            # Loss target: augmented beta+scale's global_orient and joints
            global_orient_target = motion["global_orient_target"]
            joints_target = motion["joints_target"]

            # Encoder uses original beta's motion (scale=1), decoder uses augmented beta + scale
            recon_global_orient, recon_joints = model(global_orient, joints, betas_augmented, scale_augmented)
            # Loss computed against augmented beta+scale's targets
            orientation_loss = quat_error_magnitude(recon_global_orient, global_orient_target).mean()
            joint_loss = F.mse_loss(recon_joints, joints_target)
            loss = orientation_loss + joint_loss
            total_val_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), orientation_loss=orientation_loss.item(), joint_loss=joint_loss.item())
            
    return total_val_loss / len(dataloader)

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    print(f"Loading motions from: {args.data_path}")
    
    # --- Paths and Logging ---
    BEST_MODEL_PATH = os.path.join(args.checkpoint_dir, args.best_model_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
        
    base_dataset = MotionDataset(
        data_path=args.data_path,
        seq_len=args.seq_len,
    )

    val_size = int(len(base_dataset) * args.val_split)
    train_size = len(base_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("val_split must be between 0 and 1 (exclusive) and dataset must have enough samples.")

    train_dataset, val_dataset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    body_model = SMPL(model_path=args.body_model_path, gender="neutral").to(device)

    # --- Model, Optimizer, Scheduler ---
    model = AutoEncoder(
        latent_dim=args.latent_dim,
        orient_dim=4,
        num_joints=24,
        joint_dim=3,
        beta_dim=10,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = num_params / 1e6
    print(f"Number of trainable parameters: {params_m:.2f}M")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.t_max,
        eta_min=args.eta_min
    )

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf') # For saving the best model
    global_step = 0

    for epoch in range(args.epochs):
        avg_train_loss = train_one_epoch(model, train_dataloader, body_model, device)
        avg_val_loss = validate(model, val_dataloader, body_model, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning Rate/lr', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model found! Saving to {BEST_MODEL_PATH}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, BEST_MODEL_PATH)
    writer.close()
    print("Training finished!")