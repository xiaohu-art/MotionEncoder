import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import joblib
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MotionDataset
from model import AutoEncoder

def parse_args():
    p = argparse.ArgumentParser(description="Train AutoEncoder with configurable args.")
    # data and path
    p.add_argument("--dataset", type=str, default="amass_train", 
                   choices=["amass_train", "amass_test",
                            "accad", "bmlhandball", "bmlmovi", "bmlrub", "cmu", 
                            "dancedb", "dfaust", "ekut", "eyes_japan", "hdm05",
                            "human4d", "humaneva", "kit", "mosh", "poseprior",
                            "sfu", "soma", "ssm", "tcdhands", "totalcapture", "transitions"],
                   help="dataset name (will auto-assign data path)")
    p.add_argument("--data-path", type=str, default=None,
                   help="data path (overrides dataset-based path if specified)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="checkpoint dir")
    p.add_argument("--best-model-name", type=str, default="ae_best_model.pth",
                   help="best model name")
    p.add_argument("--log-dir", type=str, default="runs/ae_experiment_split",
                   help="log dir")
    # training hyperparams
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=150)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # scheduler
    p.add_argument("--t-max", type=int, default=50, help="CosineAnnealingLR T_max, same as epochs")
    p.add_argument("--eta-min", type=float, default=1e-6, help="CosineAnnealingLR eta_min")
    # device
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_train_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        # For motion dataset: ((raw_input, scaled_input), scaled_target)
        (raw_input, augmented_input, scaled_input), scaled_target = batch
        raw_input = raw_input.to(device)
        augmented_input = augmented_input.to(device)
        scaled_input = scaled_input.to(device)
        scaled_target = scaled_target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for both input types
        raw_output = model(raw_input)
        augmented_output = model(augmented_input)
        scaled_output = model(scaled_input)
        
        # Compute losses for both input types (both targeting scaled output)
        loss_raw = nn.functional.mse_loss(raw_output, scaled_target)
        loss_augmented = nn.functional.mse_loss(augmented_output, scaled_target)
        loss_scaled = nn.functional.mse_loss(scaled_output, scaled_target)
        
        # Total loss is the sum of both losses
        total_loss = loss_raw + loss_augmented + loss_scaled
        
        total_loss.backward()
        optimizer.step()
        
        total_train_loss += total_loss.item()
        progress_bar.set_postfix(loss=total_loss.item())

    return total_train_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            # For motion dataset: ((raw_input, scaled_input), scaled_target)
            (raw_input, augmented_input, scaled_input), scaled_target = batch
            raw_input = raw_input.to(device)
            augmented_input = augmented_input.to(device)
            scaled_input = scaled_input.to(device)
            scaled_target = scaled_target.to(device)
            
            # Forward pass for both input types
            raw_output = model(raw_input)
            augmented_output = model(augmented_input)
            scaled_output = model(scaled_input)
            
            # Compute losses for both input types (both targeting scaled output)
            loss_raw = nn.functional.mse_loss(raw_output, scaled_target)
            loss_augmented = nn.functional.mse_loss(augmented_output, scaled_target)
            loss_scaled = nn.functional.mse_loss(scaled_output, scaled_target)
            
            # Total loss is the sum of both losses
            total_loss = loss_raw + loss_augmented + loss_scaled
            
            total_val_loss += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item())
            
    return total_val_loss / len(dataloader)

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Determine data path
    print(f"Using dataset: {args.dataset}")
    raw_data, scaled_data = MotionDataset.load_data_pair(args.dataset)
    
    # --- Paths and Logging ---
    BEST_MODEL_PATH = os.path.join(args.checkpoint_dir, args.best_model_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
        
    print("Loading data and splitting keys...")
    all_keys = list(raw_data.keys())
    train_keys, val_keys = train_test_split(all_keys, test_size=args.val_split, random_state=args.seed)
    print(f"Total sequences: {len(all_keys)}. Training: {len(train_keys)}, Validation: {len(val_keys)}")

    train_dataset = MotionDataset(  raw_data=raw_data, 
                                    scaled_data=scaled_data, 
                                    keys_to_use=train_keys, 
                                    seq_len=args.seq_len)
    val_dataset = MotionDataset(    raw_data=raw_data, 
                                    scaled_data=scaled_data, 
                                    keys_to_use=val_keys, 
                                    seq_len=args.seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model, Optimizer, Scheduler ---
    model = AutoEncoder(
                data_dim=train_dataset.data_dim,
                latent_dim=args.latent_dim, 
            ).to(device)

    try:
        model = torch.compile(
            model,
            backend="inductor",
            mode="max-autotune",
            fullgraph=False,
            dynamic=True,
        )
        print("[compile] torch.compile enabled.")
    except Exception as e:
        print(f"[compile] fell back to eager due to: {e}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = num_params / 1e6
    print(f"Number of trainable parameters: {params_m:.2f}M")
    
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
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        avg_val_loss = validate(model, val_dataloader, device)
        
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