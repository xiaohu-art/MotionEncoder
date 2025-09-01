import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import joblib
from sklearn.model_selection import train_test_split # New import
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import VAEDataLoader
from model import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_train_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        reconstructed_output, commit_loss, _ = model(batch)
        
        recon_loss = nn.functional.mse_loss(reconstructed_output, batch)
        total_loss = recon_loss + commit_loss
        
        total_loss.backward()
        optimizer.step()
        
        total_train_loss += total_loss.item()
        progress_bar.set_postfix(loss=total_loss.item(), recon=recon_loss.item())

    return total_train_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            batch = batch.to(device)
            reconstructed_output, commit_loss, _ = model(batch)
            
            recon_loss = nn.functional.mse_loss(reconstructed_output, batch)
            total_loss = recon_loss + commit_loss.mean()
            
            total_val_loss += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item(), recon=recon_loss.item())
            
    return total_val_loss / len(dataloader)

if __name__ == "__main__":
    
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    SEQ_LEN = 150
    LATENT_DIM = 256
    CODEBOOK_SIZE = 256
    VAL_SPLIT = 0.2 # 20% of data for validation
    
    # --- Paths and Logging ---
    DATA_PATH = "data/sfu_scaled.pkl"
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vqvae_best_model.pth") # Save best model here
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter('runs/vqvae_experiment_split')

    print("Loading data and splitting keys...")
    full_data = joblib.load(DATA_PATH)
    all_keys = list(full_data.keys())
    train_keys, val_keys = train_test_split(all_keys, test_size=VAL_SPLIT, random_state=42)
    print(f"Total sequences: {len(all_keys)}. Training: {len(train_keys)}, Validation: {len(val_keys)}")

    train_dataset = VAEDataLoader(data=full_data, keys_to_use=train_keys, seq_len=SEQ_LEN)
    val_dataset = VAEDataLoader(data=full_data, keys_to_use=val_keys, seq_len=SEQ_LEN)
    # train_dataset = VAEDataLoader(data=full_data, keys_to_use=all_keys, seq_len=SEQ_LEN)
    # val_dataset = VAEDataLoader(data=full_data, keys_to_use=all_keys, seq_len=SEQ_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Model, Optimizer, Scheduler ---
    model = VQVAE(
                data_dim=train_dataset.data_dim,
                latent_dim=LATENT_DIM, 
                codebook_size=CODEBOOK_SIZE,
            ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = num_params / 1e6
    print(f"Number of trainable parameters: {params_m:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=5,
            )

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf') # For saving the best model
    global_step = 0

    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        avg_val_loss = validate(model, val_dataloader, device)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning Rate/lr', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step(avg_val_loss)
        
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