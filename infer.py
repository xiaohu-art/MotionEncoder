import torch
import numpy as np
import joblib
from tqdm import tqdm
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import VQVAE
from dataset import VAEDataLoader

def animate_3d_comparison(joints_orig, joints_recon, title=""):
    """
    Animates a 3D comparison between original (blue) and reconstructed (red) keypoints.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_orig = ax.scatter([], [], [], c='blue', marker='o', label='Original', s=20)
    scatter_recon = ax.scatter([], [], [], c='red', marker='o', label='Reconstructed', s=20)

    all_points = np.concatenate([joints_orig, joints_recon], axis=0)
    min_vals = all_points.min(axis=(0,1))
    max_vals = all_points.max(axis=(0,1))
    ax.set_xlim(min_vals[0] - 0.2, max_vals[0] + 0.2)
    ax.set_ylim(min_vals[1] - 0.2, max_vals[1] + 0.2)
    ax.set_zlim(min_vals[2] - 0.2, max_vals[2] + 0.2)
    
    def update(num):
        scatter_orig._offsets3d = (joints_orig[num, :, 0], joints_orig[num, :, 1], joints_orig[num, :, 2])
        scatter_recon._offsets3d = (joints_recon[num, :, 0], joints_recon[num, :, 1], joints_recon[num, :, 2])
        ax.set_title(f"{title} - Frame {num}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)

    num_frames = joints_orig.shape[0]
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False) 
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN = 150
    LATENT_DIM = 256
    CODEBOOK_SIZE = 256
    DATA_PATH = "data/sfu_scaled.pkl"
    BEST_MODEL_PATH = "checkpoints/vqvae_best_model.pth"

    temp_dataset = VAEDataLoader(data={}, keys_to_use=[], seq_len=1)
    DATA_DIM = temp_dataset.data_dim
    
    print("Loading original model...")
    model = VQVAE(
        data_dim=DATA_DIM, latent_dim=LATENT_DIM, codebook_size=CODEBOOK_SIZE
    ).to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    full_data = joblib.load(DATA_PATH)
    test_key = list(full_data.keys())[0]
    motion_data = full_data[test_key]
    combined_input = torch.tensor(
        np.concatenate([
            motion_data['root_trans_offset'],
            motion_data['root_rot'],
            motion_data['smpl_joints'].reshape(motion_data['smpl_joints'].shape[0], -1)
        ], axis=1), dtype=torch.float32
    ).to(device)
    num_frames = combined_input.shape[0]

    print(f"\nStarting fixed-length streaming inference for {num_frames} frames...")
    
    context_queue = deque(maxlen=SEQ_LEN)
    reconstructions = []

    for t in tqdm(range(num_frames), desc="Streaming Inference"):
        current_frame = combined_input[t]
        context_queue.append(current_frame)
        
        current_context_tensor = torch.stack(list(context_queue), dim=0).unsqueeze(0)
        
        with torch.no_grad():
            reconstructed_context, _, _ = model(current_context_tensor)
            current_recon_frame = reconstructed_context[:, -1, :]
        
        reconstructions.append(current_recon_frame)

    reconstruction = torch.cat(reconstructions, dim=0)
    
    print("Streaming inference complete.")
    print(f"Input shape: {combined_input.shape}")
    print(f"Reconstructed shape: {reconstruction.shape}")

    print("\nPreparing data for animation...")
    
    orig_kps_np = motion_data['smpl_joints']

    recon_kps_flat = reconstruction[:, 7:]
    
    recon_kps = recon_kps_flat.reshape(num_frames, 24, 3)
    
    recon_kps_np = recon_kps.cpu().numpy()

    print("Starting animation... Close the plot window to exit.")
    animate_3d_comparison(orig_kps_np, recon_kps_np, title=f"Motion: {test_key}")