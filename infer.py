import torch
import numpy as np
import joblib
import argparse
import os
from tqdm import tqdm
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import AutoEncoder
from dataset import VAEDataLoader

def get_dataset_path(dataset_name):
    """Get data path based on dataset name."""
    dataset_paths = f"data/scaled/{dataset_name}_scaled.pkl"
    return dataset_paths

def parse_args():
    p = argparse.ArgumentParser(description="Inference with AutoEncoder for motion reconstruction.")
    # data and path
    p.add_argument("--dataset", type=str, default="sfu", 
                   choices=["accad", "bmlhandball", "bmlmovi", "bmlrub", "cmu", 
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
    p.add_argument("--output-video", type=str, default=None,
                   help="output video name (auto-generated from dataset and test-key if not specified)")
    # model params
    p.add_argument("--seq-len", type=int, default=150)
    p.add_argument("--latent-dim", type=int, default=256)
    # inference params
    p.add_argument("--test-key", type=int, default=0,
                   help="test key, use the first one if not specified")
    p.add_argument("--fps", type=int, default=30, help="fps of the output video")
    # device
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)

def animate_3d_comparison(joints_orig, joints_recon, title="", output_file="ae_sfu.mp4", fps=30):
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
    interval = int(1000 / fps)  # Convert fps to interval in milliseconds
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    ani.save(output_file, writer='ffmpeg', fps=fps)
    # plt.show()

if __name__ == "__main__":
    args = parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Determine data path
    data_path = args.data_path if args.data_path is not None else get_dataset_path(args.dataset)
    print(f"Using dataset: {args.dataset}")
    print(f"Data path: {data_path}")
    
    # Generate output video filename
    output_video = args.output_video if args.output_video is not None else f"{args.dataset}_{args.test_key}.mp4"
    print(f"Output video: {output_video}")

    temp_dataset = VAEDataLoader(data={}, keys_to_use=[], seq_len=1)
    DATA_DIM = temp_dataset.data_dim
    
    print("Loading original model...")
    model = AutoEncoder(
        data_dim=DATA_DIM, latent_dim=args.latent_dim
    ).to(device)
    BEST_MODEL_PATH = os.path.join(args.checkpoint_dir, args.best_model_name)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    full_data = joblib.load(data_path)
    test_key = list(full_data.keys())[args.test_key]
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
    
    context_queue = deque(maxlen=args.seq_len)
    reconstructions = []

    for t in tqdm(range(num_frames), desc="Streaming Inference"):
        current_frame = combined_input[t]
        context_queue.append(current_frame)
        
        current_context_tensor = torch.stack(list(context_queue), dim=0).unsqueeze(0)
        
        with torch.no_grad():
            reconstructed_context = model(current_context_tensor)
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
    animate_3d_comparison(orig_kps_np, recon_kps_np, title=f"Motion: {test_key}", 
                         output_file=output_video, fps=args.fps)