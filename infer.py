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
from dataset import MotionDataset

def parse_args():
    p = argparse.ArgumentParser(description="Inference with AutoEncoder for motion reconstruction.")
    # data and path
    p.add_argument("--dataset", type=str, default="amass_test", 
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
    # input type for visualization
    p.add_argument("--input-type", type=str, choices=["raw", "scaled"], default="raw",
                   help="Input type for inference: 'raw' or 'scaled'")
    return p.parse_args()

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)

def animate_3d_comparison(joints_input, joints_recon, title="", output_file="*.mp4", fps=30):
    """
    Animates a 3D comparison between input (blue) and reconstructed (red) keypoints.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_input = ax.scatter([], [], [], c='blue', marker='o', label='Input', s=20)
    scatter_recon = ax.scatter([], [], [], c='red', marker='o', label='Reconstructed', s=20)

    all_points = np.concatenate([joints_input, joints_recon], axis=0)
    min_vals = all_points.min(axis=(0,1))
    max_vals = all_points.max(axis=(0,1))
    ax.set_xlim(min_vals[0] - 0.2, max_vals[0] + 0.2)
    ax.set_ylim(min_vals[1] - 0.2, max_vals[1] + 0.2)
    ax.set_zlim(min_vals[2] - 0.2, max_vals[2] + 0.2)
    
    def update(num):
        scatter_input._offsets3d = (joints_input[num, :, 0], joints_input[num, :, 1], joints_input[num, :, 2])
        scatter_recon._offsets3d = (joints_recon[num, :, 0], joints_recon[num, :, 1], joints_recon[num, :, 2])
        ax.set_title(f"{title} - Frame {num}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)

    num_frames = joints_input.shape[0]
    interval = int(1000 / fps)  # Convert fps to interval in milliseconds
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    ani.save(output_file, writer='ffmpeg', fps=fps)
    # plt.show()

def _quat_angle_deg(q1, q2):
    cos = np.abs(np.sum(q1 * q2, axis=-1))
    cos = np.clip(cos, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cos))

def animate_3d_and_quat(joints_input, joints_recon, quat_input, quat_recon,
                        title="", output_file="*.mp4", fps=30):
    """
    Left: 3D keypoints (blue=input, red=recon) with optional orientation triads.
    Right-top: quaternion components (w,x,y,z) for input (dashed) and recon (solid).
    Right-bottom: geodesic angle error (degrees).
    """
    ang_deg     = _quat_angle_deg(quat_input, quat_recon)  # [T]

    T = joints_input.shape[0]
    assert joints_recon.shape[0] == T == quat_input.shape[0] == quat_recon.shape[0]

    # --- figure layout ---
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.4], height_ratios=[2.5, 1.0])
    ax3d   = fig.add_subplot(gs[:, 0], projection='3d')
    axQuat = fig.add_subplot(gs[0, 1])
    axErr  = fig.add_subplot(gs[1, 1])

    # --- 3D scatter (keypoints) ---
    scatter_input = ax3d.scatter([], [], [], c='blue', marker='o', label='Input', s=15)
    scatter_recon = ax3d.scatter([], [], [], c='red', marker='o', label='Reconstructed', s=15)

    all_points = np.concatenate([joints_input, joints_recon], axis=0)  # [2T, J, 3]
    min_vals = all_points.min(axis=(0,1))
    max_vals = all_points.max(axis=(0,1))
    pad = 0.2
    ax3d.set_xlim(min_vals[0]-pad, max_vals[0]+pad)
    ax3d.set_ylim(min_vals[1]-pad, max_vals[1]+pad)
    ax3d.set_zlim(min_vals[2]-pad, max_vals[2]+pad)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title(title)
    ax3d.legend(loc='upper right')

    # global frame
    ax3d.quiver(0,0,0, 1,0,0, color='r', length=0.12)
    ax3d.quiver(0,0,0, 0,1,0, color='g', length=0.12)
    ax3d.quiver(0,0,0, 0,0,1, color='b', length=0.12)

    # --- Quaternion components panel ---
    t = np.arange(T)
    labels = ['w','x','y','z']
    lines_in = []
    lines_rec = []
    for i in range(4):
        (line_in,)  = axQuat.plot(t, quat_input[:, i],  linestyle='--', linewidth=1.2, label=f'{labels[i]} (in)')
        (line_rec,) = axQuat.plot(t, quat_recon[:, i],  linestyle='-',  linewidth=1.5, label=f'{labels[i]} (rec)')
        lines_in.append(line_in); lines_rec.append(line_rec)
    cursor_quat = axQuat.axvline(0, color='k', linewidth=1)
    axQuat.set_xlim(0, T-1)
    axQuat.set_ylim(-1.05, 1.05)
    axQuat.set_title('Root rotation quaternion (components)')
    axQuat.set_xlabel('Frame'); axQuat.set_ylabel('value')
    axQuat.legend(ncol=2, fontsize=8)

    # --- Angle error panel ---
    (line_err,) = axErr.plot(t, ang_deg)
    cursor_err = axErr.axvline(0, color='k', linewidth=1)
    axErr.set_xlim(0, T-1)
    axErr.set_ylim(0.0, max(1.0, np.nanmax(ang_deg) * 1.1))
    axErr.set_title('Geodesic angle error (deg)')
    axErr.set_xlabel('Frame'); axErr.set_ylabel('deg')

    # --- update function ---
    def update(f):
        # keypoints
        p_in  = joints_input[f]
        p_rec = joints_recon[f]
        scatter_input._offsets3d = (p_in[:,0],  p_in[:,1],  p_in[:,2])
        scatter_recon._offsets3d = (p_rec[:,0], p_rec[:,1], p_rec[:,2])
        ax3d.set_title(f"{title} - Frame {f}")
        # move the cursors
        cursor_quat.set_xdata([f, f])
        cursor_err.set_xdata([f, f])
        return (scatter_input, scatter_recon, cursor_quat, cursor_err)

    interval = int(1000 / fps)
    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    ani.save(output_file, writer='ffmpeg', fps=fps)

if __name__ == "__main__":
    args = parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load both raw and scaled data
    print(f"Using dataset: {args.dataset}")
    raw_data, scaled_data = MotionDataset.load_data_pair(args.dataset)
    
    # Generate output video filename
    output_video = args.output_video if args.output_video is not None else f"{args.dataset}_{args.input_type}_{args.test_key}.mp4"
    print(f"Output video: {output_video}")

    # Get data dimension from a sample
    temp_dataset = MotionDataset(raw_data=raw_data, scaled_data=scaled_data, keys_to_use=[list(raw_data.keys())[0]], seq_len=1)
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

    test_key = list(raw_data.keys())[args.test_key]
    
    # Choose input data based on input_type argument
    if args.input_type == "raw":
        motion_data = raw_data[test_key]
        print(f"Using raw data for inference")
    else:  # scaled
        motion_data = scaled_data[test_key]
        print(f"Using scaled data for inference")
    
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
    
    # Input keypoints (model input)
    input_kps_np = motion_data['smpl_joints']
    input_quat_np = motion_data['root_rot']
    
    # Reconstructed keypoints (model output)
    recon_kps_flat = reconstruction[:, 7:]
    recon_kps = recon_kps_flat.reshape(num_frames, 24, 3)
    recon_kps_np = recon_kps.cpu().numpy()

    recon_quat_np = reconstruction[:, 3:7].cpu().numpy()

    print("Starting animation... Close the plot window to exit.")
    
    # Create appropriate title based on input type
    if args.input_type == "raw":
        title = f"Raw→Scaled: {test_key}"
        # Show: Raw input (blue) vs Reconstructed scaled output (red)
        # animate_3d_comparison(input_kps_np, recon_kps_np, title=title, 
        #                      output_file=output_video, fps=args.fps)
        animate_3d_and_quat(input_kps_np, recon_kps_np, input_quat_np, recon_quat_np, title=title, 
                            output_file=output_video, fps=args.fps)
    else:  # scaled
        title = f"Scaled→Scaled: {test_key}"
        # Show: Scaled input (blue) vs Reconstructed scaled output (red)
        # animate_3d_comparison(input_kps_np, recon_kps_np, title=title, 
        #                      output_file=output_video, fps=args.fps)
        animate_3d_and_quat(input_kps_np, recon_kps_np, input_quat_np, recon_quat_np, title=title, 
                            output_file=output_video, fps=args.fps)