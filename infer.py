import argparse
import os
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import MotionDataset
from model import AutoEncoder
from smplx import SMPL

from utils import prepare_motion_batch


def parse_args():
    p = argparse.ArgumentParser(description="Inference with AutoEncoder for motion reconstruction.")
    p.add_argument("--data-path", type=str, default="data",
                   help="root directory containing *.npz motion files")
    p.add_argument("--body-model-path", type=str, default="smpl",
                   help="directory containing SMPL body model files")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="directory containing trained checkpoints")
    p.add_argument("--best-model-name", type=str, default="ae_best_model.pth",
                   help="checkpoint filename to load")
    p.add_argument("--output-video", type=str, default=None,
                   help="output video path; defaults to traj_<idx>.mp4")
    p.add_argument("--seq-len", type=int, default=150,
                   help="sequence length for model context")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--traj-idx", type=int, default=0,
                   help="index of the trajectory to visualize")
    p.add_argument("--traj-name", type=str, default=None,
                   help="explicit trajectory key to visualize (overrides --traj-idx)")
    p.add_argument("--fps", type=int, default=30, help="fps of the output video")
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)

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

    device = get_device(args.device)
    print(f"Using device: {device}")

    dataset = MotionDataset(
        data_path=args.data_path,
        seq_len=args.seq_len,
    )
    if len(dataset.trajs) == 0:
        raise RuntimeError("Dataset contains no trajectories.")

    traj_keys = sorted(dataset.trajs.keys())
    if args.traj_name is not None:
        if args.traj_name not in dataset.trajs:
            raise ValueError(f"Trajectory {args.traj_name} not found in dataset.")
        traj_key = args.traj_name
    else:
        if args.traj_idx >= len(traj_keys):
            raise ValueError(f"traj-idx {args.traj_idx} out of range (len={len(traj_keys)})")
        traj_key = traj_keys[args.traj_idx]

    print(f"Selected trajectory: {traj_key}")
    traj = dataset.trajs[traj_key]
    poses = torch.from_numpy(traj["poses"]).float().unsqueeze(0)
    trans = torch.from_numpy(traj["trans"]).float().unsqueeze(0)
    betas = torch.from_numpy(traj["betas"][:10]).float().unsqueeze(0)

    output_video = args.output_video or f"{traj_key.replace('/', '_')}.mp4"
    print(f"Output video: {output_video}")

    print("Loading SMPL body model...")
    body_model = SMPL(model_path=args.body_model_path, gender="neutral").to(device)

    print("Loading trained autoencoder...")
    model = AutoEncoder(
        latent_dim=args.latent_dim,
        orient_dim=4,
        num_joints=24,
        joint_dim=3,
        beta_dim=10,
    ).to(device)
    model = torch.compile(model)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, args.best_model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    print("Preparing motion via SMPL...")
    batched_sample = {
        "poses": poses,
        "trans": trans,
        "betas": betas,
    }
    # Disable beta and scale augmentation during inference (set std=0)
    motion = prepare_motion_batch(batched_sample, body_model, device, beta_augment_std=0.0, scale_augment_std=0.0)

    # For inference, use original betas and scale (no augmentation)
    betas_seq = motion["betas"]
    scale_seq = motion["scale"]  # Original scale (1.0)
    global_orient_seq = motion["global_orient"]
    joints_seq = motion["joints"]

    B, T, _, _ = joints_seq.shape
    assert B == 1, "Inference currently supports batch size 1."

    context_orient = deque(maxlen=args.seq_len)
    context_joints = deque(maxlen=args.seq_len)

    recon_quats = []
    recon_joints = []
    input_quats = []
    input_joints = []

    print("Streaming through the trajectory...")
    for t in tqdm(range(T), desc="Streaming Inference"):
        orient_t = global_orient_seq[0, t]
        joints_t = joints_seq[0, t]
        input_quats.append(orient_t.detach().cpu())
        input_joints.append(joints_t.detach().cpu())

        context_orient.append(orient_t)
        context_joints.append(joints_t)

        if len(context_orient) < args.seq_len:
            recon_quats.append(orient_t.detach().cpu())
            recon_joints.append(joints_t.detach().cpu())
            continue

        orient_tensor = torch.stack(list(context_orient), dim=0).unsqueeze(0)
        joints_tensor = torch.stack(list(context_joints), dim=0).unsqueeze(0)

        with torch.no_grad():
            recon_orient_seq, recon_joints_seq = model(
                orient_tensor, joints_tensor, betas_seq, scale_seq
            )
        recon_quats.append(recon_orient_seq[:, -1].squeeze(0).detach().cpu())
        recon_joints.append(recon_joints_seq[:, -1].squeeze(0).detach().cpu())

    input_quats_np = torch.stack(input_quats, dim=0).numpy()
    input_joints_np = torch.stack(input_joints, dim=0).numpy()
    recon_quats_np = torch.stack(recon_quats, dim=0).numpy()
    recon_joints_np = torch.stack(recon_joints, dim=0).numpy()

    print("Starting animation... Close the plot window to exit.")
    animate_3d_and_quat(
        input_joints_np,
        recon_joints_np,
        input_quats_np,
        recon_quats_np,
        title=traj_key,
        output_file=output_video,
        fps=args.fps,
    )