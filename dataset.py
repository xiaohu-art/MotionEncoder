import os
import numpy as np

import glob
from smplx import SMPL

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_from_angle_axis(angle_axis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert axis-angle rotation vectors to quaternions.

    Args:
        angle_axis: Rotation vectors of shape (..., 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).
    """
    angle = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)
    axis = angle_axis / angle.clamp(min=eps)
    theta = angle / 2
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


class MotionDataset(Dataset):
    """
    Dataset for SMPL motions stored as AMASS-style .npz files.
    Each .npz file is expected to contain at least the keys:
    `poses` (T, 156) and `trans` (T, 3). Optional keys such as
    `betas`, `dmpls`, `mocap_framerate`, and `gender` are preserved
    if present.
    """

    input_fps: int = 30

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        scale_range: tuple = (0.5, 1.5),
        body_model_path: str = "./smpl"
    ):
        """
        Args:
            data_path: Root directory that contains *.npz motion files.
            seq_len: Number of frames per training sample.
            scale_range: Reserved for future augmentation support.
        """
        super().__init__()

        self.data_path = os.path.abspath(data_path)
        self.seq_len = int(seq_len)
        self.scale_range = scale_range

        glob_pattern = "**/*.npz"
        self.file_paths = glob.glob(os.path.join(self.data_path, glob_pattern), recursive=True)

        self.trajs = {}
        self.index = []
        self._load_all_npz()
        print(f"Loaded {len(self.trajs)} trajectories with {len(self.index)} total samples")

        self.body_model = SMPL(model_path=body_model_path, gender="neutral")

    def _load_all_npz(self):
        for file_path in self.file_paths:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if "poses" not in data or "trans" not in data:
                        print(f"[WARN] '{file_path}' missing required keys, skipping.")
                        continue
                    
                    parts = os.path.splitext(file_path)[0].split("/")[-3:]
                    name = "-".join(parts)
                    traj = dict(
                        poses=data["poses"],  # (T, 156)
                        trans=data["trans"],  # (T, 3)
                        betas=data["betas"]
                    )
            except Exception as exc:
                print(f"[WARN] Failed to load '{file_path}': {exc}")
                continue

            T = traj["poses"].shape[0]
            if T < self.seq_len:
                print(
                    f"[WARN] '{file_path}' has only {T} frames (seq_len={self.seq_len}), skipping."
                )
                continue

            traj["length"] = T
            self.trajs[name] = traj

        if not self.trajs:
            raise RuntimeError(
                "No valid motion trajectories were loaded. "
                "Please verify the contents of your data directory."
            )
        
        for name, traj in self.trajs.items():
            T = traj["length"]
            last_idx = T - self.seq_len
            for s in range(0, last_idx+1):
                self.index.append(
                    dict(
                        name=name,
                        start=s,
                        end=s + self.seq_len
                    )
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample = self.index[idx]
        traj = self.trajs[sample["name"]]
        start, end = sample["start"], sample["end"]

        poses = traj["poses"][start:end].reshape(self.seq_len, -1, 3)
        trans = traj["trans"][start:end]
        betas = traj["betas"][:10]

        trans = torch.from_numpy(trans).float()
        betas = torch.from_numpy(betas).reshape(-1, 10).float()
        poses = torch.from_numpy(poses)[:, :22].float()
        global_orient = poses[:, 0].float()
        body_pose = poses[:, 1:].float()
        hand_pose = torch.zeros(poses.shape[0], 2, 3).float()

        smpl_output = self.body_model.forward(
            betas=betas,
            body_pose=torch.cat([body_pose, hand_pose], dim=1).reshape(-1, 69).float(),
            global_orient=global_orient,
            transl=trans,
        )

        vertices = smpl_output.vertices
        global_orient = smpl_output.global_orient
        joints = smpl_output.joints[:, :24, :]

        height_offset = vertices[..., 2].min()
        vertices[..., 2] -= height_offset
        joints[..., 2] -= height_offset
        
        results = {
            "betas": betas,
            "global_orient": quat_from_angle_axis(global_orient),
            "joints": joints,
        }

        return results

if __name__ == "__main__":
    # Test the dataset
    dataset = MotionDataset(
        data_path="data/train",
        seq_len=150
    )
    breakpoint()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_idx, (results) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Betas shape: {results['betas'].shape}")
        print(f"  Global orient shape: {results['global_orient'].shape}")
        print(f"  Joints shape: {results['joints'].shape}")
        break