import os
import numpy as np

import glob

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        scale_range: tuple = (0.5, 1.5)
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
                        poses=data["poses"].astype(np.float32),  # (T, 156)
                        trans=data["trans"].astype(np.float32),  # (T, 3)
                        betas=data["betas"].astype(np.float32)
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

        poses = torch.from_numpy(traj["poses"][start:end]).float()
        trans = torch.from_numpy(traj["trans"][start:end]).float()
        betas = torch.from_numpy(traj["betas"][:10]).float()

        return {
            "poses": poses,
            "trans": trans,
            "betas": betas,
        }

if __name__ == "__main__":
    dataset = MotionDataset(
        data_path="data/train",
        seq_len=150
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Poses shape: {batch['poses'].shape}")
    print(f"Trans shape: {batch['trans'].shape}")
    print(f"Betas shape: {batch['betas'].shape}")