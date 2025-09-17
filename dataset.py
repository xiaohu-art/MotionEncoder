import os
import joblib
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MotionDataset(Dataset):
    """
    Dataset for raw-to-scaled motion reconstruction.
    Loads both raw and scaled data simultaneously.
    Input: raw motion trajectories
    Target: scaled motion trajectories
    """
    fps: int = 30
    
    def __init__(self, 
                 raw_data: dict, 
                 scaled_data: dict,
                 seq_len: int,
                 keys_to_use: list,
                 scale_range: tuple = (0.8, 1.0)):
        """
        Args:
        - raw_data: The pre-loaded raw data dictionary from joblib.load().
        - scaled_data: The pre-loaded scaled data dictionary from joblib.load().
        - seq_len: Sequence length for each sample.
        - keys_to_use: A list of motion sequence keys to include in this dataset.
        - scale_range: Tuple of (min_scale, max_scale) for the scale augmentation.
        """
        self.keys_to_use = keys_to_use
        self.seq_len = seq_len
        self.scale_range = scale_range

        self.trajs = []
        for key in self.keys_to_use:
            if key not in raw_data or key not in scaled_data:
                print(f"Warning: Key {key} not found in both raw and scaled data, skipping...")
                continue
                
            raw_dump = raw_data[key]
            scaled_dump = scaled_data[key]
            
            # Extract raw data
            raw_root_trans = raw_dump['root_trans_offset']
            raw_root_rot = raw_dump['root_rot']
            raw_kps = raw_dump['smpl_joints']

            # Get augmented data
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            augmented_root_trans, augmented_root_rot, augmented_kps = MotionDataset._scale_augment(
                raw_root_trans, raw_root_rot, raw_kps, scale_factor
            )
            
            # Extract scaled data (target)
            scaled_root_trans = scaled_dump['root_trans_offset']
            scaled_root_rot = scaled_dump['root_rot']
            scaled_kps = scaled_dump['smpl_joints']

            T = raw_root_trans.shape[0]
            
            # Verify both raw and scaled have same length
            if T != scaled_root_trans.shape[0]:
                print(f"Warning: Key {key} has different lengths in raw ({T}) and scaled ({scaled_root_trans.shape[0]}) data, skipping...")
                continue

            self.trajs.append(dict(
                # Raw data (input)
                raw_root_trans=raw_root_trans,
                raw_root_rot=raw_root_rot,
                raw_kps=raw_kps,
                # Augmented data (input)
                augmented_root_trans=augmented_root_trans,
                augmented_root_rot=augmented_root_rot,
                augmented_kps=augmented_kps,
                # Scaled data (target)
                scaled_root_trans=scaled_root_trans,
                scaled_root_rot=scaled_root_rot,
                scaled_kps=scaled_kps,
                T=T,
                key=key
            ))

        self.index = []
        for ti, traj in enumerate(self.trajs):
            T = traj['T']
            last_idx = T - self.seq_len
            for s in range(0, last_idx+1):
                self.index.append(dict(
                    traj_idx=ti,
                    start_idx=s,
                    end_idx=s + self.seq_len
                ))

        # Data dimension: 3 (root_trans) + 4 (root_rot) + 24*3 (joints) = 79
        self.data_dim = 3 + 4 + 24 * 3
        
        print(f"Loaded {len(self.trajs)} trajectories with {len(self.index)} total samples")

    @staticmethod
    def _scale_augment(root_trans: np.ndarray, root_rot: np.ndarray, kps: np.ndarray, scale_factor: float):
        seq_len, num_kps, _ = kps.shape
        scaled_kps = (kps - root_trans[:, None, :]) * scale_factor + root_trans[:, None, :]
        min_z = scaled_kps[..., 2].min()
        scaled_kps[..., 2] -= min_z
        scaled_root_trans = scaled_kps[:, 0].copy()
        scaled_root_rot = root_rot.copy()
        return scaled_root_trans, scaled_root_rot, scaled_kps
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ti, si, ei = self.index[idx]['traj_idx'], self.index[idx]['start_idx'], self.index[idx]['end_idx']
        traj = self.trajs[ti]
        
        # Get raw data
        raw_root_trans = torch.tensor(traj['raw_root_trans'][si:ei], dtype=torch.float32)
        raw_root_rot = torch.tensor(traj['raw_root_rot'][si:ei], dtype=torch.float32)
        raw_kps = torch.tensor(traj['raw_kps'][si:ei], dtype=torch.float32)

        # Get augmented data
        augmented_root_trans = torch.tensor(traj['augmented_root_trans'][si:ei], dtype=torch.float32)
        augmented_root_rot = torch.tensor(traj['augmented_root_rot'][si:ei], dtype=torch.float32)
        augmented_kps = torch.tensor(traj['augmented_kps'][si:ei], dtype=torch.float32)
        
        # Get scaled data
        scaled_root_trans = torch.tensor(traj['scaled_root_trans'][si:ei], dtype=torch.float32)
        scaled_root_rot = torch.tensor(traj['scaled_root_rot'][si:ei], dtype=torch.float32)
        scaled_kps = torch.tensor(traj['scaled_kps'][si:ei], dtype=torch.float32)
        
        assert raw_root_trans.shape[0] == raw_root_rot.shape[0] == raw_kps.shape[0] == self.seq_len
        assert augmented_root_trans.shape[0] == augmented_root_rot.shape[0] == augmented_kps.shape[0] == self.seq_len
        assert scaled_root_trans.shape[0] == scaled_root_rot.shape[0] == scaled_kps.shape[0] == self.seq_len

        # Flatten keypoints
        raw_kp_flat = raw_kps.reshape(self.seq_len, -1)
        augmented_kp_flat = augmented_kps.reshape(self.seq_len, -1)
        scaled_kp_flat = scaled_kps.reshape(self.seq_len, -1)
        
        # Concatenate features
        raw_data = torch.cat([raw_root_trans, raw_root_rot, raw_kp_flat], dim=1).to(torch.float32)
        augmented_data = torch.cat([augmented_root_trans, augmented_root_rot, augmented_kp_flat], dim=1).to(torch.float32)
        scaled_data = torch.cat([scaled_root_trans, scaled_root_rot, scaled_kp_flat], dim=1).to(torch.float32)
        scaled_target = torch.cat([scaled_root_trans, scaled_root_rot, scaled_kp_flat], dim=1).to(torch.float32)

        return (raw_data, augmented_data, scaled_data), scaled_target

    @classmethod
    def load_data_pair(cls, dataset_name, data_dir="data"):
        """
        Helper method to load both raw and scaled data for a given dataset.
        
        Args:
        - dataset_name: Name of the dataset (e.g., 'sfu', 'cmu', etc.)
        - data_dir: Base directory containing raw and scaled subdirectories
        
        Returns:
        - raw_data: Raw data dictionary
        - scaled_data: Scaled data dictionary
        """
        raw_path = os.path.join(data_dir, "raw", f"{dataset_name}_raw.pkl")
        scaled_path = os.path.join(data_dir, "scaled", f"{dataset_name}_scaled.pkl")
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        if not os.path.exists(scaled_path):
            raise FileNotFoundError(f"Scaled data file not found: {scaled_path}")
            
        print(f"Loading raw data from: {raw_path}")
        raw_data = joblib.load(raw_path)
        
        print(f"Loading scaled data from: {scaled_path}")
        scaled_data = joblib.load(scaled_path)
        
        return raw_data, scaled_data

if __name__ == "__main__":
    # Test the dataset
    try:
        raw_data, scaled_data = MotionDataset.load_data_pair("sfu")
        keys_to_use = list(raw_data.keys())[:5]  # Use first 5 keys for testing
        dataset = MotionDataset(
            raw_data=raw_data, 
            scaled_data=scaled_data, 
            keys_to_use=keys_to_use, 
            seq_len=100
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        for batch_idx, ((raw_input, augmented_input, scaled_data), scaled_target) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Raw input shape: {raw_input.shape}")      # [B, seq_len, data_dim]
            print(f"  Augmented input shape: {augmented_input.shape}")  # [B, seq_len, data_dim]
            print(f"  Scaled data shape: {scaled_data.shape}")  # [B, seq_len, data_dim]
            print(f"  Scaled target shape: {scaled_target.shape}")  # [B, seq_len, data_dim]
            break
            
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure you have both raw and scaled data files available.")