import torch
from torch.utils.data import Dataset
import joblib
from torch.utils.data import DataLoader

class VAEDataLoader(Dataset):
    fps: int = 30
    def __init__(self, 
                 data: dict, 
                 seq_len: int,
                 keys_to_use: list):
        """
        Args:
        - data: The pre-loaded data dictionary from joblib.load().
        - keys_to_use: A list of motion sequence keys to include in this dataset.
        """
        self.keys_to_use = keys_to_use
        self.seq_len = seq_len

        self.trajs = []
        for key in self.keys_to_use:
            data_dump = data[key]
            root_trans = data_dump['root_trans_offset']
            root_rot = data_dump['root_rot']
            kps = data_dump['smpl_joints']

            T = root_trans.shape[0]

            self.trajs.append(dict(
                root_trans=root_trans,
                root_rot=root_rot,
                kps=kps,
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

        self.data_dim = 3 + 4 + 24 * 3
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ti, si, ei = self.index[idx]['traj_idx'], self.index[idx]['start_idx'], self.index[idx]['end_idx']
        traj = self.trajs[ti]
        root_trans = torch.tensor(traj['root_trans'][si:ei], dtype=torch.float32)
        root_rot = torch.tensor(traj['root_rot'][si:ei], dtype=torch.float32)
        kps = torch.tensor(traj['kps'][si:ei], dtype=torch.float32)
        assert root_trans.shape[0] == root_rot.shape[0] == kps.shape[0] == self.seq_len

        kp_flat = kps.reshape(self.seq_len, -1)
        return torch.cat([root_trans, root_rot, kp_flat], dim=1).to(torch.float32)

if __name__ == "__main__":
    data = joblib.load("data/sfu_scaled.pkl")
    keys_to_use = list(data.keys())
    dataset = VAEDataLoader(data=data, keys_to_use=keys_to_use, seq_len=100)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch.shape)  # [B, seq_len, data_dim]
        break