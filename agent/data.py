import numpy as np

from torch.utils.data import Dataset
import shelve

class WaymaxRasterDatasetFromBuffer(Dataset):
    def __init__(self, filename):
        self.buffer = shelve.open(filename)

    def __len__(self):
        return 100_000

    def __getitem__(self, idx):
        data = self.buffer[f'{idx}']
        waypoints = data['waypoints']

        target_time = np.random.randint(9, 180)
        target_time = np.clip(9, 90, target_time)
        data['target_point'] = np.array(waypoints[target_time])
        return data
