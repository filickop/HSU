from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class GlueDataset(Dataset):
    ROOT_DIR = "dataset/Glue"
    NPZ_PATH = "dataset/Glue/data.npz"
    def __init__(self, *, device=None, long_dim=1024):
        super().__init__()

        self.data = np.load(self.NPZ_PATH)
        self.img_count = len(self.data['names'])
        self.img_dir = Path(self.ROOT_DIR)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.long_dim = long_dim

    def __len__(self):
        # return self.img_count * self.img_count
        return 100

    def _load_image_and_kps(self, idx):
        img = cv2.imread(self.img_dir / self.data['names'][idx].item(), cv2.IMREAD_GRAYSCALE)
        points = self.data['points'][idx]

        h, w = img.shape[:2]
        scale = self.long_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        points *= scale
        img_tensor = torch.from_numpy(img/255.)[None,None].to(self.device).float()
        return img_tensor, points

    def __getitem__(self, idx):
        idx1 = idx // self.img_count
        idx2 = idx % self.img_count

        img1, kp1 = self._load_image_and_kps(idx1)
        img2, kp2 = self._load_image_and_kps(idx2)

        return {
            'image1': img1,
            'image2': img2,
            'points1': kp1,
            'points2': kp2,
        }
