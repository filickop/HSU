from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class GlueDataset(Dataset):
    LONG_DIM = 1024
    ROOT_DIR = "dataset/Glue"
    NPZ_PATH = "dataset/Glue/data.npz"
    def __init__(self, *, device=None):
        super().__init__()

        self.data = np.load(self.NPZ_PATH)
        self.img_count = len(self.data['names'])
        self.img_dir = Path(self.ROOT_DIR)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device


    def __len__(self):
        # return self.img_count * self.img_count
        return 100

    def _resize_image(self, img, points):
        h, w = img.shape[:2]
        scale = self.LONG_DIM / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale))), points * scale


    def __getitem__(self, idx):
        idx1 = idx // self.img_count
        idx2 = idx % self.img_count

        image1 = cv2.imread(self.img_dir / self.data['names'][idx1].item(), cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.img_dir / self.data['names'][idx2].item(), cv2.IMREAD_GRAYSCALE)
        image1, points1 = self._resize_image(image1, self.data['points'][idx1])
        image2, points2 = self._resize_image(image2, self.data['points'][idx2])

        image1 = torch.from_numpy(image1/255.)[None,None].to(self.device).float()
        image2 = torch.from_numpy(image2/255.)[None,None].to(self.device).float()

        data = {
            'image1': image1,
            'image2': image2,
            'points1': points1,
            'points2': points2,
        }

        return data
