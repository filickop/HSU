import json
import cv2
import torch
import numpy as np
from pathlib import Path


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_dir, long_dim=1024, device=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = Path(image_dir)
        self.long_dim = long_dim
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def __len__(self):
        return len(self.data) - 1  # napr. susedné páry

    def _load_img_and_kps(self, entry):
        img_path = self.image_dir / Path(entry["path"]).name
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kps = np.array([[kp["x"], kp["y"]] for kp in entry["keypoints"]])

        h, w = img.shape[:2]
        scale = self.long_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        kps *= scale

        img_tensor = torch.from_numpy(img/255.)[None, None].to(self.device).float()
        return img_tensor, kps

    def __getitem__(self, idx):
        img1, kp1 = self._load_img_and_kps(self.data[idx])
        img2, kp2 = self._load_img_and_kps(self.data[idx + 1])

        return {
            "image1": img1,
            "image2": img2,
            "points1": kp1,
            "points2": kp2,
        }
