import json
import cv2
import torch
import numpy as np
from pathlib import Path


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str | list[str], image_dir: str | list[str], *, long_dim=1024, device=None):
        if isinstance(json_path, str):
            json_path = [json_path]
        if isinstance(image_dir, str):
            image_dir = [image_dir]

        assert len(json_path) == len(image_dir)

        self.data = []
        self.total_image_count = 0
        for path in json_path:
            with open(path, 'r') as f:
                self.data.append(json.load(f))
                self.total_image_count += len(self.data[-1])

        self.image_dirs = [Path(x) for x in image_dir]
        self.long_dim = long_dim
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def __len__(self):
        return self.total_image_count - 1  # napr. susedné páry

    def _get_data(self, idx):
        for dataset_idx, data in enumerate(self.data):
            if idx < len(data):
                return data[idx], dataset_idx
            idx -= len(data)

    def _load_img_and_kps(self, entry, image_dir):
        img_path = image_dir / Path(entry["path"]).name
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kps = np.array([[kp["x"], kp["y"]] for kp in entry["keypoints"]])

        h, w = img.shape[:2]
        scale = self.long_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        kps *= scale

        img_tensor = torch.from_numpy(img/255.)[None, None].to(self.device).float()
        return img_tensor, kps

    def __getitem__(self, idx):
        data1, dataset_idx1 = self._get_data(idx)
        img1, kp1 = self._load_img_and_kps(data1, self.image_dirs[dataset_idx1])
        data2, dataset_idx2 = self._get_data(idx + 1)
        img2, kp2 = self._load_img_and_kps(data2, self.image_dirs[dataset_idx2])

        return {
            "image1": img1,
            "image2": img2,
            "points1": kp1,
            "points2": kp2,
        }
