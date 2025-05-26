import torch
import os
import json
import cv2
from torch.utils.data import Dataset


class NoteKeypointDataset(Dataset):
    def __init__(self, json_path, image_dir, image_size=(200, 200), transform=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform

        with open(json_path, 'r') as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(entry["path"]))
        keypoints = entry["keypoints"]

        # Načítanie a predspracovanie obrázka
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # [1, H, W]

        # Načítanie keypointov
        keypoints_tensor = torch.tensor([[kp["x"], kp["y"]] for kp in keypoints], dtype=torch.float32)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return {
            "image": img_tensor,          # shape: [1, H, W]
            "keypoints": keypoints_tensor, # shape: [N, 2]
            "path": img_path
        }
