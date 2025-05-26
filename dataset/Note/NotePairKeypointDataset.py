import json
import os
import cv2
import torch


class NotePairKeypointDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_dir, image_size=(200, 200)):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.data) - 1  # napr. susedné páry

    def __getitem__(self, idx):
        def load_img_and_kps(entry):
            img_path = os.path.join(self.image_dir, os.path.basename(entry["path"]))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.image_size)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            kps = torch.tensor([[kp["x"], kp["y"]] for kp in entry["keypoints"]], dtype=torch.float32)
            return img_tensor, kps

        img0, kp0 = load_img_and_kps(self.data[idx])
        img1, kp1 = load_img_and_kps(self.data[idx + 1])

        return {
            "image0": img0, "keypoints0": kp0,
            "image1": img1, "keypoints1": kp1
        }
