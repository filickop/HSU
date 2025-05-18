import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
import numpy as np
from kornia.feature import LoFTR, LoFTRConfig, LoFTRPreprocessor

# === Dataset ===
class KeypointMatchingDataset(Dataset):
    def __init__(self, dataset_dir, image_size=(200, 200)):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.image_files = sorted(os.listdir(os.path.join(dataset_dir, 'images')))
        self.keypoint_files = sorted(os.listdir(os.path.join(dataset_dir, 'keypoints')))

    def __len__(self):
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        img0_path = os.path.join(self.dataset_dir, 'images', self.image_files[idx])
        img1_path = os.path.join(self.dataset_dir, 'images', self.image_files[idx + 1])
        kp0_path = os.path.join(self.dataset_dir, 'keypoints', self.keypoint_files[idx])
        kp1_path = os.path.join(self.dataset_dir, 'keypoints', self.keypoint_files[idx + 1])

        img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, self.image_size)
        img1 = cv2.resize(img1, self.image_size)

        img0 = torch.from_numpy(img0).float().unsqueeze(0).unsqueeze(0) / 255.0
        img1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0) / 255.0

        with open(kp0_path, 'r') as f:
            kp0 = json.load(f)['keypoints']
        with open(kp1_path, 'r') as f:
            kp1 = json.load(f)['keypoints']

        return img0, img1, kp0, kp1

# === Tréningová funkcia ===
def train(model, dataloader, optimizer, device, num_epochs=5):
    model.train()
    preprocessor = LoFTRPreprocessor()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for img0, img1, kp0_dicts, kp1_dicts in dataloader:
            img0 = img0.squeeze(0).to(device)
            img1 = img1.squeeze(0).to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                data = {"image0": img0, "image1": img1}
                data = preprocessor(data)
                output = model(data)

            mkpts0 = output['keypoints0']
            mkpts1 = output['keypoints1']

            if len(mkpts0) == 0:
                continue

            kp0_names = [kp['name'] for kp in kp0_dicts]
            gt_kp1 = {kp['name']: np.array([kp['x'], kp['y']]) for kp in kp1_dicts}

            matched_targets = []
            ground_truths = []

            for name, pt0 in zip(kp0_names, mkpts0.cpu().numpy()):
                dists = np.linalg.norm(mkpts0.cpu().numpy() - pt0, axis=1)
                idx = np.argmin(dists)
                pt1 = mkpts1[idx].cpu().numpy()

                if name in gt_kp1:
                    matched_targets.append(pt1)
                    ground_truths.append(gt_kp1[name])

            if len(matched_targets) == 0:
                continue

            pred = torch.tensor(matched_targets, dtype=torch.float32, device=device)
            gt = torch.tensor(ground_truths, dtype=torch.float32, device=device)

            loss = nn.functional.mse_loss(pred, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss:.4f}")

# === Hlavná časť ===
if __name__ == "__main__":
    dataset_dir = "dataset"  # cesta k tvojmu datasetu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset a DataLoader
    dataset = KeypointMatchingDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Model a optimalizér
    model = LoFTR(config=LoFTRConfig()).to(device)
    state_dict = torch.load("loftr_outdoor.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Spustenie tréningu
    train(model, dataloader, optimizer, device, num_epochs=10)

    # Uloženie modelu
    torch.save(model.state_dict(), "loftr_finetuned.pth")
