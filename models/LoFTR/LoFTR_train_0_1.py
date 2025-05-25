import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import kornia


# ========== Načítanie obrázka ==========
def load_image(path, size=(200, 200)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    return img


# ========== Načítanie keypointov ==========
def load_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [(kp['name'], kp['x'], kp['y']) for kp in data['keypoints']]


# ========== Dataset trieda ==========
class KeypointPairDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.kp_dir = os.path.join(root_dir, 'keypoints')
        self.filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.filenames) - 1  # páry

    def __getitem__(self, idx):
        fname1 = self.filenames[idx]
        fname2 = self.filenames[idx + 1]

        img1_path = os.path.join(self.images_dir, fname1)
        img2_path = os.path.join(self.images_dir, fname2)
        kp1_path = os.path.join(self.kp_dir, fname1.replace('.png', '.json'))
        kp2_path = os.path.join(self.kp_dir, fname2.replace('.png', '.json'))

        image0 = load_image(img1_path)
        image1 = load_image(img2_path)

        keypoints0 = load_keypoints(kp1_path)
        keypoints1 = load_keypoints(kp2_path)

        return {
            'image0': image0,
            'image1': image1,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1
        }


# ========== Tréningová slučka ==========
def train(model, dataloader, optimizer, device, epochs=1):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataloader:
            image0 = batch['image0'].squeeze(0).to(device)
            image1 = batch['image1'].squeeze(0).to(device)

            input_dict = {
                "image0": image0,
                "image1": image1
            }

            optimizer.zero_grad()
            out = model(input_dict)

            if 'mkpts0_f' in out and 'mkpts1_f' in out:
                loss = torch.tensor(0.0, device=device)  # placeholder, môžeš nahradiť vlastnou stratou
                loss.backward()
                optimizer.step()


# ========== Hlavný skript ==========
if __name__ == '__main__':
    dataset_path = 'D:/SKOLA/HSU/SEM/dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = KeypointPairDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = kornia.feature.LoFTR(pretrained='outdoor').to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, dataloader, optimizer, device, epochs=10)
