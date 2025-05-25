from torch.utils.data import Dataset
import os
import cv2
import json
import random
import torch
from kornia.feature.loftr import LoFTR
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def init_model():
    global device, model, optimizer, loss_fn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicializujeme model s danou konfiguráciou
    model = LoFTR(pretrained=None).to(device)

    # Načítame uložené váhy
    state_dict = torch.load("loftr_outdoor.pth", map_location=device)
    model.load_state_dict(state_dict)

    # Teraz je model pripravený na tréning s načítanými váhami
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()


def generate_random_pairs(image_dir, count=100):
    """
    Vygeneruje náhodné páry obrázkov na základe ich mien (bez prípony .png).

    :param image_dir: cesta k priečinku s obrázkami
    :param count: počet náhodných párov, ktoré sa majú vygenerovať
    :return: zoznam dvojíc ako [("note_000", "note_042"), ...]
    """
    # Získaj všetky mená súborov bez prípony
    image_files = [f[:-4] for f in os.listdir(image_dir) if f.endswith(".png")]

    # Over, že máme dosť obrázkov
    if len(image_files) < 2:
        raise ValueError("Potrebujeme aspoň 2 obrázky na vytvorenie párov.")

    pairs = []
    for _ in range(count):
        pair = random.sample(image_files, 2)
        pairs.append((pair[0], pair[1]))

    return pairs

def model_train(num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Prenesieme na device len potrebné dáta pre model
            input_dict = {
                'image0': batch['image0'].to(device),  # (B, 1, H, W)
                'image1': batch['image1'].to(device),
            }

            # Forward pass - len s obrazmi
            output = model(input_dict)

            # Extrahuj predikované kľúčové body
            mkpts0 = output['keypoints0']  # Tensor, shape (N, 2)
            mkpts1 = output['keypoints1']

            # Z ground truth vyrob tensor kľúčových bodov (batch size=1)
            gt_kps0 = batch['keypoints0'][0]  # list of tuples (name, x, y)
            gt_kps1 = batch['keypoints1'][0]

            gt_pts0 = torch.tensor([[kp[1], kp[2]] for kp in gt_kps0], dtype=torch.float32).to(device)
            gt_pts1 = torch.tensor([[kp[1], kp[2]] for kp in gt_kps1], dtype=torch.float32).to(device)

            # Ak model nevygeneroval žiadne keypoints, preskočíme batch
            if mkpts0.shape[0] == 0 or gt_pts0.shape[0] == 0:
                continue

            # Orežeme na minimálnu dĺžku pre bezpečný loss
            min_len = min(len(gt_pts0), len(gt_pts1), mkpts0.shape[0])

            # Loss na párované body (tu MSE na polohy predikcií a GT)
            loss = loss_fn(mkpts0[:min_len], gt_pts0[:min_len]) + loss_fn(mkpts1[:min_len], gt_pts1[:min_len])

            # Backpropagation a optimalizácia
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Avg loss = {total_loss / len(loader):.4f}")




class KeypointMatchingDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir, pairs, image_size=(200, 200)):
        self.image_dir = image_dir
        self.keypoint_dir = keypoint_dir
        self.pairs = pairs  # zoznam dvojíc ako [("note_000", "note_001"), ...]
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def load_image(self, filename):
        path = os.path.join(self.image_dir, filename + ".png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img = torch.from_numpy(img).float()[None] / 255.0
        return img

    def load_keypoints(self, filename):
        path = os.path.join(self.keypoint_dir, filename + ".json")
        with open(path, 'r') as f:
            data = json.load(f)
        return [(kp['name'], kp['x'], kp['y']) for kp in data['keypoints']]

    def __getitem__(self, idx):
        fname0, fname1 = self.pairs[idx]
        image0 = self.load_image(fname0)
        image1 = self.load_image(fname1)
        keypoints0 = self.load_keypoints(fname0)
        keypoints1 = self.load_keypoints(fname1)

        matches = [(i, i) for i in range(len(keypoints0))]
        spv_b_ids = torch.zeros(len(matches), dtype=torch.long)  # batch indexy, ak batch=1, všetko nula
        spv_query_inds = torch.tensor([m[0] for m in matches], dtype=torch.long)
        spv_gt_inds = torch.tensor([m[1] for m in matches], dtype=torch.long)

        return {
            'image0': image0,
            'image1': image1,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'spv_b_ids': spv_b_ids,  # tensor [N]
            'spv_query_inds': spv_query_inds,
            'spv_gt_inds': spv_gt_inds,
        }


init_model()
pairs = generate_random_pairs("../../dataset/Note/images", count=100)
dataset = KeypointMatchingDataset("../../dataset/Note/images", "dataset/Note/keypoints", pairs)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model_train(1)