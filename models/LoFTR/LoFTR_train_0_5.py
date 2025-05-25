import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import kornia.feature as KF
from tqdm import tqdm

# === Dataset ===
class LoFTRKeypointDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir):
        self.image_dir = image_dir
        self.keypoint_dir = keypoint_dir
        self.image_names = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
        self.transform = T.Compose([
            T.Resize((200, 200)),
            T.Grayscale(),  # convert to 1 channel
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names) - 1

    def __getitem__(self, idx):
        name0 = self.image_names[idx]
        name1 = self.image_names[idx + 1]
        img0 = self.transform(Image.open(os.path.join(self.image_dir, name0)))
        img1 = self.transform(Image.open(os.path.join(self.image_dir, name1)))
        kp0 = self._load_keypoints(name0)
        kp1 = self._load_keypoints(name1)
        return {'image0': img0, 'image1': img1, 'keypoints0': kp0, 'keypoints1': kp1}

    def _load_keypoints(self, image_filename):
        json_path = os.path.join(self.keypoint_dir, image_filename.replace('.png', '.json'))
        with open(json_path, 'r') as f:
            data = json.load(f)
        keypoints = [(kp["x"] / 200 * 200, kp["y"] / 200 * 200) for kp in data["keypoints"]]
        return torch.tensor(keypoints, dtype=torch.float32)

# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LoFTRKeypointDataset("../../dataset/Note/images", "dataset/Note/keypoints")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = KF.LoFTR(pretrained='outdoor').to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for i, batch in enumerate(tqdm(dataloader)):
        img0 = batch['image0'].to(device)
        img1 = batch['image1'].to(device)
        kp0 = batch['keypoints0'].squeeze(0).to(device)
        kp1 = batch['keypoints1'].squeeze(0).to(device)

        print(f"\n[Batch {i}] img0: {img0.shape}, img1: {img1.shape}")
        print(f"keypoints0: {kp0.shape}, keypoints1: {kp1.shape}")

        if kp0.shape != kp1.shape or kp0.numel() == 0:
            print("⚠️ Skipping due to empty or mismatched keypoints")
            continue

        N = kp0.shape[0]
        input_dict = {
            'image0': img0,
            'image1': img1,
            'spv_pos_b_ids': torch.zeros((1, N), dtype=torch.long, device=device),
            'spv_pos_gt': kp1.unsqueeze(0),
            'spv_pos_mask': torch.ones((1, N), dtype=torch.bool, device=device),
            'spv_b_ids': torch.zeros((1, N), dtype=torch.long, device=device),
        }

        try:
            output = model(input_dict)
            if 'keypoints0' in output and 'keypoints1' in output:
                loss = loss_fn(output['keypoints0'], output['keypoints1'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"✅ Loss: {loss.item():.4f}")
            else:
                print("⚠️ Output missing keypoints")
        except Exception as e:
            print(f"❌ Error: {e}")
            break

    torch.save(model.state_dict(), "loftr_finetuned.pth")
    print("✅ Model saved to 'loftr_finetuned.pth'")

if __name__ == "__main__":
    main()
