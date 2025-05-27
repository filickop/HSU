import json
import os
import cv2
import torch


class PairKeypointDataset(torch.utils.data.Dataset):
    def __init__(self,ref_img, ref_json, json_path, image_dir, image_size=(200, 200)):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.ref_img = ref_img
        self.ref_json = ref_json
        self.image_dir = image_dir
        self.image_size = image_size

        with (open(ref_json, 'r')) as f:
            self.data2 = json.load(f)

        self.img0, self.kp0 = load_img_and_kps(self.data2[0], self.ref_img, self.image_size)


    def __len__(self):
        return len(self.data) - 1  # napr. susedné páry

    def __getitem__(self, idx):

        img1, kp1 = load_img_and_kps(self.data[idx + 1], self.image_dir, self.image_size)

        return {
            "image0": self.img0, "keypoints0": self.kp0,
            "image1": img1, "keypoints1": kp1
        }

def load_img_and_kps(entry, image_dir, image_size):
    img_path = os.path.join(image_dir, os.path.basename(entry["path"])).replace("\\", "/")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    original_height, original_width = img.shape[:2]
    target_width, target_height = image_size

    img = cv2.resize(img, image_size)
    img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    kps = torch.tensor([[kp["x"] * scale_x, kp["y"] * scale_y] for kp in entry["keypoints"]],dtype=torch.float32)
    return img_tensor, kps