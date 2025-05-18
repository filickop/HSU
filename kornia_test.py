import hydra
import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import kornia as K
from kornia.x import Configuration, ImageClassifierTrainer, ModelCheckpoint

from dataclasses import dataclass
from kornia.x import Configuration

@dataclass
class MyConfig(Configuration):
    train_data_path: str = "./dataset"
    val_data_path: str = "./dataset"
    image_size: int = 200
    batch_size: int = 16
    lr: float = 0.0001
    num_epochs: int = 20

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

class KeypointDataset(Dataset):
    def __init__(self, images_dir, keypoints_dir, transform=None):
        self.images_dir = images_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        keypoint_path = os.path.join(self.keypoints_dir, image_name.replace('.png', '.json'))

        image = Image.open(image_path).convert('RGB')
        with open(keypoint_path, 'r') as f:
            keypoints = json.load(f)

        if self.transform:
            image = self.transform(image)

        return {'input': image, 'target': keypoints}

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def my_app(config: MyConfig) -> None:

    # Definovanie transformácií
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ])

    train_dataset = KeypointDataset(
        images_dir=to_absolute_path(os.path.join(config.train_data_path, "images")),
        keypoints_dir=to_absolute_path(os.path.join(config.train_data_path, "keypoints")),
        transform=transform
    )

    val_dataset = KeypointDataset(
        images_dir=to_absolute_path(os.path.join(config.val_data_path, "images")),
        keypoints_dir=to_absolute_path(os.path.join(config.val_data_path, "keypoints")),
        transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Inicializácia modelu
    model = kornia.feature.LoFTR(pretrained=None).cuda()
    state_dict = torch.load("loftr_outdoor.pth", map_location='cuda')
    model.load_state_dict(state_dict)

    # Loss funkcia – potrebuješ prispôsobiť podľa výstupu
    criterion = nn.MSELoss()

    # Optimalizátor a scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs * len(train_dataloader)
    )

    # Augmentácie (voliteľné)
    _augmentations = nn.Sequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomAffine(degrees=10.0),
        K.augmentation.PatchSequential(
            K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.8),
            grid_size=(2, 2),
            patchwise_apply=False,
        ),
    )

    def augmentations(sample: dict) -> dict:
        out = _augmentations(sample["input"])
        return {"input": out, "target": sample["target"]}

    # Tu môžeš pokračovať v integrácii s Kornia Trainer, alebo ručne napísať tréningovú slučku


    def augmentations(sample: dict) -> dict:
        out = _augmentations(sample["input"])
        return {"input": out, "target": sample["target"]}

    # Nastavenie checkpointovania
    model_checkpoint = ModelCheckpoint(filepath="./outputs", monitor="top5")

    # Inicializácia trénera
    trainer = ImageClassifierTrainer(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        scheduler,
        config,
        callbacks={"augmentations": augmentations, "on_checkpoint": model_checkpoint},
    )

    # Spustenie tréningu
    trainer.fit()

if __name__ == "__main__":
    my_app()
