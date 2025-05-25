import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import time

from Hourglass import HeatmapNet
from TowelDataset import ViCoSTowelHeatmapDataset


class Trainer:
    def __init__(self):
        self.config = {
            'batch_size': 160,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'heatmap_size': 64,
            'sigma': 3,
            'output_stride': 4,
            'dataset_dir': r"D:\SKOLA\HSU\ViCoSTowelDataset\ViCoSTowelDataset",
            'annotations_file': r"D:\SKOLA\HSU\ViCoSTowelDataset\ViCoSTowelDataset\annotations.json"
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transforms()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _calculate_accuracy(self, pred_heatmaps, true_heatmaps, threshold=3.0):
        """Vypočíta presnosť predikcie kľúčových bodov"""
        batch_size, num_keypoints, h, w = pred_heatmaps.shape

        pred_kps = torch.zeros(batch_size, num_keypoints, 2)
        true_kps = torch.zeros(batch_size, num_keypoints, 2)

        for b in range(batch_size):
            for k in range(num_keypoints):
                # Predikovaný kľúčový bod
                pred_idx = torch.argmax(pred_heatmaps[b, k])
                pred_y, pred_x = divmod(pred_idx.item(), w)
                pred_kps[b, k] = torch.tensor([pred_x, pred_y])

                # Skutočný kľúčový bod
                true_idx = torch.argmax(true_heatmaps[b, k])
                true_y, true_x = divmod(true_idx.item(), w)
                true_kps[b, k] = torch.tensor([true_x, true_y])

        # Vzdialenosť medzi predikovanými a skutočnými bodmi
        distances = torch.norm(pred_kps - true_kps, dim=2)

        # Presnosť (percento bodov v tolerancii)
        accuracy = (distances < threshold).float().mean()

        # Priemerná vzdialenosť v pixeloch
        mean_distance = distances.mean()

        return accuracy.item(), mean_distance.item()

    def _create_dataloader(self):
        """Vytvorí dátový loader pre trénovacie dáta"""
        try:
            dataset = ViCoSTowelHeatmapDataset(
                dataset_dir=self.config['dataset_dir'],
                annotations_file=self.config['annotations_file'],
                transform=self.transform,
                heatmap_size=self.config['heatmap_size'],
                sigma=self.config['sigma'],
                output_stride=self.config['output_stride'],
                printWarnings=False
            )

            print(f"Načítaných {len(dataset)} vzoriek z datasetu")
            return DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0
            )
        except Exception as e:
            print(f"Chyba pri načítaní dát: {e}")
            return None

    def train(self):
        """Hlavná trénovacia slučka"""
        dataloader = self._create_dataloader()
        if dataloader is None:
            return None

        # Inicializácia modelu
        self.model = HeatmapNet(
            in_channels=3,
            num_keypoints=4,
            num_stacks=2
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        print(f"Trénovanie na zariadení: {self.device}")
        print("Začíname trénovanie...")

        for epoch in range(self.config['num_epochs']):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_dist = 0.0

            start_time = time.time()
            for batch_idx, batch in enumerate(dataloader):

                try:
                    # Načítanie dát
                    images = batch['image'].to(self.device)
                    heatmaps = batch['heatmaps'].to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    final_output = outputs[-1]

                    # Výpočet straty
                    loss = self.criterion(final_output, heatmaps)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Výpočet metrík
                    batch_acc, batch_dist = self._calculate_accuracy(
                        final_output.detach(),
                        heatmaps.detach()
                    )

                    # Aktualizácia štatistík
                    epoch_loss += loss.item()
                    epoch_acc += batch_acc
                    epoch_dist += batch_dist

                    # Výpis každých 10 batchov
                    if batch_idx % 10 == 0:
                        end_time = time.time()
                        elapsed_time = (end_time - start_time) /60
                        print(
                            f"Epocha {epoch + 1}/{self.config['num_epochs']}, "
                            f"Batch {batch_idx}/{len(dataloader)}: "
                            f"Strata={loss.item():.4f}, "
                            f"Presnosť={batch_acc * 100:.1f}%, "
                            f"Vzdialenosť={batch_dist:.2f}px"
                            f"Čas {elapsed_time:.4f} minút"
                        )

                except Exception as e:
                    print(f"Chyba v dávke {batch_idx}: {e}")
                    continue

            # Výpis štatistík za epochu
            avg_loss = epoch_loss / len(dataloader)
            avg_acc = epoch_acc / len(dataloader)
            avg_dist = epoch_dist / len(dataloader)

            print("\n" + "=" * 70)
            print(f"EPOCHA {epoch + 1} SÚHRN:")
            print(f"Priemerná strata: {avg_loss:.4f}")
            print(f"Priemerná presnosť: {avg_acc * 100:.2f}%")
            print(f"Priemerná vzdialenosť: {avg_dist:.2f}px")
            print("=" * 70 + "\n")

        print("Trénovanie úspešne dokončené!")
        return self.model

    def save_model(self, path="towel_heatmap_model.pth"):
        """Uloží natrénovaný model"""
        if self.model:
            #torch.onnx.export(model=self.model, torch.ones(1,3,800,800))
            torch.save(self.model.state_dict(), path)
            print(f"Model úspešne uložený ako {path}")
        else:
            print("Žiadny model na uloženie!")


if __name__ == "__main__":
    trainer = Trainer()
    trained_model = trainer.train()

    if trained_model:
        trainer.save_model()