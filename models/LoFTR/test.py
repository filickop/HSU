import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.LoFTR.Hourglass import HeatmapNet
from TowelDataset import ViCoSTowelHeatmapDataset


class BatchSizeTester:
    def __init__(self):
        self.config = {
            'dataset_dir': r"D:\SKOLA\HSU\ViCoSTowelDataset\ViCoSTowelDataset",
            'annotations_file': r"D:\SKOLA\HSU\ViCoSTowelDataset\ViCoSTowelDataset\annotations.json",
            'heatmap_size': 64,
            'sigma': 3,
            'output_stride': 4
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transforms()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_dataset(self):
        return ViCoSTowelHeatmapDataset(
            dataset_dir=self.config['dataset_dir'],
            annotations_file=self.config['annotations_file'],
            transform=self.transform,
            heatmap_size=self.config['heatmap_size'],
            sigma=self.config['sigma'],
            output_stride=self.config['output_stride'],
            printWarnings=False
        )

    def find_max_batch_size(self, start_size=4, max_tries=10):
        """Hlavná funkcia pre hľadanie maximálnej veľkosti dávky"""
        dataset = self._create_dataset()
        model = HeatmapNet(in_channels=3, num_keypoints=4, num_stacks=2).to(self.device)

        current_size = start_size
        best_size = start_size

        print(f"\nZačínam testovanie maximálnej veľkosti dávky na {self.device}...")

        for i in range(max_tries):
            try:
                # Vyčistenie GPU cache
                torch.cuda.empty_cache()

                # Vytvorenie dátového loaderu
                loader = DataLoader(dataset, batch_size=current_size, shuffle=False)

                # Načítanie jednej dávky
                batch = next(iter(loader))
                images = batch['image'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)

                # Testovanie forward a backward pass
                outputs = model(images)
                loss = torch.nn.MSELoss()(outputs[-1], heatmaps)
                loss.backward()

                print(f"✅ Veľkosť dávky {current_size} funguje")
                best_size = current_size

                # Zvýšenie veľkosti dávky
                current_size *= 2

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"❌ Veľkosť dávky {current_size} je príliš veľká")
                    current_size = (current_size + best_size) // 2
                    if current_size <= best_size:
                        break
                else:
                    raise e

        # Binárne vyhľadávanie pre presnejšie určenie
        low = best_size
        high = best_size * 2

        while low <= high and (high - low) > 1:
            mid = (low + high) // 2
            try:
                torch.cuda.empty_cache()
                loader = DataLoader(dataset, batch_size=mid, shuffle=False)
                batch = next(iter(loader))
                images = batch['image'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)
                outputs = model(images)
                loss = torch.nn.MSELoss()(outputs[-1], heatmaps)
                loss.backward()
                print(f"✅ Veľkosť dávky {mid} funguje")
                low = mid
                best_size = mid
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"❌ Veľkosť dávky {mid} je príliš veľká")
                    high = mid - 1
                else:
                    raise e

        print(f"\n🎯 Maximálna odporúčaná veľkosť dávky: {best_size}")
        return best_size


if __name__ == "__main__":
    tester = BatchSizeTester()
    max_batch_size = tester.find_max_batch_size(start_size=4)
    print(f"\nPre vašu konfiguráciu odporúčam použiť batch_size = {max_batch_size}")