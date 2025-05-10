import torch
import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter


class ViCoSTowelHeatmapDataset(Dataset):
    printWarnings = False

    def __init__(self, dataset_dir, annotations_file, transform=None,
                 heatmap_size=64, sigma=3, output_stride=4, printWarnings=False):
        """
        Args:
            dataset_dir: Path to dataset directory
            annotations_file: Path to JSON annotations
            transform: Image transformations
            heatmap_size: Output heatmap size (square)
            sigma: Gaussian kernel standard deviation
            output_stride: Ratio of input to output resolution
        """

        ViCoSTowelHeatmapDataset.printWarnings = printWarnings

        self.dataset_dir = dataset_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.output_stride = output_stride

        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Filter out invalid entries
        self.valid_indices = [i for i, k in enumerate(self.annotations.keys())
                              if self._validate_path(k)]

    def _validate_path(self, file_path):
        """Check if all required files exist for a given sample"""
        path_parts = file_path.split('/')
        bg = path_parts[0]
        cloth = path_parts[1]

        rgb_path = os.path.join(self.dataset_dir, bg, cloth, 'rgb',
                                path_parts[-1].replace('.jpg', '.jpg'))
        return os.path.exists(rgb_path)

    def __len__(self):
        return len(self.valid_indices)

    def _generate_heatmaps(self, keypoints, img_size):
        """
        Generate Gaussian heatmaps for all keypoints
        Args:
            keypoints: Array of shape (N, 2) containing corner coordinates
            img_size: Original image dimensions (width, height)
        Returns:
            heatmaps: Tensor of shape (N, heatmap_size, heatmap_size)
        """
        heatmaps = np.zeros((len(keypoints), self.heatmap_size, self.heatmap_size),
                            dtype=np.float32)

        # Scale factor from original image to heatmap
        scale_x = self.heatmap_size / (img_size[0] / self.output_stride)
        scale_y = self.heatmap_size / (img_size[1] / self.output_stride)

        for i, kp in enumerate(keypoints):
            # Ensure we have exactly 2 coordinates (x, y)
            if len(kp) != 2:
                if ViCoSTowelHeatmapDataset.printWarnings:
                    print(f"Warning: Keypoint {i} has unexpected shape {kp.shape}, skipping")
                continue

            x, y = kp[0], kp[1]

            # Scale coordinates to heatmap space
            x_hm = x * scale_x
            y_hm = y * scale_y

            # Create base heatmap with peak at keypoint location
            xx, yy = np.meshgrid(np.arange(self.heatmap_size),
                                 np.arange(self.heatmap_size))
            heatmap = np.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2 * self.sigma ** 2))

            # Normalize to [0, 1]
            heatmap = heatmap / np.max(heatmap)
            heatmaps[i] = heatmap

        return heatmaps

    def __getitem__(self, idx):
        # Get actual index from valid indices
        real_idx = self.valid_indices[idx]
        file_path = list(self.annotations.keys())[real_idx]
        annotation = self.annotations[file_path]

        # Split path and construct full paths
        path_parts = file_path.split('/')
        bg = path_parts[0]
        cloth = path_parts[1]

        rgb_path = os.path.join(self.dataset_dir, bg, cloth, 'rgb', path_parts[-1].replace('.jpg', '.jpg'))
        depth_path = os.path.join(self.dataset_dir, bg, cloth, 'depth', path_parts[-1].replace('.jpg', '.npy'))
        mask_path = os.path.join(self.dataset_dir, bg, cloth, 'mask', path_parts[-1].replace('.jpg', '.png'))

        # Load images
        try:
            rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            if rgb_img is None:
                raise FileNotFoundError(f"RGB image not found at {rgb_path}")

            depth_img = np.load(depth_path)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Mask image not found at {mask_path}")
        except Exception as e:
            print(f"Error loading images for {file_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Skip to next sample

        # Get image dimensions
        img_height, img_width = rgb_img.shape[:2]
        img_size = (img_width, img_height)

        # Process keypoints - handle multiple formats
        try:
            keypoints = np.array(annotation['points'], dtype=np.float32)

            # Reshape based on input format
            if keypoints.size == 0:
                raise ValueError("Empty keypoints array")

            # Handle different input formats
            if keypoints.ndim == 1:
                # Flattened array - assume it's [x1,y1,x2,y2,...]
                if keypoints.size % 2 != 0:
                    if ViCoSTowelHeatmapDataset.printWarnings:
                        print(f"Warning: Flattened array has odd size {keypoints.size} for {file_path}")
                    keypoints = keypoints[:keypoints.size // 2 * 2]  # Truncate to even size
                keypoints = keypoints.reshape(-1, 2)
            elif keypoints.ndim == 2:
                # 2D array - handle (N,4), (N,2) cases
                if keypoints.shape[1] == 4:
                    if ViCoSTowelHeatmapDataset.printWarnings:
                        print(f"Note: Using first 2 values from {keypoints.shape} keypoints for {file_path}")
                    keypoints = keypoints[:, :2]  # Take only first 2 values (x,y)
                elif keypoints.shape[1] != 2:
                    raise ValueError(f"Unexpected keypoints shape {keypoints.shape}")
            else:
                raise ValueError(f"Keypoints has {keypoints.ndim} dimensions")

            # Ensure we have exactly 4 points (pad or truncate if needed)
            if len(keypoints) > 4:
                if ViCoSTowelHeatmapDataset.printWarnings:
                    print(f"Warning: Truncating {len(keypoints)} points to 4 for {file_path}")
                keypoints = keypoints[:4]
            elif len(keypoints) < 4:
                if ViCoSTowelHeatmapDataset.printWarnings:
                    print(f"Warning: Padding {len(keypoints)} points to 4 for {file_path}")
                # Pad with last valid point or image corners
                while len(keypoints) < 4:
                    keypoints = np.vstack([keypoints, keypoints[-1]])

        except Exception as e:
            print(f"Error processing keypoints for {file_path}: {e}")
            # Create default keypoints (image corners) as fallback
            keypoints = np.array([
                [0, 0], [img_width - 1, 0],
                [img_width - 1, img_height - 1], [0, img_height - 1]
            ], dtype=np.float32)

        # Generate heatmaps with safety checks
        try:
            heatmaps = np.zeros((4, self.heatmap_size, self.heatmap_size), dtype=np.float32)

            scale_x = self.heatmap_size / (img_size[0] / self.output_stride)
            scale_y = self.heatmap_size / (img_size[1] / self.output_stride)

            for i, (x, y) in enumerate(keypoints):
                x_hm = x * scale_x
                y_hm = y * scale_y

                # Create grid
                xx, yy = np.meshgrid(np.arange(self.heatmap_size),
                                     np.arange(self.heatmap_size))

                # Calculate 2D Gaussian
                heatmap = np.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2 * self.sigma ** 2))

                # Safe normalization
                max_val = np.max(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val
                else:
                    heatmap = np.zeros_like(heatmap)

                heatmaps[i] = heatmap

        except Exception as e:
            print(f"Error generating heatmaps for {file_path}: {e}")
            heatmaps = np.zeros((4, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        # Apply transformations to RGB image
        if self.transform:
            try:
                rgb_img = self.transform(rgb_img)
            except Exception as e:
                print(f"Error transforming image {file_path}: {e}")
                rgb_img = torch.zeros(3, 256, 256)  # Fallback

        # Convert to tensors
        sample = {
            'image': rgb_img,
            'depth': torch.from_numpy(depth_img).float().unsqueeze(0),
            'mask': torch.from_numpy(mask_img).float().unsqueeze(0),
            'heatmaps': torch.from_numpy(heatmaps).float(),
            'keypoints': torch.from_numpy(keypoints).float(),
            'image_path': file_path
        }

        return sample