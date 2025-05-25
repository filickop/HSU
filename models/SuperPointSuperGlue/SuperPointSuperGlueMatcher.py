import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.SuperPointSuperGlue.matching import Matching
from models.SuperPointSuperGlue.utils import read_image


class SuperPointSuperGlueMatcher:
    def __init__(self, dataset_json_path, base_img_path):
        self.dataset_json_path = dataset_json_path
        self.base_img_path = base_img_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Načítaj dataset
        with open(self.dataset_json_path, "r") as f:
            self.dataset = json.load(f)

        # Inicializácia modelov
        self.matcher = Matching({
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1,
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.01,
            }
        }).eval().to(self.device)

        # Načítanie referenčného obrázka
        self.img1, self.inp1, _ = read_image(self.base_img_path, self.device, [-1], 0, False)
        if self.img1 is None:
            raise RuntimeError("Referenčný obrázok sa nenačítal!")

    def match(self, variant_index=0):
        entry = self.dataset[variant_index]
        img2_path = entry["path"]
        img2_full_path = os.path.join(os.path.dirname(self.dataset_json_path), img2_path)

        img2, inp2, _ = read_image(img2_full_path, self.device, [-1], 0, False)
        if img2 is None:
            raise RuntimeError(f"Obrázok {img2_path} sa nenačítal!")

        with torch.no_grad():
            pred = self.matcher({'image0': self.inp1, 'image1': inp2})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0_all = pred['keypoints0']
        kpts1_all = pred['keypoints1']
        matches = pred['matches0']

        valid = matches != -1
        matched_kpts0 = kpts0_all[valid]
        matched_kpts1 = kpts1_all[matches[valid]]

        return img2, matched_kpts0, matched_kpts1

    def visualize(self, img2, mkpt0, mkpt1):
        h0, w0 = self.img1.shape[:2]
        h1, w1 = img2.shape[:2]
        H = max(h0, h1)
        W = w0 + w1

        if self.img1.ndim == 2:
            canvas = np.zeros((H, W), dtype=np.uint8)
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

        canvas[:h0, :w0] = self.img1
        canvas[:h1, w0:] = img2

        plt.figure(figsize=(12, 8))
        plt.imshow(canvas, cmap='gray' if self.img1.ndim == 2 else None)

        if len(mkpt1) > 0:
            plt.plot([mkpt0[0, 0], mkpt1[0, 0] + w0], [mkpt0[0, 1], mkpt1[0, 1]], color='red', linewidth=1.5)
            plt.scatter(mkpt0[:, 0], mkpt0[:, 1], c='cyan', s=40, label='Image 0 selected point')
            plt.scatter(mkpt1[:, 0] + w0, mkpt1[:, 1], c='lime', s=40, label='Image 1 matched point')
        else:
            plt.scatter(mkpt0[:, 0], mkpt0[:, 1], c='cyan', s=40, label='Image 0 selected point')

        plt.axis('off')
        plt.title('Matched keypoints (SuperPoint + SuperGlue)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def select_nearest_keypoint(matched_kpts0, matched_kpts1, target_point):
        """
        Vyberie najbližší keypoint z matched_kpts0 k bodu target_point
        a vráti tento bod aj jeho match v matched_kpts1.

        :param matched_kpts0: np.ndarray tvaru (N, 2) – keypointy z obrázka 0
        :param matched_kpts1: np.ndarray tvaru (N, 2) – zodpovedajúce keypointy z obrázka 1
        :param target_point: Tuple[int, int] – súradnice bodu (x, y)
        :return: mkpt0 (1, 2), mkpt1 (1, 2) – vybraný pár keypointov
        """
        if matched_kpts0.shape[0] == 0:
            print("⚠️ Žiadne matched keypointy.")
            return np.empty((0, 2)), np.empty((0, 2))

        x, y = target_point
        dists = np.linalg.norm(matched_kpts0 - np.array([x, y]), axis=1)
        nearest_idx = np.argmin(dists)

        mkpt0 = matched_kpts0[nearest_idx:nearest_idx + 1]
        mkpt1 = matched_kpts1[nearest_idx:nearest_idx + 1]

        return mkpt0, mkpt1